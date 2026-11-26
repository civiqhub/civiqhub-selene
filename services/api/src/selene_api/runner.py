# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Run one computation in a child process, and stop it for real when needed.

:class:`Runner` is the narrow interface the job manager sees: run a prepared task within
a time limit and return its outcome. :class:`ProcessRunner` starts
``python -m selene_api.worker`` in its own process group, writes the task on its
standard input and reads one JSON document from its standard output, up to a size
limit. On a timeout, a cancellation or an oversized output, it ends the whole process
group (SIGTERM, then SIGKILL after a grace period) and waits for the process before it
returns: when ``run`` returns or raises, no computation process of the task is left, so
the slot can be given again. Nothing the process writes on its error stream is read:
it may hold an identifier of the input.
"""

import asyncio
import contextlib
import json
import os
import signal
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = ["ProcessRunner", "RunOutcome", "Runner", "worker_environment"]

_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_INHERITED = ("PATH", "LANG", "LC_ALL", "TMPDIR", "PYTHONPATH", "HOME", "VIRTUAL_ENV")
_READ_CHUNK = 1 << 16


@dataclass(frozen=True)
class RunOutcome:
    """How a computation ended.

    Attributes:
        result: The contract result, when the engine answered (a value or an
            explained absence); ``None`` otherwise.
        error_code: ``timeout``, ``worker_crashed``, ``result_too_large``,
            ``engine_error``, ``memory_exceeded`` or ``interrupted`` when there is no
            result.
        retryable: Whether the same content may succeed on a new submission.
        peak_rss_bytes: The peak resident memory the process reported, if any.
    """

    result: dict[str, Any] | None = None
    error_code: str | None = None
    retryable: bool = False
    peak_rss_bytes: int | None = None


class Runner(Protocol):
    """What the job manager needs to run a computation."""

    async def run(self, task: Mapping[str, Any], *, timeout: float) -> RunOutcome:
        """Run ``task`` (``{"kind", "schema_version", "payload"}``) within ``timeout`` seconds.

        Cancelling the call stops the computation; the call returns or raises only once
        its process has ended.
        """
        ...

    async def describe(self, *, timeout: float) -> dict[str, Any]:
        """Start a computation process once and return what it says of the engine."""
        ...


def worker_environment(base: Mapping[str, str]) -> dict[str, str]:
    """Return the environment of a computation process: few variables, one thread."""
    environment = {name: base[name] for name in _INHERITED if name in base}
    environment.update(dict.fromkeys(_THREAD_VARIABLES, "1"))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    return environment


class ProcessRunner:
    """Run each computation in a fresh ``python -m selene_api.worker`` process.

    Args:
        max_result_bytes: Largest output read from the process.
        memory_bytes: Address-space limit of the process, or ``None`` for none.
        cpu_seconds: CPU-time limit of the process, a backstop behind the timeout, or
            ``None`` for none.
        kill_grace_seconds: Time between SIGTERM and SIGKILL when the process is stopped.
        command: The command that starts the process; defaults to this interpreter.
        environment: The environment the process starts from; defaults to this one.
    """

    def __init__(
        self,
        *,
        max_result_bytes: int,
        memory_bytes: int | None = None,
        cpu_seconds: int | None = None,
        kill_grace_seconds: float = 2.0,
        command: Sequence[str] = (sys.executable, "-m", "selene_api.worker"),
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.max_result_bytes = max_result_bytes
        self.kill_grace_seconds = kill_grace_seconds
        self.command = tuple(command)
        self.environment = worker_environment(os.environ if environment is None else environment)
        self.environment["SELENE_WORKER_PARENT"] = str(os.getpid())
        if memory_bytes is not None:
            self.environment["SELENE_WORKER_MEMORY_BYTES"] = str(memory_bytes)
        if cpu_seconds is not None:
            self.environment["SELENE_WORKER_CPU_SECONDS"] = str(cpu_seconds)
        # Called with timeout, result_too_large or cancelled when a process is stopped.
        self.on_stop: Callable[[str], None] | None = None

    async def _start(self, *arguments: str) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            *self.command,
            *arguments,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=self.environment,
            start_new_session=True,
            close_fds=True,
        )

    async def _started(self, *arguments: str) -> asyncio.subprocess.Process:
        """Start a process; if cancelled meanwhile, stop it before letting the cancel through."""
        starting = asyncio.ensure_future(self._start(*arguments))
        try:
            return await asyncio.shield(starting)
        except asyncio.CancelledError:
            process = await starting
            await self._stop(process)
            raise

    async def _stop(self, process: asyncio.subprocess.Process) -> None:
        """End the process group of ``process`` and wait for the process."""
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), self.kill_grace_seconds)
        # The group may hold descendants even after the leader ended.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        await process.wait()

    async def _exchange(self, process: asyncio.subprocess.Process, data: bytes | None) -> bytes:
        assert process.stdin is not None
        assert process.stdout is not None
        if data is not None:
            process.stdin.write(data)
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                await process.stdin.drain()
        process.stdin.close()
        received = bytearray()
        while chunk := await process.stdout.read(_READ_CHUNK):
            received += chunk
            if len(received) > self.max_result_bytes:
                raise _TooLargeError
        await process.wait()
        return bytes(received)

    async def run(self, task: Mapping[str, Any], *, timeout: float) -> RunOutcome:
        data = json.dumps(task, allow_nan=False).encode("utf-8")
        process = await self._started()
        try:
            output = await asyncio.wait_for(self._exchange(process, data), timeout)
        except TimeoutError:
            self._stopped("timeout")
            return RunOutcome(error_code="timeout", retryable=False)
        except _TooLargeError:
            self._stopped("result_too_large")
            return RunOutcome(error_code="result_too_large", retryable=False)
        except asyncio.CancelledError:
            self._stopped("cancelled")
            raise
        finally:
            await asyncio.shield(self._stop(process))
        return self._outcome(process.returncode, output)

    def _stopped(self, reason: str) -> None:
        if self.on_stop is not None:
            self.on_stop(reason)

    def _outcome(self, returncode: int | None, output: bytes) -> RunOutcome:
        if returncode != 0:
            return RunOutcome(error_code="worker_crashed", retryable=True)
        try:
            answer = json.loads(output)
        except ValueError:
            return RunOutcome(error_code="worker_crashed", retryable=True)
        if not isinstance(answer, dict):
            return RunOutcome(error_code="worker_crashed", retryable=True)
        if answer.get("outcome") == "result" and isinstance(answer.get("result"), dict):
            peak = answer.get("peak_rss_bytes")
            return RunOutcome(
                result=answer["result"], peak_rss_bytes=peak if isinstance(peak, int) else None
            )
        if answer.get("outcome") == "error" and answer.get("code") in {
            "engine_error",
            "memory_exceeded",
        }:
            return RunOutcome(error_code=answer["code"], retryable=False)
        return RunOutcome(error_code="worker_crashed", retryable=True)

    async def describe(self, *, timeout: float) -> dict[str, Any]:
        process = await self._started("--describe")
        try:
            output = await asyncio.wait_for(self._exchange(process, None), timeout)
        finally:
            await asyncio.shield(self._stop(process))
        if process.returncode != 0:
            raise RuntimeError(f"the computation process ended with {process.returncode}")
        answer = json.loads(output)
        if not isinstance(answer, dict):
            raise RuntimeError("the computation process did not describe the engine")
        return answer


class _TooLargeError(Exception):
    """The process wrote more than ``max_result_bytes``."""
