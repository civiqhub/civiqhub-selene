# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Entry point of the computation process: ``python -m selene_api.worker``.

The server starts one such process per job and reads nothing else from it than one JSON
document on its standard output:

- input, on standard input: ``{"kind", "schema_version", "payload"}``, the normalised
  payload of an adapter;
- output: ``{"outcome": "result", "result": {...}}`` or
  ``{"outcome": "error", "code": "engine_error", "exception": "<class name>"}``.

With ``--describe``, it reads nothing and writes the identity of the engine and of each
adapter method; the server runs it once at start-up to check that a computation process
can start, before it declares itself ready.

Before it reads anything, the process hardens itself (Linux):

- the numerical libraries are limited to one thread before they are imported;
- no core dump (it would hold the votes) and no ptrace by other processes of the user;
- an address-space limit (``SELENE_WORKER_MEMORY_BYTES``): an allocation beyond it
  fails with ``MemoryError``, reported as ``memory_exceeded``, instead of putting the
  whole pod at risk;
- a CPU-time limit (``SELENE_WORKER_CPU_SECONDS``) as a last resort behind the timeout
  of the server;
- it dies with the server: ``PR_SET_PDEATHSIG`` sends it SIGKILL if the server ends,
  and it exits at once if the server is already gone (``SELENE_WORKER_PARENT``).

The engine message of an exception is never written: it may name an identifier of the
input.
"""

import os

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

import ctypes  # noqa: E402 - the thread limits come before any numerical import
import json  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
from typing import Any, BinaryIO  # noqa: E402

__all__ = ["EX_ORPHANED", "harden", "main"]

EX_ORPHANED = 70
"""Exit status when the server that started the process is already gone."""

_PR_SET_PDEATHSIG = 1
_PR_SET_DUMPABLE = 4


def _limit(kind: int, value: int) -> None:
    _soft, hard = resource.getrlimit(kind)
    if hard != resource.RLIM_INFINITY:
        value = min(value, hard)
    resource.setrlimit(kind, (value, value if hard == resource.RLIM_INFINITY else hard))


def harden(environ: dict[str, str]) -> None:
    """Apply the limits of a computation process, before it reads its input.

    Raises:
        SystemExit: With ``EX_ORPHANED`` if the server that started it is gone.
    """
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    memory = environ.get("SELENE_WORKER_MEMORY_BYTES")
    if memory:
        _limit(resource.RLIMIT_AS, int(memory))
    cpu = environ.get("SELENE_WORKER_CPU_SECONDS")
    if cpu:
        _limit(resource.RLIMIT_CPU, int(cpu))
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    parent = environ.get("SELENE_WORKER_PARENT")
    if parent and os.getppid() != int(parent):
        raise SystemExit(EX_ORPHANED)


def _peak_rss_bytes() -> int:
    # ru_maxrss is in kibibytes on Linux.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _describe() -> dict[str, Any]:
    import numpy

    import civiqhub_selene
    from selene_api.adapters import ADAPTERS

    # A product of matrices starts the threads of the linear algebra library, if any.
    numpy.ones((256, 256)) @ numpy.ones((256, 256))
    threads = len(os.listdir("/proc/self/task")) if sys.platform == "linux" else None
    return {
        "engine_version": civiqhub_selene.__version__,
        "numpy_version": numpy.__version__,
        "python_version": platform.python_version(),
        "threads": threads,
        "kinds": [
            {"kind": kind, "schema_version": version, "method": adapter.describe()}
            for (kind, version), adapter in sorted(ADAPTERS.items())
        ],
    }


def _compute(document: dict[str, Any]) -> dict[str, Any]:
    from civiqhub_selene.errors import SeleneError
    from selene_api.adapters import adapter_for

    adapter = adapter_for(document["kind"], document["schema_version"])
    if adapter is None:
        return {"outcome": "error", "code": "engine_error", "exception": "UnsupportedKind"}
    try:
        result = adapter.compute(document["payload"])
    except SeleneError as error:
        return {"outcome": "error", "code": "engine_error", "exception": type(error).__name__}
    except MemoryError:
        return {"outcome": "error", "code": "memory_exceeded", "exception": "MemoryError"}
    return {"outcome": "result", "result": result, "peak_rss_bytes": _peak_rss_bytes()}


def main(argv: list[str] | None = None) -> int:
    """Run one computation, or describe the engine, and write one JSON document."""
    arguments = sys.argv[1:] if argv is None else argv
    harden(dict(os.environ))
    output: BinaryIO = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    # Anything a library prints goes to the error stream, never into the protocol.
    sys.stdout = sys.stderr
    if arguments == ["--describe"]:
        answer = _describe()
    elif not arguments:
        try:
            answer = _compute(json.loads(sys.stdin.buffer.read()))
        except MemoryError:
            answer = {"outcome": "error", "code": "memory_exceeded", "exception": "MemoryError"}
    else:
        return 2
    output.write(json.dumps(answer, allow_nan=False).encode("utf-8"))
    output.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
