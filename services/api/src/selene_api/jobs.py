# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Volatile tracking of jobs: admission, states, idempotency and bounded retention.

The :class:`JobManager` holds every job in memory and nowhere else: a restart forgets
them, and the contract says so. It admits one computation at a time:

- a submission whose key is bound to a retained job with the same content gets that
  job back, even while busy; with other content, it is a conflict;
- otherwise, if the slot is taken, it is refused as busy: there is no queue;
- otherwise the slot is reserved before anything is awaited, so two concurrent
  submissions never start two computations, and the computation starts in the
  background through the :class:`~selene_api.runner.Runner`.

The slot is given back only when the runner has returned, that is once the computation
process has ended. A finished job is kept until its expiry, or until room is needed for
newer ones; the running job is never evicted. Inputs are not kept: once the computation
process has read them, only the content fingerprint remains.
"""

import asyncio
import contextlib
import secrets
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from selene_api.adapters import PreparedTask
from selene_api.runner import Runner, RunOutcome

__all__ = ["BusyError", "ConflictError", "Job", "JobManager", "ShuttingDownError", "Submitted"]

State = Literal["running", "succeeded", "failed"]

_MESSAGES = {
    "timeout": "The computation ran longer than the time limit and was stopped.",
    "worker_crashed": "The computation process ended abnormally.",
    "interrupted": "The instance was shutting down; the computation was stopped.",
    "result_too_large": "The outcome of the computation is larger than the limit.",
    "engine_error": "The engine refused the input.",
    "memory_exceeded": "The computation needed more memory than the limit of this instance.",
}


class BusyError(Exception):
    """The computation slot is taken."""


class ConflictError(Exception):
    """The key is bound to a retained job with other content."""


class ShuttingDownError(Exception):
    """The manager no longer admits computations."""


@dataclass
class Job:
    """One job and its temporary outcome. Mutated by the manager only."""

    id: str
    kind: str
    schema_version: int
    submission_key: str
    caller_reference: str | None
    content_fingerprint: str
    created_at: datetime
    state: State = "running"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    expires_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    expires_monotonic: float | None = None
    task: "asyncio.Task[None] | None" = field(default=None, repr=False)

    def document(self, versions: dict[str, str]) -> dict[str, Any]:
        """Return the job as contract v1 writes it."""
        return {
            "id": self.id,
            "kind": self.kind,
            "schema_version": self.schema_version,
            "submission_key": self.submission_key,
            "caller_reference": self.caller_reference,
            "content_fingerprint": self.content_fingerprint,
            "state": self.state,
            "created_at": _timestamp(self.created_at),
            "started_at": _timestamp(self.started_at),
            "finished_at": _timestamp(self.finished_at),
            "expires_at": _timestamp(self.expires_at),
            "result": self.result,
            "error": self.error,
            "service": versions,
        }


def _timestamp(moment: datetime | None) -> str | None:
    if moment is None:
        return None
    return moment.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class Submitted:
    """The answer to a submission: the job, and whether it is a replay."""

    job: Job
    replayed: bool


def _new_id() -> str:
    return "job_" + secrets.token_urlsafe(18)


class JobManager:
    """Admit one computation at a time and keep jobs in memory for a bounded time.

    Args:
        runner: Runs a computation in a supervised process.
        job_timeout_seconds: Time limit of a computation.
        result_ttl_seconds: How long a finished job is kept at most.
        max_retained_jobs: How many finished jobs are kept at most.
        slots: Computations run at a time. Contract v1 runs one.
        clock: Monotonic clock, for expiry decisions.
        now: Wall clock, for the timestamps of the jobs.
        on_finish: Called with each job when it finishes, for metrics and logs.
    """

    def __init__(
        self,
        runner: Runner,
        *,
        job_timeout_seconds: float,
        result_ttl_seconds: float,
        max_retained_jobs: int,
        slots: int = 1,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        on_finish: Callable[[Job, RunOutcome | None], None] | None = None,
    ) -> None:
        if slots != 1:
            raise ValueError("contract v1 runs one computation at a time")
        self._runner = runner
        self.job_timeout_seconds = job_timeout_seconds
        self.result_ttl_seconds = result_ttl_seconds
        self.max_retained_jobs = max_retained_jobs
        self.slots = slots
        self._clock = clock
        self._now = now
        self._on_finish = on_finish
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._by_key: dict[str, str] = {}
        self._running: Job | None = None
        self._accepting = True

    # Reading ---------------------------------------------------------------------------

    @property
    def running(self) -> int:
        """Computations running now."""
        return 0 if self._running is None else 1

    @property
    def accepting(self) -> bool:
        """Whether a new submission would start now."""
        return self._accepting and self._running is None

    @property
    def held(self) -> int:
        """Jobs held in memory now, without expiring any: safe to read from another thread."""
        return len(self._jobs)

    @property
    def retained(self) -> int:
        """Jobs held in memory, running or finished."""
        self._expire()
        return len(self._jobs)

    def get(self, job_id: str) -> Job | None:
        """Return a retained job, or ``None`` when it is unknown, expired or deleted."""
        self._expire()
        return self._jobs.get(job_id)

    # Writing ---------------------------------------------------------------------------

    def submit(self, task: PreparedTask) -> Submitted:
        """Admit a prepared task: replay, conflict, busy, or a new running job.

        Synchronous on purpose: the checks and the reservation of the slot happen
        without any await between them.

        Raises:
            ConflictError: The key is bound to a retained job with other content.
            ShuttingDownError: The manager no longer admits computations.
            BusyError: The slot is taken.
        """
        self._expire()
        known = self._by_key.get(task.submission_key)
        if known is not None:
            job = self._jobs[known]
            if job.content_fingerprint != task.content_fingerprint:
                raise ConflictError
            return Submitted(job, replayed=True)
        if not self._accepting:
            raise ShuttingDownError
        if self._running is not None:
            raise BusyError
        moment = self._now()
        job = Job(
            id=_new_id(),
            kind=task.kind,
            schema_version=task.schema_version,
            submission_key=task.submission_key,
            caller_reference=task.caller_reference,
            content_fingerprint=task.content_fingerprint,
            created_at=moment,
            started_at=moment,
        )
        self._running = job
        self._jobs[job.id] = job
        self._by_key[job.submission_key] = job.id
        job.task = asyncio.get_running_loop().create_task(
            self._execute(job, task.worker_input()), name=f"selene-{job.id}"
        )
        return Submitted(job, replayed=False)

    async def _execute(self, job: Job, task: dict[str, Any]) -> None:
        outcome: RunOutcome | None = None
        try:
            outcome = await self._runner.run(task, timeout=self.job_timeout_seconds)
        except asyncio.CancelledError:
            # Deleted or shutting down: the runner has ended the process.
            self._finish(job, RunOutcome(error_code="interrupted", retryable=True))
            raise
        except Exception:
            outcome = RunOutcome(error_code="worker_crashed", retryable=True)
            self._finish(job, outcome)
        else:
            self._finish(job, outcome)

    def _finish(self, job: Job, outcome: RunOutcome) -> None:
        moment = self._now()
        job.finished_at = moment
        job.expires_at = moment + timedelta(seconds=self.result_ttl_seconds)
        job.expires_monotonic = self._clock() + self.result_ttl_seconds
        if outcome.result is not None:
            job.state = "succeeded"
            job.result = outcome.result
        else:
            code = outcome.error_code or "worker_crashed"
            job.state = "failed"
            job.error = {
                "code": code,
                "message": _MESSAGES.get(code, "The computation failed."),
                "retryable": outcome.retryable,
            }
        job.task = None
        if self._running is job:
            self._running = None
        if self._on_finish is not None:
            self._on_finish(job, outcome)
        self._expire()

    async def delete(self, job_id: str) -> bool:
        """Forget a job; a running computation is stopped first. Return whether it existed."""
        self._expire()
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.task is not None:
            job.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await job.task
        self._forget(job)
        return True

    async def shutdown(self) -> None:
        """Refuse new computations, stop the running one and forget every job."""
        self._accepting = False
        running = self._running
        if running is not None and running.task is not None:
            running.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await running.task
        self._jobs.clear()
        self._by_key.clear()

    # Retention -------------------------------------------------------------------------

    def _forget(self, job: Job) -> None:
        self._jobs.pop(job.id, None)
        if self._by_key.get(job.submission_key) == job.id:
            del self._by_key[job.submission_key]

    def _expire(self) -> None:
        moment = self._clock()
        for job in list(self._jobs.values()):
            if job.expires_monotonic is not None and job.expires_monotonic <= moment:
                self._forget(job)
        finished = [job for job in self._jobs.values() if job.state != "running"]
        # The oldest finished jobs go first; the running job is never evicted.
        finished.sort(key=lambda job: job.finished_at or job.created_at)
        while len(finished) > self.max_retained_jobs:
            self._forget(finished.pop(0))
