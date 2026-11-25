# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""The job manager admits one computation, replays, refuses and forgets as the contract says."""

import asyncio
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from servicetest import FakeRunner, settle

from selene_api.adapters import PreparedTask
from selene_api.jobs import BusyError, ConflictError, JobManager, ShuttingDownError
from selene_api.runner import RunOutcome


def task(key: str = "test-only:key:0001", fingerprint: str = "sha256:" + "a" * 64) -> PreparedTask:
    return PreparedTask(
        kind="coclustering.single",
        schema_version=1,
        submission_key=key,
        caller_reference=None,
        payload={"seed": 1},
        content_fingerprint=fingerprint,
    )


class Clock:
    def __init__(self) -> None:
        self.seconds = 1000.0
        self.wall = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)

    def advance(self, seconds: float) -> None:
        self.seconds += seconds
        self.wall += timedelta(seconds=seconds)


def manager(runner: FakeRunner, clock: Clock, **settings: Any) -> JobManager:
    options: dict[str, Any] = {
        "job_timeout_seconds": 60.0,
        "result_ttl_seconds": 900.0,
        "max_retained_jobs": 4,
    }
    options.update(settings)
    return JobManager(runner, clock=lambda: clock.seconds, now=lambda: clock.wall, **options)


def run(scenario: Callable[[], Coroutine[Any, Any, None]]) -> None:
    asyncio.run(scenario())


def test_a_submission_runs_then_succeeds() -> None:
    async def scenario() -> None:
        runner, clock = FakeRunner(), Clock()
        jobs = manager(runner, clock)
        submitted = jobs.submit(task())
        job = submitted.job
        assert (submitted.replayed, job.state, jobs.running) == (False, "running", 1)
        assert job.document({"service_version": "x"})["expires_at"] is None
        await settle(lambda: job.state != "running")
        assert job.state == "succeeded"
        assert job.result == {"schema": "fake", "available": True}
        assert job.error is None
        assert job.expires_at == clock.wall + timedelta(seconds=900)
        assert jobs.running == 0
        assert runner.started == [task().worker_input()]
        assert runner.timeouts == [60.0]

    run(scenario)


def test_the_same_key_and_content_replays_even_while_busy() -> None:
    async def scenario() -> None:
        runner = FakeRunner()
        runner.hold = True
        jobs = manager(runner, Clock())
        first = jobs.submit(task())
        second = jobs.submit(task())
        assert second.replayed
        assert second.job is first.job
        assert len(runner.started) <= 1
        runner.release()
        await settle(lambda: first.job.state == "succeeded")
        third = jobs.submit(task())
        assert third.replayed
        assert third.job is first.job
        assert len(runner.started) == 1

    run(scenario)


def test_the_same_key_with_other_content_is_a_conflict() -> None:
    async def scenario() -> None:
        runner = FakeRunner()
        runner.hold = True
        jobs = manager(runner, Clock())
        jobs.submit(task())
        with pytest.raises(ConflictError):
            jobs.submit(task(fingerprint="sha256:" + "b" * 64))
        runner.release()

    run(scenario)


def test_another_submission_while_running_is_busy_and_starts_nothing() -> None:
    async def scenario() -> None:
        runner = FakeRunner()
        runner.hold = True
        jobs = manager(runner, Clock())
        first = jobs.submit(task())
        with pytest.raises(BusyError):
            jobs.submit(task(key="test-only:key:0002", fingerprint="sha256:" + "b" * 64))
        await settle(lambda: len(runner.started) == 1)
        assert jobs.retained == 1
        runner.release()
        await settle(lambda: first.job.state == "succeeded")
        second = jobs.submit(task(key="test-only:key:0002", fingerprint="sha256:" + "b" * 64))
        assert not second.replayed

    run(scenario)


def test_concurrent_submissions_never_start_two_computations() -> None:
    async def scenario() -> None:
        runner = FakeRunner()
        runner.hold = True
        jobs = manager(runner, Clock())

        async def one(number: int) -> str:
            try:
                jobs.submit(
                    task(
                        key=f"test-only:key:{number:04d}", fingerprint="sha256:" + f"{number:064d}"
                    )
                )
            except BusyError:
                return "busy"
            return "started"

        answers = await asyncio.gather(*(one(number) for number in range(20)))
        assert answers.count("started") == 1
        await settle(lambda: len(runner.started) == 1)
        await asyncio.sleep(0.05)
        assert len(runner.started) == 1
        runner.release()

    run(scenario)


def test_the_slot_is_free_only_once_the_runner_returned() -> None:
    async def scenario() -> None:
        runner = FakeRunner()
        runner.hold = True
        jobs = manager(runner, Clock())
        jobs.submit(task())
        await settle(lambda: len(runner.started) == 1)
        assert not jobs.accepting
        runner.release()
        await settle(lambda: jobs.accepting)

    run(scenario)


@pytest.mark.parametrize(
    ("outcome", "code", "retryable"),
    [
        (RunOutcome(error_code="timeout"), "timeout", False),
        (RunOutcome(error_code="worker_crashed", retryable=True), "worker_crashed", True),
        (RunOutcome(error_code="result_too_large"), "result_too_large", False),
        (RunOutcome(error_code="engine_error"), "engine_error", False),
    ],
)
def test_a_failure_says_why_and_whether_to_retry(
    outcome: RunOutcome, code: str, retryable: bool
) -> None:
    async def scenario() -> None:
        jobs = manager(FakeRunner(outcome), Clock())
        job = jobs.submit(task()).job
        await settle(lambda: job.state != "running")
        assert job.state == "failed"
        assert job.result is None
        assert job.error is not None
        assert (job.error["code"], job.error["retryable"]) == (code, retryable)
        assert "seed" not in job.error["message"]

    run(scenario)


def test_an_unexpected_runner_error_is_a_crash() -> None:
    class Broken(FakeRunner):
        async def run(self, task: Any, *, timeout: float) -> RunOutcome:
            raise OSError("cannot start")

    async def scenario() -> None:
        jobs = manager(Broken(), Clock())
        job = jobs.submit(task()).job
        await settle(lambda: job.state != "running")
        assert job.error == {
            "code": "worker_crashed",
            "message": "The computation process ended abnormally.",
            "retryable": True,
        }
        assert jobs.accepting

    run(scenario)


def test_a_finished_job_expires_and_frees_its_key() -> None:
    async def scenario() -> None:
        clock = Clock()
        jobs = manager(FakeRunner(), clock, result_ttl_seconds=60.0)
        job = jobs.submit(task()).job
        await settle(lambda: job.state == "succeeded")
        clock.advance(59)
        assert jobs.get(job.id) is job
        clock.advance(2)
        assert jobs.get(job.id) is None
        again = jobs.submit(task(fingerprint="sha256:" + "c" * 64))
        assert not again.replayed

    run(scenario)


def test_the_oldest_finished_jobs_make_room_but_never_the_running_one() -> None:
    async def scenario() -> None:
        clock = Clock()
        runner = FakeRunner()
        jobs = manager(runner, clock, max_retained_jobs=2)
        finished = []
        for number in range(3):
            job = jobs.submit(task(key=f"test-only:key:{number:04d}")).job
            await settle(lambda job=job: job.state == "succeeded")
            finished.append(job)
            clock.advance(1)
        assert [jobs.get(job.id) is not None for job in finished] == [False, True, True]
        runner.hold = True
        running = jobs.submit(task(key="test-only:key:0099")).job
        assert jobs.get(running.id) is running
        assert jobs.retained == 3
        runner.release()
        await settle(lambda: running.state == "succeeded")
        assert jobs.retained == 2
        assert jobs.get(running.id) is running

    run(scenario)


def test_deleting_a_running_job_stops_its_computation() -> None:
    async def scenario() -> None:
        runner = FakeRunner()
        runner.hold = True
        jobs = manager(runner, Clock())
        job = jobs.submit(task()).job
        await settle(lambda: len(runner.started) == 1)
        assert await jobs.delete(job.id)
        assert runner.cancelled == 1
        assert jobs.get(job.id) is None
        assert jobs.accepting
        assert not jobs.submit(task()).replayed

    run(scenario)


def test_deleting_an_unknown_job_says_so() -> None:
    async def scenario() -> None:
        jobs = manager(FakeRunner(), Clock())
        assert not await jobs.delete("job_" + "x" * 22)

    run(scenario)


def test_shutdown_stops_the_computation_and_forgets_everything() -> None:
    async def scenario() -> None:
        runner = FakeRunner()
        runner.hold = True
        jobs = manager(runner, Clock())
        job = jobs.submit(task()).job
        await settle(lambda: len(runner.started) == 1)
        await jobs.shutdown()
        assert runner.cancelled == 1
        assert jobs.get(job.id) is None
        assert jobs.retained == 0
        with pytest.raises(ShuttingDownError):
            jobs.submit(task(key="test-only:key:0002"))

    run(scenario)


def test_one_slot_only() -> None:
    with pytest.raises(ValueError, match="one computation"):
        JobManager(
            FakeRunner(), job_timeout_seconds=1, result_ttl_seconds=1, max_retained_jobs=1, slots=2
        )


def test_job_identifiers_are_opaque_and_unique() -> None:
    async def scenario() -> None:
        jobs = manager(FakeRunner(), Clock(), max_retained_jobs=100)
        seen = set()
        for number in range(50):
            job = jobs.submit(task(key=f"test-only:key:{number:04d}")).job
            await settle(lambda job=job: job.state != "running")
            assert job.id.startswith("job_")
            seen.add(job.id)
        assert len(seen) == 50

    run(scenario)
