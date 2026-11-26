# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""The process runner computes in a child process and stops it for real."""

import asyncio
import json
import os
import pathlib
import sys
import time
from typing import Any

import pytest

import make_fixtures
from selene_api.adapters import InstanceLimits
from selene_api.adapters.coclustering_single import CoclusteringSingleV1
from selene_api.runner import ProcessRunner, worker_environment

WIDE = InstanceLimits(max_voters=20_000, max_items=5_000, max_cells=2_000_000, max_work=10**13)


def reference_task() -> dict[str, Any]:
    submission = make_fixtures.reference_submission()
    return CoclusteringSingleV1().prepare(json.dumps(submission).encode(), WIDE).worker_input()


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie is ended; only its parent can reap it.
    status = pathlib.Path(f"/proc/{pid}/stat")
    return not status.exists() or status.read_text().split()[2] != "Z"


def script(tmp_path: pathlib.Path, body: str) -> list[str]:
    path = tmp_path / "fake_worker.py"
    path.write_text(body, encoding="utf-8")
    return [sys.executable, str(path)]


def test_the_child_process_computes_what_the_adapter_computes() -> None:
    task = reference_task()
    outcome = asyncio.run(ProcessRunner(max_result_bytes=10**7).run(task, timeout=120))
    assert outcome.error_code is None
    assert outcome.result == CoclusteringSingleV1().compute(task["payload"])
    assert outcome.peak_rss_bytes is not None
    assert outcome.peak_rss_bytes > 0


def test_the_child_process_describes_the_engine() -> None:
    from civiqhub_selene import __version__

    description = asyncio.run(ProcessRunner(max_result_bytes=10**6).describe(timeout=120))
    assert description["engine_version"] == __version__
    assert description["kinds"] == [
        {
            "kind": "coclustering.single",
            "schema_version": version,
            "method": {"name": "latent_block_model", "version": "0.1.0"},
        }
        for version in (1, 2)
    ]


def test_the_child_runs_with_one_numerical_thread() -> None:
    environment = worker_environment(
        {"PATH": "/usr/bin", "SELENE_TOKEN_FILE": "/secret", "HOME": "/h"}
    )
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"
    assert environment["MKL_NUM_THREADS"] == "1"
    assert "SELENE_TOKEN_FILE" not in environment


def test_an_engine_refusal_is_an_engine_error() -> None:
    task = reference_task()
    task["payload"] = json.loads(json.dumps(task["payload"]))
    task["payload"]["votes"].append(list(task["payload"]["votes"][0]))  # bypasses prepare
    outcome = asyncio.run(ProcessRunner(max_result_bytes=10**7).run(task, timeout=120))
    assert (outcome.error_code, outcome.retryable, outcome.result) == ("engine_error", False, None)


def test_a_crash_is_reported(tmp_path: pathlib.Path) -> None:
    runner = ProcessRunner(
        max_result_bytes=1000, command=script(tmp_path, "import sys; sys.exit(3)")
    )
    outcome = asyncio.run(runner.run({"kind": "k"}, timeout=10))
    assert (outcome.error_code, outcome.retryable) == ("worker_crashed", True)


def test_a_killed_process_is_a_crash(tmp_path: pathlib.Path) -> None:
    body = "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"
    runner = ProcessRunner(max_result_bytes=1000, command=script(tmp_path, body))
    outcome = asyncio.run(runner.run({"kind": "k"}, timeout=10))
    assert outcome.error_code == "worker_crashed"


def test_unreadable_output_is_a_crash(tmp_path: pathlib.Path) -> None:
    runner = ProcessRunner(max_result_bytes=1000, command=script(tmp_path, "print('not json')"))
    outcome = asyncio.run(runner.run({"kind": "k"}, timeout=10))
    assert outcome.error_code == "worker_crashed"


def test_an_oversized_output_is_refused(tmp_path: pathlib.Path) -> None:
    body = "import sys, time; sys.stdout.write('x' * 5000); sys.stdout.flush(); time.sleep(60)"
    runner = ProcessRunner(max_result_bytes=1000, command=script(tmp_path, body))
    started = time.monotonic()
    outcome = asyncio.run(runner.run({"kind": "k"}, timeout=30))
    assert outcome.error_code == "result_too_large"
    assert time.monotonic() - started < 10


def test_a_timeout_stops_the_process_and_its_descendants(tmp_path: pathlib.Path) -> None:
    pids = tmp_path / "pids"
    body = (
        "import os, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        f"open({str(pids)!r}, 'w').write(f'{{os.getpid()}} {{child.pid}}')\n"
        "time.sleep(600)\n"
    )
    runner = ProcessRunner(
        max_result_bytes=1000, kill_grace_seconds=0.5, command=script(tmp_path, body)
    )
    started = time.monotonic()
    outcome = asyncio.run(runner.run({"kind": "k"}, timeout=2))
    elapsed = time.monotonic() - started
    assert outcome.error_code == "timeout"
    assert elapsed < 6, elapsed
    leader, descendant = (int(pid) for pid in pids.read_text().split())
    deadline = time.monotonic() + 5
    while alive(descendant) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not alive(leader)
    assert not alive(descendant)


def test_a_process_that_ignores_sigterm_is_killed(tmp_path: pathlib.Path) -> None:
    body = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)"
    runner = ProcessRunner(
        max_result_bytes=1000, kill_grace_seconds=0.5, command=script(tmp_path, body)
    )
    started = time.monotonic()
    outcome = asyncio.run(runner.run({"kind": "k"}, timeout=1))
    assert outcome.error_code == "timeout"
    assert time.monotonic() - started < 5


def test_cancelling_a_run_ends_its_process(tmp_path: pathlib.Path) -> None:
    pids = tmp_path / "pid"
    body = f"import os, time; open({str(pids)!r}, 'w').write(str(os.getpid())); time.sleep(600)"
    runner = ProcessRunner(
        max_result_bytes=1000, kill_grace_seconds=0.5, command=script(tmp_path, body)
    )

    async def scenario() -> int:
        running = asyncio.create_task(runner.run({"kind": "k"}, timeout=600))
        while not pids.exists() or not pids.read_text():
            await asyncio.sleep(0.02)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        return int(pids.read_text())

    pid = asyncio.run(scenario())
    assert not alive(pid)
