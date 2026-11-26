# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""The computation process is bounded, cannot outlive the server and runs one thread."""

import asyncio
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import Any

import pytest

import make_fixtures
from selene_api.adapters import InstanceLimits
from selene_api.adapters.coclustering_single import CoclusteringSingleV1
from selene_api.runner import ProcessRunner, worker_environment
from selene_api.worker import EX_ORPHANED
from servicetest import FakeRunner, service_for, settings_for

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux process controls")

WIDE = InstanceLimits(max_voters=20_000, max_items=5_000, max_cells=5_000_000, max_work=10**14)
PROBE = """
import ctypes, json, os, resource, sys
from selene_api.worker import harden
harden(dict(os.environ))
libc = ctypes.CDLL(None)
print(json.dumps({
    "core": resource.getrlimit(resource.RLIMIT_CORE),
    "address_space": resource.getrlimit(resource.RLIMIT_AS),
    "cpu": resource.getrlimit(resource.RLIMIT_CPU),
    "dumpable": libc.prctl(3, 0, 0, 0, 0),
}))
"""


def alive(pid: int) -> bool:
    status = pathlib.Path(f"/proc/{pid}/stat")
    try:
        return status.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def test_the_process_applies_its_limits() -> None:
    env = worker_environment(os.environ) | {
        "SELENE_WORKER_MEMORY_BYTES": str(512 * 1024 * 1024),
        "SELENE_WORKER_CPU_SECONDS": "150",
        "SELENE_WORKER_PARENT": str(os.getpid()),
    }
    answer = subprocess.run(
        [sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, check=True
    )
    limits = json.loads(answer.stdout)
    assert limits["core"] == [0, 0]
    assert limits["address_space"] == [512 * 1024 * 1024, 512 * 1024 * 1024]
    assert limits["cpu"][0] == 150
    assert limits["dumpable"] == 0


def test_a_process_whose_server_is_gone_exits_at_once() -> None:
    env = worker_environment(os.environ) | {"SELENE_WORKER_PARENT": "1"}
    answer = subprocess.run([sys.executable, "-m", "selene_api.worker", "--describe"], env=env)
    assert answer.returncode == EX_ORPHANED


def test_the_process_dies_with_its_server(tmp_path: pathlib.Path) -> None:
    pid_file = tmp_path / "pid"
    child = (
        "import os, time; from selene_api.worker import harden; harden(dict(os.environ)); "
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(600)"
    )
    parent = (
        "import os, subprocess, sys, time\n"
        "env = dict(os.environ, SELENE_WORKER_PARENT=str(os.getpid()))\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}], env=env)\n"
        "time.sleep(600)\n"
    )
    server = subprocess.Popen([sys.executable, "-c", parent])
    try:
        deadline = time.monotonic() + 30
        while not (pid_file.exists() and pid_file.read_text()) and time.monotonic() < deadline:
            time.sleep(0.05)
        worker = int(pid_file.read_text())
        assert alive(worker)
        server.send_signal(signal.SIGKILL)
        server.wait()
        deadline = time.monotonic() + 5
        while alive(worker) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not alive(worker)
    finally:
        if server.poll() is None:
            server.kill()
            server.wait()


def large_task() -> dict[str, Any]:
    """A synthetic task whose arrays need far more than 300 MiB."""
    body = make_fixtures.reference_submission()
    payload = body["payload"]
    payload["voters"] = [f"vtr-{number:05d}" for number in range(4000)]
    payload["items"] = [{"id": f"itm-{number:04d}", "revision": "r1"} for number in range(1000)]
    payload["votes"] = [[row, row % 1000, row % 5 - 2] for row in range(4000)]
    payload["exposure"] = {"mode": "complete"}
    payload["parameters"] = {"n_groups": 2, "n_blocks": 2, "n_starts": 1, "max_iterations": 1}
    return CoclusteringSingleV1().prepare(json.dumps(body).encode(), WIDE).worker_input()


def test_a_computation_beyond_the_memory_limit_is_reported_not_fatal() -> None:
    runner = ProcessRunner(max_result_bytes=10**8, memory_bytes=300 * 1024 * 1024)
    outcome = asyncio.run(runner.run(large_task(), timeout=120))
    assert (outcome.error_code, outcome.retryable) == ("memory_exceeded", False)


def test_the_same_computation_runs_under_a_larger_limit() -> None:
    runner = ProcessRunner(max_result_bytes=10**8, memory_bytes=4 * 1024 * 1024 * 1024)
    outcome = asyncio.run(runner.run(large_task(), timeout=300))
    assert outcome.error_code is None
    assert outcome.result is not None


def test_the_computation_process_runs_one_thread() -> None:
    description = asyncio.run(ProcessRunner(max_result_bytes=10**6).describe(timeout=120))
    assert description["threads"] == 1


def test_a_computation_process_with_more_threads_is_refused(tmp_path: pathlib.Path) -> None:
    class Threaded(FakeRunner):
        async def describe(self, *, timeout: float) -> dict[str, Any]:
            return {**await super().describe(timeout=timeout), "threads": 8}

    service = service_for(settings_for(tmp_path), Threaded())
    with pytest.raises(RuntimeError, match="more than one thread"):
        asyncio.run(service.start())
