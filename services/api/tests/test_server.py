# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""``python -m selene_api`` starts only when configured, serves contract v1 and stops cleanly."""

import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator

import pytest
from servicetest import TOKEN

import make_fixtures
import selene_contract

EX_CONFIG = 78
# After a graceful shutdown, Uvicorn re-raises the signal it caught, so that the process
# manager sees an exit on SIGTERM; the log says whether the shutdown was graceful.
STOPPED_BY_SIGTERM = (0, -signal.SIGTERM)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


def environment(tmp_path: pathlib.Path, **variables: str) -> dict[str, str]:
    token_file = tmp_path / "token"
    token_file.write_text(TOKEN + "\n", encoding="ascii")
    base = {name: value for name, value in os.environ.items() if not name.startswith("SELENE_")}
    base.update(
        {
            "SELENE_TOKEN_FILE": str(token_file),
            "SELENE_HOST": "127.0.0.1",
            "SELENE_PORT": str(free_port()),
            "SELENE_METRICS_PORT": str(free_port()),
        }
    )
    base.update(variables)
    return base


def start(env: dict[str, str], log: pathlib.Path) -> subprocess.Popen[bytes]:
    with log.open("wb") as output:
        return subprocess.Popen(
            [sys.executable, "-m", "selene_api"], env=env, stdout=output, stderr=subprocess.STDOUT
        )


def wait_ready(
    client: selene_contract.SeleneClient, process: subprocess.Popen[bytes], log: pathlib.Path
) -> None:
    deadline = time.monotonic() + 60
    while True:
        try:
            if client.readiness().status == 200:
                return
        except OSError:
            pass
        if process.poll() is not None or time.monotonic() > deadline:
            raise AssertionError(log.read_text())
        time.sleep(0.1)


@pytest.mark.parametrize(
    ("variables", "message"),
    [
        ({"SELENE_TOKEN_FILE": ""}, "SELENE_TOKEN_FILE is required"),
        ({"SELENE_TOKEN_FILE": "/nonexistent/token"}, "does not name a readable file"),
        ({"SELENE_HOST": "0.0.0.0"}, "without TLS the service only listens on the loopback"),
        ({"SELENE_TOKN_FILE": "/x"}, "unknown setting SELENE_TOKN_FILE"),
        ({"SELENE_MAX_VOTERS": "50000"}, "SELENE_MAX_VOTERS is between 1 and 20000"),
        ({"SELENE_JOB_TIMEOUT_SECONDS": "soon"}, "SELENE_JOB_TIMEOUT_SECONDS is an integer"),
    ],
)
def test_a_bad_configuration_stops_the_start(
    tmp_path: pathlib.Path, variables: dict[str, str], message: str
) -> None:
    env = environment(tmp_path, **variables)
    if variables.get("SELENE_TOKEN_FILE") == "":
        del env["SELENE_TOKEN_FILE"]
    answer = subprocess.run(
        [sys.executable, "-m", "selene_api"], env=env, capture_output=True, text=True, timeout=60
    )
    assert answer.returncode == EX_CONFIG
    assert message in answer.stderr
    assert TOKEN not in answer.stderr + answer.stdout


def test_a_short_token_stops_the_start(tmp_path: pathlib.Path) -> None:
    env = environment(tmp_path)
    pathlib.Path(env["SELENE_TOKEN_FILE"]).write_text("short-token\n", encoding="ascii")
    answer = subprocess.run(
        [sys.executable, "-m", "selene_api"], env=env, capture_output=True, text=True, timeout=60
    )
    assert answer.returncode == EX_CONFIG
    assert "shorter than 32 characters" in answer.stderr
    assert "short-token" not in answer.stderr


@pytest.fixture
def server(
    tmp_path: pathlib.Path,
) -> Iterator[tuple[selene_contract.SeleneClient, subprocess.Popen[bytes], pathlib.Path]]:
    env = environment(tmp_path, SELENE_ENVIRONMENT="test-only")
    log = tmp_path / "server.log"
    process = start(env, log)
    client = selene_contract.SeleneClient(
        f"http://127.0.0.1:{env['SELENE_PORT']}", TOKEN, allow_insecure_http=True, timeout=10
    )
    try:
        wait_ready(client, process, log)
        yield client, process, log
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_the_server_serves_contract_v1(
    server: tuple[selene_contract.SeleneClient, subprocess.Popen[bytes], pathlib.Path],
) -> None:
    client, process, log = server
    reports = selene_contract.run_check(client, wait_timeout=120)
    assert [report for report in reports if not report["passed"]] == []
    assert client.service().body["deployment"]["environment"] == "test-only"

    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=30) in STOPPED_BY_SIGTERM
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    events = [line["event"] for line in lines]
    assert events[0] == "service_starting"
    assert "service_ready" in events
    assert "job_finished" in events
    assert events[-1] == "service_stopped"
    assert all(line["level"] != "error" for line in lines)


def test_the_logs_carry_no_token_no_body_and_no_identifier(
    server: tuple[selene_contract.SeleneClient, subprocess.Popen[bytes], pathlib.Path],
) -> None:
    client, process, log = server
    selene_contract.run_check(client, wait_timeout=120)
    client.with_token("Bearer-like-but-wrong-token-1234567890-abc").service()
    process.send_signal(signal.SIGTERM)
    process.wait(timeout=30)
    text = log.read_text()
    payload = make_fixtures.reference_submission()["payload"]
    forbidden = [
        TOKEN,
        "Bearer-like-but-wrong",
        "someone@example.org",
        "test-only:check",
        "test-only/generation-1",
        payload["scope"]["project"],
        *payload["voters"],
        *(item["id"] for item in payload["items"]),
        *(item["revision"] for item in payload["items"]),
        "shown_without_vote",
        "group_memberships",
        "Traceback",
    ]
    leaked = [value for value in forbidden if value in text]
    assert leaked == []


def slow_submission() -> dict[str, object]:
    """A synthetic submission that computes for a few seconds, within the default budget."""
    body = make_fixtures.reference_submission()
    payload = body["payload"]
    body["submission_key"] = "test-only:slow:0001"
    payload["voters"] = [f"vtr-{number:05d}" for number in range(2000)]
    payload["items"] = [{"id": f"itm-{number:03d}", "revision": "r1"} for number in range(120)]
    payload["votes"] = [
        [row, column, (row * 7 + column * 3) % 5 - 2]
        for row in range(2000)
        for column in range(120)
    ]
    payload["exposure"] = {"mode": "complete"}
    payload["parameters"] = {"n_groups": 3, "n_blocks": 3, "initialization": "random"}
    return body


def test_sigterm_during_a_computation_leaves_no_process(tmp_path: pathlib.Path) -> None:
    env = environment(tmp_path)
    log = tmp_path / "server.log"
    process = start(env, log)
    try:
        client = selene_contract.SeleneClient(
            f"http://127.0.0.1:{env['SELENE_PORT']}", TOKEN, allow_insecure_http=True, timeout=30
        )
        wait_ready(client, process, log)
        answer = client.submit(slow_submission())
        assert answer.status == 202, answer.body
        deadline = time.monotonic() + 10
        children: list[str] = []
        while not children and time.monotonic() < deadline:
            children = subprocess.run(
                ["pgrep", "-f", "selene_api.worker"], capture_output=True, text=True, check=False
            ).stdout.split()
            time.sleep(0.05)
        assert children, "the computation process should be running"
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=30) in STOPPED_BY_SIGTERM
        assert log.read_text().splitlines()[-1].endswith('"event":"service_stopped"}')
        time.sleep(0.5)
        left = [pid for pid in children if pathlib.Path(f"/proc/{pid}").exists()]
        assert left == []
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
