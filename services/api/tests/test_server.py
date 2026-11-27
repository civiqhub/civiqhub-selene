# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""``python -m selene_api`` starts only when configured, serves contract v1 and stops cleanly."""

import json
import pathlib
import subprocess
import sys
import time

import pytest

import make_fixtures
import selene_contract
from servicetest import (
    STOPPED_BY_SIGTERM,
    TOKEN,
    environment,
    free_port,
    running_server,
    slow_submission,
)

EX_CONFIG = 78


@pytest.mark.parametrize(
    ("variables", "message"),
    [
        ({"SELENE_TOKEN_FILE": ""}, "SELENE_TOKEN_FILE or SELENE_TOKEN is required"),
        ({"SELENE_TOKEN_FILE": "/nonexistent/token"}, "does not name a readable file"),
        ({"SELENE_HOST": "0.0.0.0"}, "without TLS the service only listens on the loopback"),
        ({"SELENE_TOKN_FILE": "/x"}, "unknown setting SELENE_TOKN_FILE"),
        ({"SELENE_MAX_VOTERS": "50000"}, "SELENE_MAX_VOTERS is between 1 and 20000"),
        ({"SELENE_JOB_TIMEOUT_SECONDS": "soon"}, "SELENE_JOB_TIMEOUT_SECONDS is an integer"),
        ({"SELENE_WORKER_MEMORY_BYTES": "1000"}, "SELENE_WORKER_MEMORY_BYTES is between"),
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


def test_the_server_serves_contract_v1(tmp_path: pathlib.Path) -> None:
    with running_server(tmp_path, SELENE_ENVIRONMENT="test-only") as server:
        client = server.client()
        reports = selene_contract.run_check(client, wait_timeout=120)
        assert [report for report in reports if not report["passed"]] == []
        assert client.service().body["deployment"]["environment"] == "test-only"
        assert server.stop() in STOPPED_BY_SIGTERM
        lines = [json.loads(line) for line in server.log.read_text().splitlines()]
    events = [line["event"] for line in lines]
    assert events[0] == "service_starting"
    assert "service_ready" in events
    assert "job_finished" in events
    assert events[-1] == "service_stopped"
    assert all(line["level"] != "error" for line in lines)


def test_the_logs_carry_no_token_no_body_and_no_identifier(tmp_path: pathlib.Path) -> None:
    with running_server(tmp_path) as server:
        client = server.client()
        selene_contract.run_check(client, wait_timeout=120)
        client.with_token("Bearer-like-but-wrong-token-1234567890-abc").service()
        client.request(
            "POST",
            "/v1/jobs",
            raw_body=b'{"voters": ["leak-me-please"',
            content_type="application/json",
        )
        server.stop()
        text = server.log.read_text()
    payload = make_fixtures.reference_submission()["payload"]
    forbidden = [
        TOKEN,
        "Bearer-like-but-wrong",
        "leak-me-please",
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


def test_sigterm_during_a_computation_leaves_no_process(tmp_path: pathlib.Path) -> None:
    with running_server(tmp_path) as server:
        answer = server.client().submit(slow_submission())
        assert answer.status == 202, answer.body
        deadline = time.monotonic() + 10
        workers: list[int] = []
        while not workers and time.monotonic() < deadline:
            workers = server.worker_pids()
            time.sleep(0.05)
        assert workers, "the computation process should be running"
        started = time.monotonic()
        assert server.stop() in STOPPED_BY_SIGTERM
        assert time.monotonic() - started < 15
        assert server.log.read_text().splitlines()[-1].endswith('"event":"service_stopped"}')
    time.sleep(0.5)
    assert [pid for pid in workers if pathlib.Path(f"/proc/{pid}").exists()] == []


CONTRACT_CLIENT = pathlib.Path(__file__).resolve().parents[1] / "contract" / "selene_contract.py"
ENV_TOKEN = "test-token-abcdefghijklmnopqrstuvwxyzabcdef"
OTHER_TOKEN = "test-token-zyxwvutsrqponmlkjihgfedcbaabcdef"
FILE_OTHER = "test-token-fileotherabcdefghijklmnopqrstuvwxyz"


def _start_output(tmp_path: pathlib.Path, **variables: str) -> subprocess.CompletedProcess[str]:
    env = environment(tmp_path, **variables)
    if variables.get("SELENE_TOKEN_FILE") == "":
        del env["SELENE_TOKEN_FILE"]
    return subprocess.run(
        [sys.executable, "-m", "selene_api"], env=env, capture_output=True, text=True, timeout=60
    )


def test_an_invalid_tls_terminated_by_proxy_stops_the_start(tmp_path: pathlib.Path) -> None:
    answer = _start_output(tmp_path, SELENE_TLS_TERMINATED_BY_PROXY="yes")
    assert answer.returncode == EX_CONFIG
    assert "SELENE_TLS_TERMINATED_BY_PROXY" in answer.stderr
    assert TOKEN not in answer.stderr + answer.stdout


def test_proxy_mode_with_tls_files_stops_the_start(tmp_path: pathlib.Path) -> None:
    cert = tmp_path / "tls.crt"
    key = tmp_path / "tls.key"
    cert.write_text("cert")
    key.write_text("key")
    answer = _start_output(
        tmp_path,
        SELENE_TLS_TERMINATED_BY_PROXY="true",
        SELENE_TLS_CERT_FILE=str(cert),
        SELENE_TLS_KEY_FILE=str(key),
    )
    assert answer.returncode == EX_CONFIG
    assert "SELENE_TLS_TERMINATED_BY_PROXY" in answer.stderr
    assert TOKEN not in answer.stderr + answer.stdout


def test_proxy_mode_serves_http_off_loopback(tmp_path: pathlib.Path) -> None:
    with running_server(
        tmp_path, SELENE_TLS_TERMINATED_BY_PROXY="true", SELENE_HOST="0.0.0.0"
    ) as server:
        assert server.client().health().status == 200
        client = subprocess.run(
            [
                sys.executable,
                str(CONTRACT_CLIENT),
                "run",
                "--url",
                server.url,
                "--allow-insecure-http",
                "--token-file",
                server.env["SELENE_TOKEN_FILE"],
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert client.returncode == 0, client.stdout + client.stderr
        assert TOKEN not in client.stdout + client.stderr
        text = server.metrics()
        assert TOKEN not in text
        server.stop()
        log = server.log.read_text()
    lines = [json.loads(line) for line in log.splitlines() if line.startswith("{")]
    starting = next(line for line in lines if line.get("event") == "service_starting")
    assert starting["tls_terminated_by_proxy"] is True
    assert starting["tls"] is False
    assert TOKEN not in log


def test_proxy_mode_listens_on_port_when_selene_port_is_unset(tmp_path: pathlib.Path) -> None:
    from servicetest import RunningServer, start, wait_ready

    listen = str(free_port())
    env = environment(
        tmp_path, SELENE_TLS_TERMINATED_BY_PROXY="true", SELENE_HOST="0.0.0.0", PORT=listen
    )
    del env["SELENE_PORT"]
    log = tmp_path / "server.log"
    process = start(env, log)
    env["SELENE_PORT"] = listen
    server = RunningServer(process, log, env, "http", None)
    try:
        wait_ready(server.client(), process, log)
        assert server.client().health().status == 200
        assert server.stop() in STOPPED_BY_SIGTERM
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_no_token_source_stops_the_start_without_leaking_a_value(tmp_path: pathlib.Path) -> None:
    env = environment(tmp_path, SELENE_TOKEN=ENV_TOKEN)
    del env["SELENE_TOKEN_FILE"]
    del env["SELENE_TOKEN"]
    answer = subprocess.run(
        [sys.executable, "-m", "selene_api"], env=env, capture_output=True, text=True, timeout=60
    )
    assert answer.returncode == EX_CONFIG
    assert "SELENE_TOKEN_FILE" in answer.stderr
    assert "SELENE_TOKEN" in answer.stderr
    assert ENV_TOKEN not in answer.stderr + answer.stdout
    assert TOKEN not in answer.stderr + answer.stdout


def test_an_environment_token_starts_without_writing_a_file(tmp_path: pathlib.Path) -> None:
    from servicetest import RunningServer, start, wait_ready

    env = environment(tmp_path, SELENE_TOKEN=ENV_TOKEN)
    token_file = pathlib.Path(env["SELENE_TOKEN_FILE"])
    token_file.unlink()
    del env["SELENE_TOKEN_FILE"]
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}
    log = tmp_path / "server.log"
    process = start(env, log)
    server = RunningServer(process, log, env, "http", None)
    try:
        wait_ready(server.client(ENV_TOKEN), process, log)
        assert server.client(ENV_TOKEN).service().status == 200
        assert server.client(OTHER_TOKEN).service().status == 401
        after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}
        assert after - before == {pathlib.Path("server.log")}
        text = log.read_text() + server.metrics()
        assert ENV_TOKEN not in text
        assert OTHER_TOKEN not in text
        server.stop()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_both_token_sources_accept_the_file_tokens(tmp_path: pathlib.Path) -> None:
    from servicetest import RunningServer, start, wait_ready

    env = environment(tmp_path, SELENE_TOKEN=TOKEN)
    pathlib.Path(env["SELENE_TOKEN_FILE"]).write_text(f"{TOKEN}\n{FILE_OTHER}\n", encoding="ascii")
    log = tmp_path / "server.log"
    process = start(env, log)
    server = RunningServer(process, log, env, "http", None)
    try:
        wait_ready(server.client(TOKEN), process, log)
        assert server.client(TOKEN).service().status == 200
        assert server.client(FILE_OTHER).service().status == 200
        text = log.read_text() + server.metrics()
        assert TOKEN not in text
        assert FILE_OTHER not in text
        server.stop()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_inconsistent_token_sources_stop_the_start(tmp_path: pathlib.Path) -> None:
    answer = _start_output(tmp_path, SELENE_TOKEN=OTHER_TOKEN)
    assert answer.returncode == EX_CONFIG
    assert "SELENE_TOKEN" in answer.stderr
    assert "SELENE_TOKEN_FILE" in answer.stderr
    assert OTHER_TOKEN not in answer.stderr + answer.stdout
    assert TOKEN not in answer.stderr + answer.stdout


@pytest.mark.parametrize(
    "value",
    ["test-token-short", "test-token-abcdefghijklmnopqrstuvwxyz abc"],
)
def test_a_malformed_environment_token_stops_the_start(tmp_path: pathlib.Path, value: str) -> None:
    env = environment(tmp_path, SELENE_TOKEN=value)
    del env["SELENE_TOKEN_FILE"]
    answer = subprocess.run(
        [sys.executable, "-m", "selene_api"], env=env, capture_output=True, text=True, timeout=60
    )
    assert answer.returncode == EX_CONFIG
    assert "SELENE_TOKEN" in answer.stderr
    assert value not in answer.stderr + answer.stdout
