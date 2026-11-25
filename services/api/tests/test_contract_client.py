# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""The contract client speaks contract v1 with the standard library only."""

import ast
import copy
import json
import pathlib
import sys
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

import pytest

import selene_contract

Reader = Callable[[str], Any]
CLIENT = pathlib.Path(selene_contract.__file__)


def test_the_client_imports_the_standard_library_only() -> None:
    tree = ast.parse(CLIENT.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert imported <= sys.stdlib_module_names, imported - sys.stdlib_module_names


def test_plain_http_needs_an_explicit_flag() -> None:
    with pytest.raises(ValueError, match="allow_insecure_http"):
        selene_contract.SeleneClient("http://127.0.0.1:1", "token")
    with pytest.raises(ValueError, match="https"):
        selene_contract.SeleneClient("ftp://example", "token")
    client = selene_contract.SeleneClient("http://127.0.0.1:1", "t", allow_insecure_http=True)
    assert client.base_url == "http://127.0.0.1:1"


def test_tls_verification_is_on() -> None:
    client = selene_contract.SeleneClient("https://127.0.0.1:1", "token")
    context = client._context
    assert context is not None
    assert context.check_hostname is True
    assert context.verify_mode.name == "CERT_REQUIRED"


class TestReferenceComparison:
    def job(self, read_fixture: Reader) -> dict[str, Any]:
        job: dict[str, Any] = read_fixture("examples/job-succeeded.json")
        return job

    def reference(self, read_fixture: Reader) -> dict[str, Any]:
        reference: dict[str, Any] = read_fixture("coclustering-single-1/reference.expected.json")
        return reference

    def test_the_example_matches_the_reference(self, read_fixture: Reader) -> None:
        job = self.job(read_fixture)
        selene_contract.check_job(job)
        assert selene_contract.compare_with_reference(job, self.reference(read_fixture)) == []

    def test_a_different_partition_is_reported(self, read_fixture: Reader) -> None:
        job = self.job(read_fixture)
        job["result"]["value"]["group_index"][0] = 1 - job["result"]["value"]["group_index"][0]
        problems = selene_contract.compare_with_reference(job, self.reference(read_fixture))
        assert problems == ["group_index differs"]

    def test_a_number_outside_the_tolerance_is_reported(self, read_fixture: Reader) -> None:
        job = self.job(read_fixture)
        job["result"]["value"]["objective"] *= 1 + 1e-6
        problems = selene_contract.compare_with_reference(job, self.reference(read_fixture))
        assert problems == ["objective outside the tolerance"]

    def test_a_number_inside_the_tolerance_passes(self, read_fixture: Reader) -> None:
        job = self.job(read_fixture)
        job["result"]["value"]["objective"] *= 1 + 1e-12
        assert selene_contract.compare_with_reference(job, self.reference(read_fixture)) == []

    def test_another_method_version_is_reported(self, read_fixture: Reader) -> None:
        job = self.job(read_fixture)
        job["result"]["provenance"]["method"]["version"] = "9.9.9"
        problems = selene_contract.compare_with_reference(job, self.reference(read_fixture))
        assert "method version differs: the reference must be regenerated" in problems

    def test_the_summary_prints_no_individual_value(self, read_fixture: Reader) -> None:
        job = self.job(read_fixture)
        printed = json.dumps(selene_contract.summary(job))
        for voter in job["result"]["value"]["voters"]:
            assert voter not in printed
        for item in job["result"]["value"]["items"]:
            assert item not in printed
        assert "memberships" not in printed
        assert json.loads(printed)["group_sizes"] == [6, 6]


def test_a_job_without_a_required_field_is_refused(read_fixture: Reader) -> None:
    job = read_fixture("examples/job-running.json")
    selene_contract.check_job(job)
    broken = copy.deepcopy(job)
    del broken["expires_at"]
    with pytest.raises(selene_contract.ContractError, match="expires_at"):
        selene_contract.check_job(broken)


def test_a_job_with_unknown_fields_is_accepted(read_fixture: Reader) -> None:
    job = read_fixture("examples/job-failed-timeout.json")
    job["a_field_of_a_later_minor_version"] = {"anything": True}
    selene_contract.check_job(job)


class _Recorder(BaseHTTPRequestHandler):
    seen: ClassVar[list[dict[str, Any]]] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        self.seen.append({"path": self.path, "headers": dict(self.headers), "body": body})
        self._answer(
            409, {"error": {"code": "submission_conflict", "message": "m", "retryable": False}}
        )

    def do_GET(self) -> None:
        self.seen.append({"path": self.path, "headers": dict(self.headers), "body": b""})
        self._answer(200, {"status": "ok"})

    def _answer(self, status: int, body: object) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def recorder() -> Iterator[str]:
    _Recorder.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_the_client_sends_the_token_and_reads_errors(recorder: str) -> None:
    client = selene_contract.SeleneClient(recorder, "synthetic-token", allow_insecure_http=True)
    answer = client.submit({"kind": "coclustering.single"})
    assert answer.status == 409
    assert answer.error_code == "submission_conflict"
    sent = _Recorder.seen[0]
    assert sent["path"] == "/v1/jobs"
    assert sent["headers"]["Authorization"] == "Bearer synthetic-token"
    assert sent["headers"]["Content-Type"] == "application/json"
    assert json.loads(sent["body"]) == {"kind": "coclustering.single"}


def test_the_probes_go_without_a_token(recorder: str) -> None:
    client = selene_contract.SeleneClient(recorder, "synthetic-token", allow_insecure_http=True)
    assert client.with_token(None).health().body == {"status": "ok"}
    assert "Authorization" not in _Recorder.seen[0]["headers"]
    assert client.token == "synthetic-token"
