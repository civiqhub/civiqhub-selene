# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Contract client of the SELENE service, version 1. Standard library only. TEST ONLY data.

It knows the HTTP contract (``services/api/openapi.yaml``) and nothing of the service
code or of the engine classes: it is the consumer side of the contract. The CI runs the
client of the approved base branch against a new server, so a server that breaks /v1
fails even if its own tests pass. The synthetic acceptance test of each deployment can
run it as well.

Commands, from the root of the repository::

    # one synthetic computation: submit, follow, compare with the local reference
    python services/api/contract/selene_contract.py run \\
        --url https://selene.example.internal:8443 --ca-file ca.pem --token-file token \\
        --reference services/api/contract/fixtures/v1/coclustering-single-1/reference

    # the consumer suite of contract v1: probes, refusals, replay, conflict, result
    python services/api/contract/selene_contract.py check --url ... --ca-file ... \\
        --token-file ...

The token is read from ``--token-file`` or from the ``SELENE_TOKEN`` environment
variable, never from the command line. Plain ``http://`` needs ``--allow-insecure-http``
and is meant for a server bound to the loopback interface. Outputs name job identifiers,
versions, fingerprints, states, group sizes and timings; they never print a vote, a
pseudonym or a membership.
"""

import argparse
import json
import math
import os
import pathlib
import ssl
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CONTRACT_MAJOR",
    "ContractError",
    "HTTPResult",
    "SeleneClient",
    "compare_with_reference",
    "load_reference",
    "run_check",
]

CONTRACT_MAJOR = 1
"""The major version of the contract this client speaks: the ``/v1`` prefix."""

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "v1"
TERMINAL_STATES = frozenset({"succeeded", "failed"})
JOB_FIELDS = (
    "id",
    "kind",
    "schema_version",
    "submission_key",
    "caller_reference",
    "content_fingerprint",
    "state",
    "created_at",
    "started_at",
    "finished_at",
    "expires_at",
    "result",
    "error",
    "service",
)


class ContractError(AssertionError):
    """The server does not answer as contract v1 says."""


@dataclass(frozen=True)
class HTTPResult:
    """One answer of the server: status, selected headers and decoded JSON body."""

    status: int
    headers: Mapping[str, str]
    body: Any

    @property
    def error_code(self) -> str | None:
        """The ``error.code`` of an error body, or ``None``."""
        if isinstance(self.body, dict) and isinstance(self.body.get("error"), dict):
            code = self.body["error"].get("code")
            return code if isinstance(code, str) else None
        return None


class SeleneClient:
    """A small client of contract v1.

    Args:
        base_url: ``https://host:port``; plain HTTP needs ``allow_insecure_http``.
        token: The machine token, or ``None`` to send no ``Authorization`` header.
        ca_file: PEM bundle that signs the server certificate. Without it, the system
            trust store is used. Verification is never turned off.
        timeout: Seconds allowed to each HTTP request.
        allow_insecure_http: Accept a ``http://`` URL, for a loopback test server.
    """

    def __init__(
        self,
        base_url: str,
        token: str | None,
        *,
        ca_file: str | None = None,
        timeout: float = 30.0,
        allow_insecure_http: bool = False,
    ) -> None:
        if base_url.startswith("http://") and not allow_insecure_http:
            raise ValueError("plain http:// needs allow_insecure_http=True")
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("base_url starts with https://")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._context: ssl.SSLContext | None = None
        if base_url.startswith("https://"):
            context = ssl.create_default_context(cafile=ca_file)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            self._context = context

    def with_token(self, token: str | None) -> "SeleneClient":
        """Return the same client with another token, for refusal tests."""
        clone = object.__new__(SeleneClient)
        clone.__dict__.update(self.__dict__)
        clone.token = token
        return clone

    def request(
        self,
        method: str,
        path: str,
        body: object | None = None,
        *,
        raw_body: bytes | None = None,
        content_type: str = "application/json",
    ) -> HTTPResult:
        """Send one request and return the answer, whatever its status."""
        data = raw_body
        if body is not None:
            data = json.dumps(body).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = content_type
        if self.token is not None:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            self.base_url + path, data=data, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout, context=self._context
            ) as response:
                return _decoded(response.status, dict(response.headers.items()), response.read())
        except urllib.error.HTTPError as error:
            with error:
                return _decoded(error.code, dict(error.headers.items()), error.read())

    # Operations of the contract --------------------------------------------------------

    def health(self) -> HTTPResult:
        """``GET /healthz``."""
        return self.request("GET", "/healthz")

    def readiness(self) -> HTTPResult:
        """``GET /readyz``."""
        return self.request("GET", "/readyz")

    def service(self) -> HTTPResult:
        """``GET /v1/service``."""
        return self.request("GET", f"/v{CONTRACT_MAJOR}/service")

    def submit(self, submission: Mapping[str, Any]) -> HTTPResult:
        """``POST /v1/jobs``."""
        return self.request("POST", f"/v{CONTRACT_MAJOR}/jobs", submission)

    def job(self, job_id: str) -> HTTPResult:
        """``GET /v1/jobs/{job_id}``."""
        return self.request("GET", f"/v{CONTRACT_MAJOR}/jobs/{job_id}")

    def delete(self, job_id: str) -> HTTPResult:
        """``DELETE /v1/jobs/{job_id}``."""
        return self.request("DELETE", f"/v{CONTRACT_MAJOR}/jobs/{job_id}")

    def wait(self, job_id: str, *, timeout: float = 300.0, interval: float = 0.5) -> HTTPResult:
        """Poll a job until it is ``succeeded`` or ``failed``, and return its last answer.

        Raises:
            ContractError: If the job disappears, an answer is not a job, or ``timeout``
                passes first.
        """
        deadline = time.monotonic() + timeout
        while True:
            answer = self.job(job_id)
            expect(answer.status == 200, f"GET job answered {answer.status} ({answer.error_code})")
            check_job(answer.body)
            if answer.body["state"] in TERMINAL_STATES:
                return answer
            if time.monotonic() > deadline:
                raise ContractError(f"job {job_id} still {answer.body['state']} after {timeout}s")
            time.sleep(interval)


def _decoded(status: int, headers: Mapping[str, str], raw: bytes) -> HTTPResult:
    kept = {
        name.lower(): value
        for name, value in headers.items()
        if name.lower() in {"content-type", "location", "retry-after", "www-authenticate"}
    }
    body: Any = None
    if raw:
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
    return HTTPResult(status, kept, body)


# Checks of the shapes the contract promises ----------------------------------------------


def expect(condition: bool, message: str) -> None:
    """Raise :class:`ContractError` with ``message`` unless ``condition`` holds."""
    if not condition:
        raise ContractError(message)


def check_error(answer: HTTPResult, status: int, code: str | None = None) -> None:
    """Check an error answer: its status, the error body and, if given, its code."""
    expect(answer.status == status, f"expected status {status}, got {answer.status}")
    expect(answer.error_code is not None, f"status {status} without an error body")
    error = answer.body["error"]
    expect(isinstance(error.get("message"), str), "error.message is not text")
    expect(isinstance(error.get("retryable"), bool), "error.retryable is not a boolean")
    if code is not None:
        expect(answer.error_code == code, f"expected error {code}, got {answer.error_code}")


def check_job(job: object) -> None:
    """Check the fields contract v1 requires in a job; unknown fields are ignored."""
    expect(isinstance(job, dict), "a job is a JSON object")
    assert isinstance(job, dict)
    missing = [name for name in JOB_FIELDS if name not in job]
    expect(not missing, f"job without {', '.join(missing)}")
    expect(job["state"] in {"running", "succeeded", "failed"}, f"unknown state {job['state']!r}")
    if job["state"] == "succeeded":
        expect(isinstance(job["result"], dict), "a succeeded job has a result")
        expect(job["error"] is None, "a succeeded job has no error")
        check_result(job["result"])
    elif job["state"] == "failed":
        expect(job["result"] is None, "a failed job has no result")
        expect(isinstance(job["error"], dict), "a failed job has an error")
        expect(isinstance(job["error"].get("retryable"), bool), "job error without retryable")
    else:
        expect(job["result"] is None and job["error"] is None, "a running job has no outcome")
    for name in ("service_version", "contract_version", "engine_version"):
        expect(isinstance(job["service"].get(name), str), f"job.service without {name}")


def check_result(result: Mapping[str, Any]) -> None:
    """Check the fields of a ``selene.coclustering.single.result/1``."""
    expect(result.get("schema") == "selene.coclustering.single.result/1", "unknown result schema")
    expect(isinstance(result.get("available"), bool), "result.available is not a boolean")
    provenance = result.get("provenance")
    expect(isinstance(provenance, dict), "result without provenance")
    assert isinstance(provenance, dict)
    for name in ("method", "engine_version", "input_fingerprint", "randomness", "fingerprint"):
        expect(name in provenance, f"provenance without {name}")
    if result["available"]:
        value = result.get("value")
        expect(isinstance(value, dict), "an available result has a value")
        assert isinstance(value, dict)
        expect(len(value["group_index"]) == len(value["voters"]), "one group per voter")
        expect(len(value["block_index"]) == len(value["items"]), "one block per item")
    else:
        unavailable = result.get("unavailable")
        expect(isinstance(unavailable, dict), "an absent result says why")
        assert isinstance(unavailable, dict)
        expect(
            unavailable.get("reason") in {"no_votes", "not_enough_data", "unsupported_input"},
            "unknown reason of absence",
        )


# Reference comparison --------------------------------------------------------------------


def load_reference(stem: str | pathlib.Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the submission and the expected outcome of a fixture, by its path stem.

    ``.../reference`` reads ``reference.submission.json`` and ``reference.expected.json``.
    """
    base = pathlib.Path(stem)
    submission = json.loads(base.with_name(base.name + ".submission.json").read_text("utf-8"))
    expected = json.loads(base.with_name(base.name + ".expected.json").read_text("utf-8"))
    return submission, expected


def _close(found: object, wanted: object, relative: float, absolute: float) -> bool:
    if isinstance(wanted, list):
        return (
            isinstance(found, list)
            and len(found) == len(wanted)
            and all(_close(f, w, relative, absolute) for f, w in zip(found, wanted, strict=True))
        )
    if wanted is None or found is None:
        return found is wanted
    if isinstance(wanted, (int, float)) and isinstance(found, (int, float)):
        return math.isclose(float(found), float(wanted), rel_tol=relative, abs_tol=absolute)
    return found == wanted


def compare_with_reference(job: Mapping[str, Any], reference: Mapping[str, Any]) -> list[str]:
    """Return the differences between a finished job and its reference; empty if none."""
    problems: list[str] = []
    if job["content_fingerprint"] != reference["content_fingerprint"]:
        problems.append("content_fingerprint differs")
    if job["state"] != "succeeded":
        return [*problems, f"job {job['state']}, not succeeded"]
    result = job["result"]
    provenance = result["provenance"]
    if provenance["method"]["name"] != reference["method"]["name"]:
        problems.append("method name differs")
    if provenance["method"]["version"] != reference["method"]["version"]:
        problems.append("method version differs: the reference must be regenerated")
    if provenance["input_fingerprint"] != reference["input_fingerprint"]:
        problems.append("input_fingerprint differs: the adaptation of the input changed")
    if provenance["fingerprint"] != reference["computation"]:
        problems.append("computation fingerprint differs")
    if result["available"] != reference["available"]:
        return [*problems, "availability differs"]
    expected = reference["expected"]
    if not result["available"]:
        if result["unavailable"]["reason"] != expected["reason"]:
            problems.append("reason of absence differs")
        return problems
    relative = reference["tolerance"]["relative"]
    absolute = reference["tolerance"]["absolute"]
    value = result["value"]
    for name in ("group_index", "block_index"):
        if value[name] != expected[name]:
            problems.append(f"{name} differs")
    for name in ("objective", "integrated_likelihood", "group_memberships", "block_memberships"):
        if not _close(value[name], expected[name], relative, absolute):
            problems.append(f"{name} outside the tolerance")
    return problems


def summary(job: Mapping[str, Any]) -> dict[str, Any]:
    """Return what may be printed of a job: no vote, pseudonym or membership."""
    data: dict[str, Any] = {
        "job": job["id"],
        "state": job["state"],
        "content_fingerprint": job["content_fingerprint"],
        "service": job["service"],
    }
    if job["state"] == "failed":
        data["error"] = {"code": job["error"]["code"], "retryable": job["error"]["retryable"]}
    elif job["state"] == "succeeded":
        result = job["result"]
        data["method"] = result["provenance"]["method"]
        data["computation"] = result["provenance"]["fingerprint"]
        data["available"] = result["available"]
        if result["available"]:
            value = result["value"]
            sizes = [
                value["group_index"].count(group) for group in range(1 + max(value["group_index"]))
            ]
            blocks = [
                value["block_index"].count(block) for block in range(1 + max(value["block_index"]))
            ]
            data["group_sizes"] = sizes
            data["block_sizes"] = blocks
            data["objective"] = value["objective"]
        else:
            data["reason"] = result["unavailable"]["reason"]
    return data


# The consumer suite of contract v1 -------------------------------------------------------


def _fresh_key(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4().hex}"


def run_check(client: SeleneClient, *, wait_timeout: float = 300.0) -> list[dict[str, Any]]:
    """Run the consumer suite of contract v1 and return one report per check.

    Each report holds ``name``, ``passed`` and, on failure, ``problem``. The suite leaves
    no job behind: it deletes the jobs it created.
    """
    reports: list[dict[str, Any]] = []
    created: list[str] = []
    reference_submission, reference = load_reference(
        FIXTURES / "coclustering-single-1" / "reference"
    )
    no_votes_submission, no_votes = load_reference(FIXTURES / "coclustering-single-1" / "no-votes")

    def step(name: str, check: Callable[[], None]) -> None:
        try:
            check()
        except ContractError as problem:
            reports.append({"name": name, "passed": False, "problem": str(problem)})
        except (OSError, ValueError, KeyError, TypeError) as problem:
            reports.append({"name": name, "passed": False, "problem": type(problem).__name__})
        else:
            reports.append({"name": name, "passed": True})

    def probes() -> None:
        health = client.with_token(None).health()
        expect(health.status == 200 and health.body == {"status": "ok"}, "healthz is not ok")
        ready = client.with_token(None).readiness()
        expect(ready.status == 200 and ready.body.get("status") == "ready", "readyz not ready")

    def service_info() -> None:
        answer = client.service()
        expect(answer.status == 200, f"GET /v1/service answered {answer.status}")
        info = answer.body
        for name in (
            "service",
            "contract",
            "engine",
            "build",
            "deployment",
            "kinds",
            "limits",
            "capacity",
        ):
            expect(name in info, f"service info without {name}")
        expect(info["contract"]["version"].split(".")[0] == str(CONTRACT_MAJOR), "not contract v1")
        kinds = {
            (kind["kind"], version) for kind in info["kinds"] for version in kind["schema_versions"]
        }
        expect(("coclustering.single", 1) in kinds, "coclustering.single/1 is not supported")

    def refuses_missing_token() -> None:
        answer = client.with_token(None).service()
        check_error(answer, 401, "unauthenticated")
        expect(
            answer.headers.get("www-authenticate", "").startswith("Bearer"), "no WWW-Authenticate"
        )
        check_error(client.with_token(None).submit(reference_submission), 401, "unauthenticated")

    def refuses_wrong_token() -> None:
        wrong = client.with_token("not-the-token-" + uuid.uuid4().hex)
        check_error(wrong.service(), 401, "unauthenticated")
        check_error(wrong.job("job_" + "A" * 22), 401, "unauthenticated")

    def refuses_invalid() -> None:
        for path in sorted((FIXTURES / "coclustering-single-1" / "invalid").glob("*.json")):
            case = json.loads(path.read_text("utf-8"))
            body = dict(case["body"])
            body["submission_key"] = (
                body["submission_key"]
                if path.stem == "short-submission-key"
                else _fresh_key("test-only:invalid")
            )
            answer = client.submit(body)
            try:
                check_error(answer, case["expect"]["status"], case["expect"]["code"])
            except ContractError as problem:
                raise ContractError(f"{path.stem}: {problem}") from None

    def refuses_malformed() -> None:
        check_error(
            client.request("POST", "/v1/jobs", raw_body=b"{not json"), 400, "malformed_request"
        )
        check_error(
            client.request("POST", "/v1/jobs", raw_body=b"{}", content_type="text/plain"),
            415,
            "unsupported_media_type",
        )

    def unknown_job() -> None:
        check_error(client.job("job_" + uuid.uuid4().hex), 404, "job_not_found")
        check_error(client.delete("job_" + uuid.uuid4().hex), 404, "job_not_found")

    def computation() -> None:
        submission = dict(reference_submission)
        submission["submission_key"] = _fresh_key("test-only:check")
        first = client.submit(submission)
        expect(first.status == 202, f"submission answered {first.status} ({first.error_code})")
        check_job(first.body)
        created.append(first.body["id"])
        expect(first.headers.get("location") == f"/v1/jobs/{first.body['id']}", "bad Location")
        expect(first.body["submission_key"] == submission["submission_key"], "key not echoed")
        expect(first.body["content_fingerprint"] == reference["content_fingerprint"], "fingerprint")

        replay = client.submit(submission)
        expect(replay.status == 200, f"replay answered {replay.status} ({replay.error_code})")
        expect(replay.body["id"] == first.body["id"], "a replay started another job")

        changed = json.loads(json.dumps(submission))
        changed["payload"]["seed"] += 1
        check_error(client.submit(changed), 409, "submission_conflict")

        done = client.wait(first.body["id"], timeout=wait_timeout)
        problems = compare_with_reference(done.body, reference)
        expect(not problems, "; ".join(problems))
        expect(done.body["expires_at"] is not None, "a finished job has expires_at")

        after = client.submit(submission)
        expect(after.status == 200 and after.body["id"] == first.body["id"], "replay after finish")

    def explained_absence() -> None:
        submission = dict(no_votes_submission)
        submission["submission_key"] = _fresh_key("test-only:no-votes")
        answer = client.submit(submission)
        expect(answer.status == 202, f"submission answered {answer.status} ({answer.error_code})")
        created.append(answer.body["id"])
        done = client.wait(answer.body["id"], timeout=wait_timeout)
        problems = compare_with_reference(done.body, no_votes)
        expect(not problems, "; ".join(problems))

    def deletion() -> None:
        for job_id in list(created):
            answer = client.delete(job_id)
            expect(answer.status == 204, f"DELETE answered {answer.status}")
            check_error(client.job(job_id), 404, "job_not_found")
            created.remove(job_id)

    step("probes", probes)
    step("service-info", service_info)
    step("refuses-missing-token", refuses_missing_token)
    step("refuses-wrong-token", refuses_wrong_token)
    step("refuses-malformed", refuses_malformed)
    step("refuses-invalid", refuses_invalid)
    step("unknown-job", unknown_job)
    step("computation", computation)
    step("explained-absence", explained_absence)
    step("deletion", deletion)
    for job_id in created:
        client.delete(job_id)
    return reports


# Command line ----------------------------------------------------------------------------


def _token(arguments: argparse.Namespace) -> str:
    if arguments.token_file:
        return pathlib.Path(arguments.token_file).read_text("utf-8").strip().splitlines()[0]
    token = os.environ.get("SELENE_TOKEN")
    if not token:
        sys.exit("give --token-file or set SELENE_TOKEN")
    return token


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Contract client of the SELENE service, v1.")
    parser.add_argument("command", choices=["run", "check", "service"])
    parser.add_argument("--url", required=True, help="https://host:port of the service")
    parser.add_argument("--ca-file", help="PEM bundle that signs the server certificate")
    parser.add_argument("--token-file", help="file whose first line is the machine token")
    parser.add_argument("--allow-insecure-http", action="store_true")
    parser.add_argument(
        "--reference",
        default=str(FIXTURES / "coclustering-single-1" / "reference"),
        help="fixture stem: <stem>.submission.json and <stem>.expected.json",
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds to wait for a job")
    arguments = parser.parse_args(argv)
    client = SeleneClient(
        arguments.url,
        _token(arguments),
        ca_file=arguments.ca_file,
        allow_insecure_http=arguments.allow_insecure_http,
    )
    if arguments.command == "service":
        answer = client.service()
        print(json.dumps({"status": answer.status, "body": answer.body}, indent=2))
        return 0 if answer.status == 200 else 1
    if arguments.command == "check":
        reports = run_check(client, wait_timeout=arguments.timeout)
        print(json.dumps({"contract": CONTRACT_MAJOR, "checks": reports}, indent=2))
        return 0 if all(report["passed"] for report in reports) else 1
    submission, reference = load_reference(arguments.reference)
    submission = {**submission, "submission_key": _fresh_key("test-only:run")}
    started = time.monotonic()
    answer = client.submit(submission)
    if answer.status not in (200, 202):
        print(json.dumps({"status": answer.status, "error": answer.error_code}))
        return 1
    done = client.wait(answer.body["id"], timeout=arguments.timeout)
    problems = compare_with_reference(done.body, reference)
    client.delete(answer.body["id"])
    report = {
        **summary(done.body),
        "seconds": round(time.monotonic() - started, 3),
        "matches_reference": not problems,
        "problems": problems,
    }
    print(json.dumps(report, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
