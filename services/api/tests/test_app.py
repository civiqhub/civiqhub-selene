# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""The HTTP layer answers contract v1: routes, authentication, refusals and documents."""

import asyncio
import json
import pathlib
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest
from jsonschema import Draft202012Validator
from servicetest import AUTH, TOKEN, FakeRunner, asgi_client, service_for, settings_for, settle

import make_fixtures
import selene_contract
from selene_api.app import BUSY_RETRY_AFTER_SECONDS

Validator = Callable[[str], Draft202012Validator]
Reader = Callable[[str], Any]
INVALID = sorted(
    path.stem
    for path in (make_fixtures.FIXTURES / "coclustering-single-1" / "invalid").glob("*.json")
)
SECURED = [
    ("GET", "/v1/service"),
    ("POST", "/v1/jobs"),
    ("GET", "/v1/jobs/job_AAAAAAAAAAAAAAAAAAAAAA"),
    ("DELETE", "/v1/jobs/job_AAAAAAAAAAAAAAAAAAAAAA"),
]


def run_with(
    tmp_path: pathlib.Path,
    scenario: Callable[[httpx.AsyncClient], Awaitable[None]],
    runner: Any = None,
    **settings: Any,
) -> None:
    service = service_for(settings_for(tmp_path, **settings), runner)

    async def main() -> None:
        async with asgi_client(service) as client:
            await scenario(client)

    asyncio.run(main())


def submission(key: str = "test-only:app:0001") -> dict[str, Any]:
    body = make_fixtures.reference_submission()
    body["submission_key"] = key
    return body


async def finished(client: httpx.AsyncClient, job_id: str) -> dict[str, Any]:
    for _ in range(3000):
        answer = await client.get(f"/v1/jobs/{job_id}", headers=AUTH)
        assert answer.status_code == 200
        if answer.json()["state"] != "running":
            job: dict[str, Any] = answer.json()
            return job
        await asyncio.sleep(0.01)
    raise AssertionError("the job did not finish")


def test_the_probes_need_no_token_and_carry_no_data(tmp_path: pathlib.Path) -> None:
    async def scenario(client: httpx.AsyncClient) -> None:
        assert (await client.get("/healthz")).json() == {"status": "ok"}
        ready = await client.get("/readyz")
        assert (ready.status_code, ready.json()) == (200, {"status": "ready"})

    run_with(tmp_path, scenario)


@pytest.mark.parametrize(("method", "path"), SECURED)
@pytest.mark.parametrize(
    "header", [None, "Bearer wrong-token-of-sufficient-length-000000", f"Basic {TOKEN}", "Bearer "]
)
def test_every_operation_needs_the_token(
    tmp_path: pathlib.Path, method: str, path: str, header: str | None
) -> None:
    async def scenario(client: httpx.AsyncClient) -> None:
        headers = {} if header is None else {"Authorization": header}
        if method == "POST":
            headers["Content-Type"] = "application/json"
        answer = await client.request(
            method, path, headers=headers, content=json.dumps(submission())
        )
        assert answer.status_code == 401
        assert answer.json()["error"]["code"] == "unauthenticated"
        assert answer.headers["www-authenticate"].startswith("Bearer")
        assert TOKEN not in answer.text

    run_with(tmp_path, scenario, FakeRunner())


@pytest.mark.parametrize("name", INVALID)
def test_each_negative_case_is_refused_as_the_contract_says(
    tmp_path: pathlib.Path, read_fixture: Reader, name: str
) -> None:
    case = read_fixture(f"coclustering-single-1/invalid/{name}.json")

    async def scenario(client: httpx.AsyncClient) -> None:
        answer = await client.post("/v1/jobs", headers=AUTH, json=case["body"])
        assert answer.status_code == case["expect"]["status"]
        assert answer.json()["error"]["code"] == case["expect"]["code"]
        assert answer.json()["error"]["retryable"] is False
        for value in ("someone@example.org", "spectral.embed", '"notes"'):
            assert value not in answer.text

    run_with(tmp_path, scenario, FakeRunner())


def test_the_refusals_of_the_transport(tmp_path: pathlib.Path) -> None:
    async def scenario(client: httpx.AsyncClient) -> None:
        bad_json = await client.post(
            "/v1/jobs", headers=AUTH | {"Content-Type": "application/json"}, content=b"{x"
        )
        assert (bad_json.status_code, bad_json.json()["error"]["code"]) == (
            400,
            "malformed_request",
        )
        nan = await client.post(
            "/v1/jobs",
            headers=AUTH | {"Content-Type": "application/json"},
            content=b'{"seed": NaN}',
        )
        assert nan.status_code == 400
        text = await client.post(
            "/v1/jobs", headers=AUTH | {"Content-Type": "text/plain"}, content=b"{}"
        )
        assert (text.status_code, text.json()["error"]["code"]) == (415, "unsupported_media_type")
        large = await client.post(
            "/v1/jobs", headers=AUTH | {"Content-Type": "application/json"}, content=b" " * 2048
        )
        assert (large.status_code, large.json()["error"]["code"]) == (413, "payload_too_large")
        array = await client.post("/v1/jobs", headers=AUTH, json=[1, 2])
        assert (array.status_code, array.json()["error"]["code"]) == (422, "invalid_request")
        version = await client.post(
            "/v1/jobs", headers=AUTH, json={"kind": "coclustering.single", "schema_version": "1"}
        )
        assert version.json()["error"]["code"] == "invalid_request"
        missing = await client.get("/v2/jobs", headers=AUTH)
        assert (missing.status_code, missing.json()["error"]["code"]) == (404, "not_found")
        method = await client.put("/v1/jobs", headers=AUTH)
        assert (method.status_code, method.json()["error"]["code"]) == (405, "method_not_allowed")
        odd = await client.get("/v1/jobs/not-a-job-id", headers=AUTH)
        assert (odd.status_code, odd.json()["error"]["code"]) == (404, "job_not_found")

    run_with(tmp_path, scenario, FakeRunner(), max_body_bytes=1024)


def test_the_service_describes_itself(tmp_path: pathlib.Path, schema_validator: Validator) -> None:
    async def scenario(client: httpx.AsyncClient) -> None:
        answer = await client.get("/v1/service", headers=AUTH)
        assert answer.status_code == 200
        info = answer.json()
        schema_validator("ServiceInfo").validate(info)
        assert info["contract"]["version"] == "1.0.0"
        assert info["deployment"]["environment"] == "dev"
        assert info["kinds"] == [
            {
                "kind": "coclustering.single",
                "schema_versions": [1],
                "method": {"name": "latent_block_model", "version": "0.1.0"},
            }
        ]
        assert info["limits"]["max_voters"] == 2000
        assert info["capacity"] == {"slots": 1, "running": 0, "accepting": True}
        assert info["engine"]["wheel_sha256"] is None

    run_with(tmp_path, scenario, environment="dev")


def test_a_real_computation_from_submission_to_result(
    tmp_path: pathlib.Path, schema_validator: Validator, read_fixture: Reader
) -> None:
    expected = read_fixture("coclustering-single-1/reference.expected.json")

    async def scenario(client: httpx.AsyncClient) -> None:
        answer = await client.post("/v1/jobs", headers=AUTH, json=submission())
        assert answer.status_code == 202
        job = answer.json()
        schema_validator("Job").validate(job)
        assert job["state"] == "running"
        assert answer.headers["location"] == f"/v1/jobs/{job['id']}"
        assert job["caller_reference"] == "test-only/generation-1"
        done = await finished(client, job["id"])
        schema_validator("Job").validate(done)
        selene_contract.check_job(done)
        assert selene_contract.compare_with_reference(done, expected) == []
        assert done["service"]["contract_version"] == "1.0.0"
        assert done["expires_at"] is not None

        replay = await client.post("/v1/jobs", headers=AUTH, json=submission())
        assert (replay.status_code, replay.json()["id"]) == (200, job["id"])
        changed = submission()
        changed["payload"]["seed"] += 1
        conflict = await client.post("/v1/jobs", headers=AUTH, json=changed)
        assert (conflict.status_code, conflict.json()["error"]["code"]) == (
            409,
            "submission_conflict",
        )

        deleted = await client.delete(f"/v1/jobs/{job['id']}", headers=AUTH)
        assert deleted.status_code == 204
        gone = await client.get(f"/v1/jobs/{job['id']}", headers=AUTH)
        assert (gone.status_code, gone.json()["error"]["code"]) == (404, "job_not_found")

    run_with(tmp_path, scenario)


def test_an_explained_absence_succeeds(tmp_path: pathlib.Path, read_fixture: Reader) -> None:
    body = read_fixture("coclustering-single-1/no-votes.submission.json")

    async def scenario(client: httpx.AsyncClient) -> None:
        answer = await client.post("/v1/jobs", headers=AUTH, json=body)
        done = await finished(client, answer.json()["id"])
        assert done["state"] == "succeeded"
        assert done["result"]["available"] is False
        assert done["result"]["unavailable"]["reason"] == "no_votes"

    run_with(tmp_path, scenario)


def test_busy_answers_503_with_retry_after_and_the_api_stays_responsive(
    tmp_path: pathlib.Path,
) -> None:
    runner = FakeRunner()
    runner.hold = True

    async def scenario(client: httpx.AsyncClient) -> None:
        first = await client.post("/v1/jobs", headers=AUTH, json=submission())
        assert first.status_code == 202
        await settle(lambda: len(runner.started) == 1)
        other = await client.post("/v1/jobs", headers=AUTH, json=submission("test-only:app:0002"))
        assert other.status_code == 503
        assert other.json()["error"] == {
            "code": "busy",
            "message": "The computation slot is taken; retry later.",
            "retryable": True,
        }
        assert other.headers["retry-after"] == str(BUSY_RETRY_AFTER_SECONDS)
        replay = await client.post("/v1/jobs", headers=AUTH, json=submission())
        assert replay.status_code == 200
        assert (await client.get("/healthz")).status_code == 200
        assert (await client.get("/readyz")).status_code == 200
        info = (await client.get("/v1/service", headers=AUTH)).json()
        assert info["capacity"] == {"slots": 1, "running": 1, "accepting": False}
        running = await client.get(f"/v1/jobs/{first.json()['id']}", headers=AUTH)
        assert running.json()["state"] == "running"
        runner.release()

    run_with(tmp_path, scenario, runner)


def test_a_service_that_is_not_ready_refuses_computations(tmp_path: pathlib.Path) -> None:
    service = service_for(settings_for(tmp_path), FakeRunner())

    async def main() -> None:
        from selene_api.app import create_app

        app = create_app(service)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://selene.test") as client:
            ready = await client.get("/readyz")
            assert (ready.status_code, ready.json()) == (503, {"status": "starting"})
            answer = await client.post("/v1/jobs", headers=AUTH, json=submission())
            assert (answer.status_code, answer.json()["error"]["code"]) == (503, "not_ready")

    asyncio.run(main())


def test_the_start_up_check_refuses_another_engine(tmp_path: pathlib.Path) -> None:
    class OtherEngine(FakeRunner):
        async def describe(self, *, timeout: float) -> dict[str, Any]:
            return {"engine_version": "9.9.9", "kinds": []}

    service = service_for(settings_for(tmp_path), OtherEngine())
    with pytest.raises(RuntimeError, match="another engine version"):
        asyncio.run(service.start())
    assert service.phase == "starting"
