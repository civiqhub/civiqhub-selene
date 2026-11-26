# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""HTTPS with a verified certificate, rotations without restart, health and metrics."""

import asyncio
import os
import pathlib
import re
import signal
import socket
import ssl
import time
import urllib.error
import warnings

import pytest

import selene_contract
from selene_api.schemas import CONTRACT_VERSION
from servicetest import (
    AUTH,
    TOKEN,
    FakeRunner,
    asgi_client,
    make_certificates,
    running_server,
    service_for,
    settings_for,
    slow_submission,
)

OTHER_TOKEN = "test-only-rotated-token-" + "0f1e2d3c4b5a6978" * 2


def served_serial(port: int, authority: pathlib.Path) -> int:
    """Return the serial number of the certificate the server presents now."""
    context = ssl.create_default_context(cafile=str(authority))
    with (
        socket.create_connection(("127.0.0.1", port), timeout=10) as raw,
        context.wrap_socket(raw, server_hostname="127.0.0.1") as tls,
    ):
        der = tls.getpeercert(binary_form=True)
    from cryptography import x509

    assert der is not None
    return int(x509.load_der_x509_certificate(der).serial_number)


def metric(text: str, name: str, **labels: str) -> float:
    """Return the value of one sample of the metrics page, or 0 when absent."""
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        sample, _, value = line.rpartition(" ")
        if sample.split("{")[0] != name:
            continue
        found = dict(re.findall(r'(\w+)="([^"]*)"', sample))
        if all(found.get(key) == wanted for key, wanted in labels.items()):
            return float(value)
    return 0.0


class TestHttps:
    def test_the_contract_holds_over_verified_https(self, tmp_path: pathlib.Path) -> None:
        certificates, _authority = make_certificates(tmp_path / "tls")
        with running_server(tmp_path, tls=certificates) as server:
            reports = selene_contract.run_check(server.client(), wait_timeout=120)
            assert [report for report in reports if not report["passed"]] == []

    def test_a_client_that_does_not_trust_the_authority_is_refused(
        self, tmp_path: pathlib.Path
    ) -> None:
        certificates, _authority = make_certificates(tmp_path / "tls")
        stranger, _other = make_certificates(tmp_path / "other")
        with running_server(tmp_path, tls=certificates) as server:
            for client in (
                server.client(ca_file=None),
                server.client(ca_file=str(stranger.authority)),
            ):
                with pytest.raises(urllib.error.URLError) as caught:
                    client.health()
                assert isinstance(caught.value.reason, ssl.SSLCertVerificationError)

    def test_a_certificate_for_another_host_is_refused(self, tmp_path: pathlib.Path) -> None:
        certificates, _authority = make_certificates(tmp_path / "tls", hosts=("selene.test",))
        with running_server(tmp_path, tls=certificates, verify_host=False) as server:
            with pytest.raises(urllib.error.URLError) as caught:
                server.client().health()
            assert isinstance(caught.value.reason, ssl.SSLCertVerificationError)

    def test_plain_http_is_not_served_on_the_https_port(self, tmp_path: pathlib.Path) -> None:
        certificates, _authority = make_certificates(tmp_path / "tls")
        with running_server(tmp_path, tls=certificates) as server:
            plain = selene_contract.SeleneClient(
                f"http://127.0.0.1:{server.env['SELENE_PORT']}", TOKEN, allow_insecure_http=True
            )
            with pytest.raises((OSError, ValueError)):
                plain.health()

    def test_tls_older_than_1_2_is_refused(self, tmp_path: pathlib.Path) -> None:
        certificates, _authority = make_certificates(tmp_path / "tls")
        with running_server(tmp_path, tls=certificates) as server:
            context = ssl.create_default_context(cafile=str(certificates.authority))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)  # TLS 1.1 on purpose
                context.minimum_version = ssl.TLSVersion.MINIMUM_SUPPORTED
                context.maximum_version = ssl.TLSVersion.TLSv1_1
            with (
                pytest.raises((ssl.SSLError, OSError)),
                socket.create_connection(("127.0.0.1", int(server.env["SELENE_PORT"]))) as raw,
                context.wrap_socket(raw, server_hostname="127.0.0.1"),
            ):
                pass

    def test_a_renewed_certificate_is_served_without_a_restart(
        self, tmp_path: pathlib.Path
    ) -> None:
        certificates, authority = make_certificates(tmp_path / "tls", serial=1001)
        with running_server(tmp_path, tls=certificates) as server:
            port = int(server.env["SELENE_PORT"])
            assert served_serial(port, certificates.authority) == 1001
            renewed, _authority = make_certificates(
                tmp_path / "renewed", authority=authority, serial=1002
            )
            os.replace(renewed.key, certificates.key)
            os.replace(renewed.certificate, certificates.certificate)
            deadline = time.monotonic() + 20
            while served_serial(port, certificates.authority) != 1002:
                assert time.monotonic() < deadline, "the renewed certificate was not served"
                time.sleep(0.5)
            assert server.process.poll() is None
            assert metric(server.metrics(), "selene_tls_reloads_total") == 1

    def test_broken_certificate_files_keep_the_previous_certificate(
        self, tmp_path: pathlib.Path
    ) -> None:
        certificates, _authority = make_certificates(tmp_path / "tls", serial=2001)
        with running_server(tmp_path, tls=certificates) as server:
            certificates.certificate.write_text("not a certificate\n")
            deadline = time.monotonic() + 20
            while metric(server.metrics(), "selene_tls_reload_failures_total") < 1:
                assert time.monotonic() < deadline
                time.sleep(0.5)
            assert served_serial(int(server.env["SELENE_PORT"]), certificates.authority) == 2001
            assert server.client().health().status == 200


def test_a_rotated_token_applies_without_a_restart(tmp_path: pathlib.Path) -> None:
    with running_server(tmp_path) as server:
        token_file = pathlib.Path(server.env["SELENE_TOKEN_FILE"])
        both = tmp_path / "both"
        both.write_text(f"{TOKEN}\n{OTHER_TOKEN}\n", encoding="ascii")
        os.replace(both, token_file)
        time.sleep(1.2)
        assert server.client().service().status == 200
        assert server.client(OTHER_TOKEN).service().status == 200
        only = tmp_path / "only"
        only.write_text(f"{OTHER_TOKEN}\n", encoding="ascii")
        os.replace(only, token_file)
        time.sleep(1.2)
        assert server.client().service().status == 401
        assert server.client(OTHER_TOKEN).service().status == 200
        assert metric(server.metrics(), "selene_token_usable") == 1


class TestHealthAndMetrics:
    def test_metrics_live_on_their_own_port_only(self, tmp_path: pathlib.Path) -> None:
        with running_server(tmp_path, SELENE_ENVIRONMENT="test-only") as server:
            text = server.metrics()
            for name in (
                "selene_ready",
                "selene_compute_busy",
                "selene_compute_slots",
                "selene_jobs_retained",
                "selene_worker_memory_limit_bytes",
                "selene_token_usable",
                "process_resident_memory_bytes",
            ):
                assert re.search(rf"^{name} ", text, re.MULTILINE), name
            assert metric(text, "selene_ready") == 1
            assert metric(text, "selene_compute_slots") == 1
            assert metric(text, "selene_worker_memory_limit_bytes") == 1024**3
            assert (
                metric(
                    text,
                    "selene_build_info",
                    environment="test-only",
                    contract_version=CONTRACT_VERSION,
                )
                == 1
            )
            on_application_port = server.client().request("GET", "/metrics")
            assert on_application_port.status == 404

    def test_a_finished_job_is_counted_without_identifiers(self, tmp_path: pathlib.Path) -> None:
        with running_server(tmp_path) as server:
            reports = selene_contract.run_check(server.client(), wait_timeout=120)
            assert all(report["passed"] for report in reports)
            text = server.metrics()
            kind = "coclustering.single"
            assert (
                metric(
                    text, "selene_jobs_finished_total", kind=kind, state="succeeded", reason="value"
                )
                == 2  # the reference of schema 1 and the survey of schema 2
            )
            assert (
                metric(
                    text,
                    "selene_jobs_finished_total",
                    kind=kind,
                    state="succeeded",
                    reason="unavailable",
                )
                == 1
            )
            # The two references and the matrix without votes.
            assert metric(text, "selene_submissions_total", kind=kind, outcome="accepted") == 3
            assert metric(text, "selene_submissions_total", kind=kind, outcome="replayed") == 2
            assert metric(text, "selene_submissions_total", kind=kind, outcome="conflict") == 1
            assert metric(text, "selene_submissions_total", kind=kind, outcome="refused") >= 15
            assert (
                metric(text, "selene_job_duration_seconds_count", kind=kind, state="succeeded") == 3
            )
            assert metric(text, "selene_worker_peak_rss_bytes") > 10 * 1024 * 1024
            assert (
                metric(
                    text,
                    "selene_http_requests_total",
                    route="/v1/jobs/{job_id}",
                    method="GET",
                    status="200",
                )
                >= 1
            )
            values = set(re.findall(r'="([^"]*)"', text))
            assert not [value for value in values if value.startswith("job_")], "a job identifier"
            for forbidden in ("vtr-", "itm-", "prj-", "test-only:", TOKEN):
                assert forbidden not in text, forbidden
            routes = set(re.findall(r'route="([^"]*)"', text))
            assert routes <= {"/v1/service", "/v1/jobs", "/v1/jobs/{job_id}", "unmatched"}

    def test_a_busy_service_stays_alive_ready_and_responsive(self, tmp_path: pathlib.Path) -> None:
        with running_server(tmp_path) as server:
            client = server.client()
            first = client.submit(slow_submission())
            assert first.status == 202
            other = client.submit(slow_submission("test-only:slow:0002"))
            assert (other.status, other.error_code) == (503, "busy")
            for _ in range(5):
                started = time.monotonic()
                assert client.health().status == 200
                assert client.readiness().status == 200
                assert client.job(first.body["id"]).body["state"] == "running"
                assert time.monotonic() - started < 1.0
            text = server.metrics()
            assert metric(text, "selene_compute_busy") == 1
            assert metric(text, "selene_ready") == 1
            assert (
                metric(text, "selene_submissions_total", kind="coclustering.single", outcome="busy")
                == 1
            )

    def test_a_timeout_stops_the_computation_and_frees_the_slot(
        self, tmp_path: pathlib.Path
    ) -> None:
        with running_server(tmp_path, SELENE_JOB_TIMEOUT_SECONDS="2") as server:
            client = server.client()
            first = client.submit(slow_submission())
            started = time.monotonic()
            done = client.wait(first.body["id"], timeout=30, interval=0.1)
            assert time.monotonic() - started < 8
            assert done.body["state"] == "failed"
            assert done.body["error"] == {
                "code": "timeout",
                "message": "The computation ran longer than the time limit and was stopped.",
                "retryable": False,
            }
            assert server.worker_pids() == []
            text = server.metrics()
            assert metric(text, "selene_worker_stops_total", reason="timeout") == 1
            assert metric(text, "selene_compute_busy") == 0
            assert client.submit(slow_submission("test-only:slow:0003")).status == 202

    def test_a_crashed_computation_fails_and_frees_the_slot(self, tmp_path: pathlib.Path) -> None:
        with running_server(tmp_path) as server:
            client = server.client()
            first = client.submit(slow_submission())
            deadline = time.monotonic() + 10
            while not server.worker_pids():
                assert time.monotonic() < deadline
                time.sleep(0.05)
            for pid in server.worker_pids():
                os.kill(pid, signal.SIGKILL)
            done = client.wait(first.body["id"], timeout=30, interval=0.1)
            assert done.body["state"] == "failed"
            assert done.body["error"]["code"] == "worker_crashed"
            assert done.body["error"]["retryable"] is True
            assert server.worker_pids() == []
            text = server.metrics()
            assert (
                metric(
                    text,
                    "selene_jobs_finished_total",
                    kind="coclustering.single",
                    state="failed",
                    reason="worker_crashed",
                )
                == 1
            )
            assert metric(text, "selene_compute_busy") == 0
            retry = client.submit(slow_submission("test-only:slow:0004"))
            assert retry.status == 202
            client.delete(retry.body["id"])
            assert server.worker_pids() == []

    def test_a_restart_forgets_every_job(self, tmp_path: pathlib.Path) -> None:
        with running_server(tmp_path) as server:
            client = server.client()
            reports = selene_contract.run_check(client, wait_timeout=120)
            assert all(report["passed"] for report in reports)
            answer = client.submit(
                selene_contract.load_reference(
                    selene_contract.FIXTURES / "coclustering-single-1" / "reference"
                )[0]
            )
            job_id = answer.body["id"]
            client.wait(job_id, timeout=60)
            env = server.env
            server.stop()
        with running_server(
            tmp_path, SELENE_PORT=env["SELENE_PORT"], SELENE_METRICS_PORT=env["SELENE_METRICS_PORT"]
        ) as again:
            lost = again.client().job(job_id)
            assert (lost.status, lost.error_code) == (404, "job_not_found")


def test_draining_refuses_computations_and_unreadies(tmp_path: pathlib.Path) -> None:
    service = service_for(settings_for(tmp_path), FakeRunner())

    async def main() -> None:
        async with asgi_client(service) as client:
            assert (await client.get("/readyz")).status_code == 200
            service.draining()
            ready = await client.get("/readyz")
            assert (ready.status_code, ready.json()) == (503, {"status": "shutting_down"})
            assert (await client.get("/healthz")).status_code == 200
            answer = await client.post(
                "/v1/jobs", headers=AUTH, json={"kind": "coclustering.single", "schema_version": 1}
            )
            assert (answer.status_code, answer.json()["error"]["code"]) == (503, "shutting_down")
            assert answer.json()["error"]["retryable"] is True

    asyncio.run(main())


def test_an_unusable_token_file_shows_in_the_metrics(tmp_path: pathlib.Path) -> None:
    service = service_for(settings_for(tmp_path), FakeRunner())

    async def main() -> None:
        async with asgi_client(service) as client:
            assert metric(_registry_text(service), "selene_token_usable") == 1
            service.tokens.path.write_text("short\n", encoding="ascii")
            service.tokens._checked = -10.0
            answer = await client.get("/v1/service", headers=AUTH)
            assert answer.status_code == 401
            assert metric(_registry_text(service), "selene_token_usable") == 0

    asyncio.run(main())


def _registry_text(service: object) -> str:
    from prometheus_client import generate_latest

    return generate_latest(service.metrics.registry).decode()  # type: ignore[attr-defined]


def test_a_metric_label_never_takes_a_value_from_the_request(tmp_path: pathlib.Path) -> None:
    service = service_for(settings_for(tmp_path), FakeRunner())

    async def main() -> None:
        async with asgi_client(service) as client:
            for kind in ("spectral.embed", "x" * 200, "job_abcdefghijklmnopqrstuv"):
                await client.post(
                    "/v1/jobs", headers=AUTH, json={"kind": kind, "schema_version": 1}
                )
            await client.request("BREW", "/v1/jobs", headers=AUTH)
            await client.get("/v1/jobs/job_" + "a" * 30, headers=AUTH)
            text = _registry_text(service)
            assert "spectral.embed" not in text
            assert "xxxx" not in text
            assert "job_abcdefgh" not in text
            assert "BREW" not in text
            assert metric(text, "selene_submissions_total", kind="unknown", outcome="refused") == 3

    asyncio.run(main())
