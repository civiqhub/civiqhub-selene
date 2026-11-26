# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""The final image serves contract v1 with the exact engine, within its limits."""

import json
import pathlib
import re
import subprocess
import time
import urllib.error
from collections.abc import Callable
from typing import Any

import pytest

import make_fixtures
import selene_contract
from imagetest import ROOT, Deployment, docker, report
from servicetest import slow_submission

Deploy = Callable[..., Deployment]


def metric(text: str, name: str, **labels: str) -> float:
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


def wait_for_workers(deployment: Deployment, timeout: float = 15) -> list[int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pids = deployment.worker_pids()
        if pids:
            return pids
        time.sleep(0.1)
    raise AssertionError("no computation process started")


class TestArtifact:
    """The image is the one build.sh described, with the exact engine wheel."""

    def test_the_image_under_test_is_the_archived_artifact(
        self, image: str, metadata: dict[str, Any]
    ) -> None:
        identifier = docker("image", "inspect", "--format", "{{.Id}}", image).stdout.strip()
        assert identifier == metadata["image_digest"]
        import hashlib

        archive = ROOT / metadata["archive"]["path"]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == metadata["archive"]["sha256"]

    def test_labels_and_build_information(self, image: str, metadata: dict[str, Any]) -> None:
        labels = json.loads(
            docker("image", "inspect", "--format", "{{json .Config.Labels}}", image).stdout
        )
        assert labels["org.opencontainers.image.revision"] == metadata["commit"]
        assert labels["org.opencontainers.image.version"] == metadata["service"]["version"]
        assert labels["org.opencontainers.image.licenses"] == "Apache-2.0"
        assert labels["io.civiqhub.selene.contract-version"] == metadata["contract"]["version"]
        assert labels["io.civiqhub.selene.engine-version"] == metadata["engine"]["version"]
        assert (
            labels["io.civiqhub.selene.engine-wheel-sha256"] == metadata["engine"]["wheel_sha256"]
        )
        build = json.loads(docker("run", "--rm", image, "cat", "/app/build-info.json").stdout)
        assert build["commit"] == metadata["commit"]
        assert build["engine_wheel_sha256"] == metadata["engine"]["wheel_sha256"]
        assert build["service_wheel_sha256"] == metadata["service"]["wheel_sha256"]

    def test_the_installed_engine_is_the_wheel_of_the_commit(
        self, image: str, metadata: dict[str, Any]
    ) -> None:
        code = (
            "import json, civiqhub_selene, importlib.metadata as m\n"
            "print(json.dumps({'engine': civiqhub_selene.__version__,"
            " 'service': m.version('selene-api'),"
            " 'files': sorted({str(f).split('/')[0] for f in m.files('civiqhub-selene')})}))"
        )
        found = json.loads(docker("run", "--rm", image, "python", "-c", code).stdout)
        assert found["engine"] == metadata["engine"]["version"]
        assert found["service"] == metadata["service"]["version"]
        assert "civiqhub_selene" in found["files"]
        wheel = next((ROOT / "dist").glob("civiqhub_selene-*.whl"))
        import hashlib

        assert hashlib.sha256(wheel.read_bytes()).hexdigest() == metadata["engine"]["wheel_sha256"]

    def test_the_image_runs_as_a_user_without_tools_to_change_it(self, image: str) -> None:
        config = json.loads(
            docker("image", "inspect", "--format", "{{json .Config}}", image).stdout
        )
        assert config["User"] == "10001:10001"
        assert config["Cmd"] == ["python", "-m", "selene_api"]
        assert set(config["ExposedPorts"]) == {"8443/tcp", "9464/tcp"}
        assert not any(
            name.startswith(("SELENE_TOKEN", "SELENE_TLS"))
            for name in "".join(config["Env"]).split("=")
        )
        found = docker(
            "run",
            "--rm",
            "--read-only",
            image,
            "sh",
            "-c",
            "id -u; python -m pip --version 2>&1 | tail -1; ls /app; touch /app/x 2>&1 || true",
        ).stdout
        assert found.splitlines()[0] == "10001"
        assert "No module named pip" in found
        assert "Read-only file system" in found

    def test_the_image_holds_no_secret_and_no_data(self, image: str) -> None:
        listing = docker(
            "run", "--rm", image, "sh", "-c", "find / -xdev -type f 2>/dev/null; true"
        ).stdout.split()
        secrets = (".pem", ".key", ".p12", "id_rsa", "id_ed25519", ".env")
        assert [path for path in listing if path.endswith(secrets)] == []
        ours = [path for path in listing if path.startswith("/app/")]
        assert not [path for path in ours if path.endswith((".csv", ".parquet", ".xlsx"))]
        assert all(
            path.startswith("/app/contract/fixtures/")
            or path
            in {"/app/build-info.json", "/app/openapi.yaml", "/app/contract/selene_contract.py"}
            for path in ours
        ), ours

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"ENGINE_WHEEL_SHA256": "0" * 64}, "FAILED"),
            ({"ENGINE_VERSION": "9.9.9"}, "is missing"),
            ({"SERVICE_VERSION": "9.9.9"}, "is missing"),
            ({"COMMIT": ""}, "build argument COMMIT is required"),
        ],
    )
    def test_a_wrong_or_missing_wheel_fails_the_build(
        self, metadata: dict[str, Any], change: dict[str, str], message: str
    ) -> None:
        arguments = {
            "ENGINE_VERSION": metadata["engine"]["version"],
            "ENGINE_WHEEL_SHA256": metadata["engine"]["wheel_sha256"],
            "SERVICE_VERSION": metadata["service"]["version"],
            "SERVICE_WHEEL_SHA256": metadata["service"]["wheel_sha256"],
            "CONTRACT_VERSION": metadata["contract"]["version"],
            "COMMIT": metadata["commit"],
            **change,
        }
        command = ["docker", "build", "--file", "services/api/Dockerfile", "--target", "build"]
        base = metadata.get("python_image")
        if base:
            command += ["--build-arg", f"PYTHON_IMAGE={base}"]
        for key, value in arguments.items():
            command += ["--build-arg", f"{key}={value}"]
        result = subprocess.run(
            [*command, "."], cwd=ROOT, capture_output=True, text=True, timeout=600, check=False
        )
        assert result.returncode != 0
        assert message in result.stdout + result.stderr


class TestService:
    """A hardened container of the image serves contract v1 in two configurations."""

    def test_the_same_image_serves_dev_and_staging(self, deploy: Deploy) -> None:
        dev = deploy("dev")
        staging = deploy("staging")
        for deployment in (dev, staging):
            reports = selene_contract.run_check(deployment.client(), wait_timeout=120)
            assert [entry for entry in reports if not entry["passed"]] == []
            info = deployment.client().service().body
            assert info["deployment"]["environment"] == deployment.environment
        assert staging.client(dev.token).service().status == 401
        assert dev.client(staging.token).service().status == 401
        with pytest.raises(urllib.error.URLError, match="CERTIFICATE_VERIFY_FAILED"):
            selene_contract.SeleneClient(
                f"https://127.0.0.1:{staging.port}",
                staging.token,
                ca_file=str(dev.certificates.authority),
            ).health()

    def test_a_computation_in_the_image_matches_the_local_reference(self, deploy: Deploy) -> None:
        deployment = deploy("dev")
        submission, reference = selene_contract.load_reference(
            make_fixtures.FIXTURES / "coclustering-single-1" / "reference"
        )
        client = deployment.client()
        answer = client.submit({**submission, "submission_key": "test-only:image:0001"})
        done = client.wait(answer.body["id"], timeout=120)
        assert selene_contract.compare_with_reference(done.body, reference) == []
        info = client.service().body
        assert info["engine"]["wheel_sha256"] is not None
        assert info["build"]["commit"] is not None

    def test_the_client_inside_the_image_runs_the_acceptance(
        self, deploy: Deploy, image: str, network: str
    ) -> None:
        deployment = deploy("dev")
        files = pathlib.Path(deployment.certificates.authority).parent
        result = docker(
            "run",
            "--rm",
            "--network",
            network,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--volume",
            f"{files}:/etc/selene:ro",
            image,
            "python",
            "/app/contract/selene_contract.py",
            "check",
            "--url",
            f"https://{deployment.name}:8443",
            "--ca-file",
            "/etc/selene/ca.pem",
            "--token-file",
            "/etc/selene/token",
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert all(entry["passed"] for entry in json.loads(result.stdout)["checks"])

    def test_the_logs_carry_no_secret_and_no_data(self, deploy: Deploy) -> None:
        deployment = deploy("dev")
        selene_contract.run_check(deployment.client(), wait_timeout=120)
        deployment.client("test-only-wrong-token-0123456789-0123456789").service()
        logs = deployment.logs()
        payload = make_fixtures.reference_submission()["payload"]
        forbidden = [
            deployment.token,
            "test-only-wrong-token",
            *payload["voters"],
            payload["scope"]["project"],
        ]
        assert [value for value in forbidden if value in logs] == []
        for line in logs.splitlines():
            json.loads(line)


class TestLimits:
    """Busy, timeout, crash, memory, restart and stop, on the real image."""

    def test_busy_then_timeout_stops_the_computation(self, deploy: Deploy) -> None:
        deployment = deploy("dev", SELENE_JOB_TIMEOUT_SECONDS="3")
        client = deployment.client()
        first = client.submit(slow_submission())
        assert first.status == 202
        wait_for_workers(deployment)
        busy = client.submit(slow_submission("test-only:slow:0002"))
        assert (busy.status, busy.error_code) == (503, "busy")
        assert client.health().status == 200
        started = time.monotonic()
        done = client.wait(first.body["id"], timeout=30, interval=0.1)
        with report("timeout") as measured:
            measured["seconds_after_submit_to_failed"] = round(time.monotonic() - started, 2)
        assert done.body["error"]["code"] == "timeout"
        assert deployment.worker_pids() == []
        assert metric(deployment.metrics(), "selene_worker_stops_total", reason="timeout") == 1

    def test_a_crash_frees_the_slot(self, deploy: Deploy) -> None:
        deployment = deploy("dev")
        client = deployment.client()
        first = client.submit(slow_submission())
        pids = wait_for_workers(deployment)
        deployment.python(f"import os, signal; [os.kill(p, signal.SIGKILL) for p in {pids}]")
        done = client.wait(first.body["id"], timeout=30, interval=0.1)
        assert (done.body["error"]["code"], done.body["error"]["retryable"]) == (
            "worker_crashed",
            True,
        )
        assert deployment.worker_pids() == []
        again = client.submit(slow_submission("test-only:slow:0003"))
        assert again.status == 202
        assert client.delete(again.body["id"]).status == 204
        assert deployment.worker_pids() == []

    def test_memory_beyond_the_limit_fails_the_job_not_the_container(self, deploy: Deploy) -> None:
        deployment = deploy(
            "dev",
            SELENE_WORKER_MEMORY_BYTES=str(256 * 1024 * 1024),
            SELENE_MAX_CELLS="2000000",
            SELENE_MAX_VOTERS="20000",
            SELENE_MAX_ITEMS="5000",
            SELENE_MAX_WORK="10000000000000",
        )
        body = make_fixtures.reference_submission()
        body["submission_key"] = "test-only:memory:0001"
        payload = body["payload"]
        payload["voters"] = [f"vtr-{number:05d}" for number in range(4000)]
        payload["items"] = [{"id": f"itm-{number:04d}", "revision": "r1"} for number in range(500)]
        payload["votes"] = [[row, row % 500, row % 5 - 2] for row in range(4000)]
        payload["exposure"] = {"mode": "complete"}
        payload["parameters"] = {"n_starts": 1, "max_iterations": 1}
        client = deployment.client()
        answer = client.submit(body)
        assert answer.status == 202, answer.body
        done = client.wait(answer.body["id"], timeout=120, interval=0.2)
        assert done.body["error"]["code"] == "memory_exceeded"
        state = docker(
            "inspect", "--format", "{{.State.OOMKilled}} {{.RestartCount}}", deployment.name
        ).stdout
        assert state.split() == ["false", "0"]
        assert client.health().status == 200

    def test_a_restart_forgets_the_jobs(self, deploy: Deploy) -> None:
        deployment = deploy("dev")
        client = deployment.client()
        submission, _reference = selene_contract.load_reference(
            make_fixtures.FIXTURES / "coclustering-single-1" / "reference"
        )
        answer = client.submit({**submission, "submission_key": "test-only:restart:0001"})
        client.wait(answer.body["id"], timeout=60)
        docker("restart", "--timeout", "20", deployment.name)
        deployment.wait_ready()
        lost = client.job(answer.body["id"])
        assert (lost.status, lost.error_code) == (404, "job_not_found")

    def test_stop_during_a_computation_is_graceful_and_bounded(self, deploy: Deploy) -> None:
        deployment = deploy("dev")
        client = deployment.client()
        assert client.submit(slow_submission()).status == 202
        wait_for_workers(deployment)
        with report("resources-under-computation") as measured:
            stats = docker("stats", "--no-stream", "--format", "{{json .}}", deployment.name).stdout
            measured["docker_stats"] = json.loads(stats)
            peak = docker("exec", deployment.name, "cat", "/sys/fs/cgroup/memory.peak", check=False)
            measured["cgroup_memory_peak_bytes"] = peak.stdout.strip() or None
            started = time.monotonic()
            docker("stop", "--timeout", "30", deployment.name)
            measured["stop_seconds"] = round(time.monotonic() - started, 2)
        assert measured["stop_seconds"] < 20
        logs = deployment.logs().splitlines()
        assert json.loads(logs[-1])["event"] == "service_stopped"
        code = docker("inspect", "--format", "{{.State.ExitCode}}", deployment.name).stdout.strip()
        assert code in {"0", "143"}
