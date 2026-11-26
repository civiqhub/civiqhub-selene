# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Fixtures of the image tests, loaded by conftest.py as a plugin. TEST ONLY."""

import json
import os
import pathlib
import secrets
from collections.abc import Iterator
from typing import Any

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]

from imagetest import HARDENED, Deployment, docker  # noqa: E402
from servicetest import free_port, make_certificates  # noqa: E402


@pytest.fixture(scope="session")
def image() -> str:
    """The image under test, from ``SELENE_IMAGE``."""
    name = os.environ.get("SELENE_IMAGE")
    if not name:
        pytest.fail("SELENE_IMAGE names no image: build one with services/api/image/build.sh")
    docker("image", "inspect", name)
    return name


@pytest.fixture(scope="session")
def metadata() -> dict[str, Any]:
    """What build.sh recorded of the image: commit, versions, wheel hashes, identifier."""
    path = pathlib.Path(
        os.environ.get("SELENE_IMAGE_METADATA", ROOT / "dist" / "image-metadata.json")
    )
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="session")
def network() -> Iterator[str]:
    """A dedicated bridge network for the containers of the session."""
    name = f"selene-test-{secrets.token_hex(4)}"
    docker("network", "create", name)
    try:
        yield name
    finally:
        docker("network", "rm", name, check=False)


@pytest.fixture
def deploy(image: str, network: str, tmp_path: pathlib.Path) -> Iterator[Any]:
    """Return a function that starts a container of the image; every one is removed after."""
    started: list[str] = []

    def start(
        environment: str = "dev", *, extra: tuple[str, ...] = (), **variables: str
    ) -> Deployment:
        name = f"selene-{environment}-{secrets.token_hex(3)}"
        files = tmp_path / name
        certificates, _authority = make_certificates(files, hosts=("127.0.0.1", "selene", name))
        token = f"test-only-{environment}-{secrets.token_urlsafe(32)}"
        (files / "token").write_text(token + "\n", encoding="ascii")
        for path in files.iterdir():
            path.chmod(0o644)
        port, metrics_port = free_port(), free_port()
        settings = {
            "SELENE_TOKEN_FILE": "/etc/selene/token",
            "SELENE_TLS_CERT_FILE": "/etc/selene/tls.crt",
            "SELENE_TLS_KEY_FILE": "/etc/selene/tls.key",
            "SELENE_ENVIRONMENT": environment,
            **variables,
        }
        command = ["run", "--detach", "--name", name, "--network", network, *HARDENED, *extra]
        command += [
            "--publish",
            f"127.0.0.1:{port}:8443",
            "--publish",
            f"127.0.0.1:{metrics_port}:9464",
        ]
        command += ["--volume", f"{files}:/etc/selene:ro"]
        for key, value in settings.items():
            command += ["--env", f"{key}={value}"]
        docker(*command, image)
        started.append(name)
        deployment = Deployment(name, environment, token, certificates, port, metrics_port)
        deployment.wait_ready()
        return deployment

    try:
        yield start
    finally:
        for name in started:
            docker("rm", "--force", name, check=False)
