# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the tests of the service. TEST ONLY: never imported by the service.

:class:`FakeRunner` stands in for the computation process where a test needs to hold a
job running or to choose its outcome. It is not a computation: the service itself only
ever runs :class:`selene_api.runner.ProcessRunner`.
"""

import asyncio
import contextlib
import dataclasses
import datetime
import os
import pathlib
import signal
import socket
import ssl
import subprocess
import sys
import time
import urllib.request
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

import httpx

import make_fixtures
import selene_contract
from selene_api.app import Service, create_app
from selene_api.config import Settings
from selene_api.runner import ProcessRunner, Runner, RunOutcome
from selene_api.security import TokenStore
from selene_api.versions import BuildInfo, Versions

TOKEN = "test-only-token-" + "0123456789abcdef" * 2
"""A synthetic machine token of 48 characters."""

AUTH = {"Authorization": f"Bearer {TOKEN}"}


def settings_for(directory: pathlib.Path, **overrides: Any) -> Settings:
    """Return loopback settings with a token file in ``directory``."""
    token_file = directory / "token"
    token_file.write_text(TOKEN + "\n", encoding="ascii")
    settings = Settings(token_file=token_file, host="127.0.0.1", port=18443, metrics_port=19464)
    return dataclasses.replace(settings, **overrides)


def service_for(settings: Settings, runner: Runner | None = None) -> Service:
    """Assemble a service around ``runner``, by default the real process runner."""
    runner = runner or ProcessRunner(max_result_bytes=settings.max_result_bytes)
    versions = Versions.current(BuildInfo())
    return Service.build(
        settings, TokenStore(settings.token_file, token=settings.token), runner, versions
    )


@contextlib.asynccontextmanager
async def asgi_client(service: Service) -> AsyncIterator[httpx.AsyncClient]:
    """Run the application in process, with its start-up and shutdown, and a client."""
    app = create_app(service)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://selene.test") as client:
            yield client


class FakeRunner:
    """A runner whose computations wait for :meth:`release`, then return ``outcome``."""

    def __init__(self, outcome: RunOutcome | None = None) -> None:
        self.outcome = outcome or RunOutcome(result={"schema": "fake", "available": True})
        self.started: list[Mapping[str, Any]] = []
        self.cancelled = 0
        self.timeouts: list[float] = []
        self._gate = asyncio.Event()
        self.hold = False

    def release(self) -> None:
        """Let every held computation finish."""
        self._gate.set()

    async def run(self, task: Mapping[str, Any], *, timeout: float) -> RunOutcome:
        self.started.append(task)
        self.timeouts.append(timeout)
        try:
            if self.hold:
                await self._gate.wait()
            else:
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return self.outcome

    async def describe(self, *, timeout: float) -> dict[str, Any]:
        from civiqhub_selene import __version__
        from selene_api.adapters import ADAPTERS

        return {
            "engine_version": __version__,
            "kinds": [
                {
                    "kind": kind,
                    "schema_version": version,
                    "method": {"name": "latent_block_model", "version": "fake"},
                }
                for kind, version in sorted(ADAPTERS)
            ],
        }


async def settle(predicate: Any, *, timeout: float = 30.0, interval: float = 0.01) -> None:
    """Wait until ``predicate()`` is true."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("the condition did not hold in time")
        await asyncio.sleep(interval)


# A real server process ------------------------------------------------------------------

# After a graceful shutdown, Uvicorn re-raises the signal it caught, so that the process
# manager sees an exit on SIGTERM; the log says whether the shutdown was graceful.
STOPPED_BY_SIGTERM = (0, -signal.SIGTERM)


def free_port() -> int:
    """Return a TCP port free on the loopback interface."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


def environment(directory: pathlib.Path, **variables: str) -> dict[str, str]:
    """Return the environment of a server on the loopback interface, with a token file."""
    token_file = directory / "token"
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
    """Start ``python -m selene_api`` with ``env``, its output going to ``log``."""
    with log.open("wb") as output:
        return subprocess.Popen(
            [sys.executable, "-m", "selene_api"], env=env, stdout=output, stderr=subprocess.STDOUT
        )


def wait_ready(
    client: selene_contract.SeleneClient,
    process: subprocess.Popen[bytes],
    log: pathlib.Path,
    *,
    verify_host: bool = True,
) -> None:
    """Wait until ``/readyz`` answers 200, or fail with the log of the server.

    With ``verify_host`` false, the certificate chain is still verified, not its name:
    for a test that serves a certificate of another host on purpose.
    """
    if not verify_host and client._context is not None:
        client = client.with_token(None)
        context = ssl.create_default_context(cafile=client.ca_file)
        context.check_hostname = False
        client._context = context
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


@dataclasses.dataclass
class RunningServer:
    """A ``python -m selene_api`` process and how to reach it."""

    process: subprocess.Popen[bytes]
    log: pathlib.Path
    env: dict[str, str]
    scheme: str
    ca_file: pathlib.Path | None

    @property
    def url(self) -> str:
        return f"{self.scheme}://127.0.0.1:{self.env['SELENE_PORT']}"

    def client(self, token: str | None = TOKEN, **options: Any) -> selene_contract.SeleneClient:
        """Return a contract client of this server."""
        options.setdefault("ca_file", None if self.ca_file is None else str(self.ca_file))
        options.setdefault("timeout", 30)
        return selene_contract.SeleneClient(
            self.url, token, allow_insecure_http=self.scheme == "http", **options
        )

    def metrics(self) -> str:
        """Return the metrics page of this server."""
        address = f"http://127.0.0.1:{self.env['SELENE_METRICS_PORT']}/metrics"
        with urllib.request.urlopen(address, timeout=10) as answer:
            text: str = answer.read().decode()
            return text

    def worker_pids(self) -> list[int]:
        """Return the computation processes this server started and that still run."""
        pids = []
        for entry in pathlib.Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                status = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                command = (entry / "cmdline").read_bytes()
            except OSError:
                continue
            if int(status[1]) == self.process.pid and b"selene_api.worker" in command:
                pids.append(int(entry.name))
        return pids

    def stop(self) -> int | None:
        """Send SIGTERM and return the exit status."""
        self.process.send_signal(signal.SIGTERM)
        return self.process.wait(timeout=60)


@contextlib.contextmanager
def running_server(
    directory: pathlib.Path,
    *,
    tls: "Certificates | None" = None,
    verify_host: bool = True,
    **variables: str,
) -> Iterator[RunningServer]:
    """Run a server process for the duration of the block, then make sure it is gone."""
    if tls is not None:
        variables.setdefault("SELENE_TLS_CERT_FILE", str(tls.certificate))
        variables.setdefault("SELENE_TLS_KEY_FILE", str(tls.key))
    env = environment(directory, **variables)
    log = directory / "server.log"
    process = start(env, log)
    server = RunningServer(
        process, log, env, "https" if tls else "http", None if tls is None else tls.authority
    )
    try:
        wait_ready(server.client(), process, log, verify_host=verify_host)
        yield server
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def slow_submission(key: str = "test-only:slow:0001") -> dict[str, Any]:
    """A synthetic submission that computes for tens of seconds within the default budget."""
    body = make_fixtures.reference_submission()
    payload = body["payload"]
    body["submission_key"] = key
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


# Certificates of a synthetic authority ----------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Certificates:
    """A synthetic certificate authority and a server certificate it signed."""

    authority: pathlib.Path
    certificate: pathlib.Path
    key: pathlib.Path
    serial: int


def make_certificates(
    directory: pathlib.Path,
    *,
    hosts: tuple[str, ...] = ("127.0.0.1", "localhost"),
    authority: tuple[Any, Any] | None = None,
    serial: int = 1,
) -> tuple[Certificates, tuple[Any, Any]]:
    """Write a CA and a server certificate for ``hosts`` under ``directory``. TEST ONLY.

    The certificates follow RFC 5280 as Python 3.13 checks it by default (key
    identifiers, key usages): a deployment certificate must do the same.

    Returns the files and the CA (certificate, key), to sign a renewed certificate.
    """
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    now = datetime.datetime.now(datetime.UTC)
    if authority is None:
        ca_key = ec.generate_private_key(ec.SECP256R1())
        ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-only selene CA")])
        ca_certificate = (
            x509.CertificateBuilder()
            .subject_name(ca_name)
            .issuer_name(ca_name)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=False,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), False)
            .sign(ca_key, hashes.SHA256())
        )
        authority = (ca_certificate, ca_key)
    ca_certificate, ca_key = authority
    key = ec.generate_private_key(ec.SECP256R1())
    names: list[x509.GeneralName] = []
    for host in hosts:
        try:
            names.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            names.append(x509.DNSName(host))
    certificate = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "selene test-only")]))
        .issuer_name(ca_certificate.subject)
        .public_key(key.public_key())
        .serial_number(serial)
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), False
        )
        .sign(ca_key, hashes.SHA256())
    )
    directory.mkdir(parents=True, exist_ok=True)
    files = Certificates(directory / "ca.pem", directory / "tls.crt", directory / "tls.key", serial)
    files.authority.write_bytes(ca_certificate.public_bytes(serialization.Encoding.PEM))
    files.certificate.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    files.key.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return files, authority
