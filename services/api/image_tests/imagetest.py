# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Helpers of the image tests. TEST ONLY."""

import contextlib
import dataclasses
import json
import pathlib
import subprocess
import time
import urllib.request
from collections.abc import Iterator
from typing import Any

import selene_contract
from servicetest import Certificates

ROOT = pathlib.Path(__file__).resolve().parents[3]

HARDENED = (
    "--read-only",
    "--tmpfs",
    "/tmp:rw,noexec,nosuid,size=64m,mode=1777",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--pids-limit",
    "128",
    "--memory",
    "1536m",
)
"""The restrictions every container of these tests runs under."""


def docker(
    *arguments: str, check: bool = True, timeout: float = 300
) -> subprocess.CompletedProcess[str]:
    """Run a docker command and return its result."""
    return subprocess.run(
        ["docker", *arguments], capture_output=True, text=True, check=check, timeout=timeout
    )


@dataclasses.dataclass
class Deployment:
    """One container of the image, configured as one environment would configure it."""

    name: str
    environment: str
    token: str
    certificates: Certificates
    port: int
    metrics_port: int

    def client(self, token: str | None = None, **options: Any) -> selene_contract.SeleneClient:
        """A contract client that trusts the synthetic CA of this deployment."""
        options.setdefault("timeout", 30)
        return selene_contract.SeleneClient(
            f"https://127.0.0.1:{self.port}",
            self.token if token is None else token,
            ca_file=str(self.certificates.authority),
            **options,
        )

    def metrics(self) -> str:
        """The metrics page of this container."""
        address = f"http://127.0.0.1:{self.metrics_port}/metrics"
        with urllib.request.urlopen(address, timeout=10) as answer:
            text: str = answer.read().decode()
            return text

    def logs(self) -> str:
        """Everything the container wrote."""
        result = docker("logs", self.name, check=False)
        return result.stdout + result.stderr

    def python(self, code: str) -> str:
        """Run Python code inside the container, as its user, and return its output."""
        return docker("exec", self.name, "python", "-c", code).stdout

    def worker_pids(self) -> list[int]:
        """The computation processes running in the container."""
        code = (
            "import os, pathlib\n"
            "for entry in pathlib.Path('/proc').iterdir():\n"
            "    if entry.name.isdigit() and int(entry.name) != os.getpid():\n"
            "        try:\n"
            "            arguments = (entry / 'cmdline').read_bytes().split(b'\\0')\n"
            "        except OSError:\n"
            "            continue\n"
            "        if arguments[1:3] == [b'-m', b'selene_api.worker']:\n"
            "            print(entry.name)\n"
        )
        return [int(line) for line in self.python(code).split()]

    def wait_ready(self, timeout: float = 90) -> None:
        """Wait until the container answers ready, or fail with its logs."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                if self.client().readiness().status == 200:
                    return
            except OSError:
                pass
            state = docker("inspect", "--format", "{{.State.Status}}", self.name, check=False)
            if (
                state.stdout.strip() not in {"running", "created", "restarting"}
                or time.monotonic() > deadline
            ):
                raise AssertionError(f"{self.name} is not ready:\n{self.logs()}")
            time.sleep(0.2)


@contextlib.contextmanager
def report(name: str) -> Iterator[dict[str, Any]]:
    """Collect measurements under ``dist/image-report/<name>.json``, for the CI artifact."""
    data: dict[str, Any] = {}
    try:
        yield data
    finally:
        directory = ROOT / "dist" / "image-report"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
