# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""The versions of a running instance, kept apart.

- the service: the version of the ``selene-api`` package;
- the contract: ``info.version`` of ``openapi.yaml``;
- the engine: ``civiqhub_selene.__version__``, and the SHA-256 of the wheel the image
  installed, written with the commit in the build information file of the image.

The digest of the image is not known from inside it; the deployment reads it from the
running pod.
"""

import json
import pathlib
import re
from dataclasses import dataclass
from importlib import metadata

import civiqhub_selene
from selene_api.schemas import CONTRACT_VERSION

__all__ = ["BuildInfo", "Versions", "service_version"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def service_version() -> str:
    """Return the version of the installed ``selene-api`` package."""
    try:
        return metadata.version("selene-api")
    except metadata.PackageNotFoundError:
        return "0+unknown"


@dataclass(frozen=True)
class BuildInfo:
    """Where the running code comes from; all ``None`` outside an image."""

    commit: str | None = None
    source: str | None = None
    engine_wheel_sha256: str | None = None

    @classmethod
    def read(cls, path: pathlib.Path) -> "BuildInfo":
        """Read the build information file of the image; a missing file gives no information."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(data, dict):
            return cls()
        commit = data.get("commit")
        source = data.get("source")
        wheel = data.get("engine_wheel_sha256")
        return cls(
            commit=commit if isinstance(commit, str) and _COMMIT.fullmatch(commit) else None,
            source=source if isinstance(source, str) and len(source) < 256 else None,
            engine_wheel_sha256=wheel
            if isinstance(wheel, str) and _SHA256.fullmatch(wheel)
            else None,
        )


@dataclass(frozen=True)
class Versions:
    """The versions of this instance."""

    service: str
    contract: str
    engine: str
    build: BuildInfo

    @classmethod
    def current(cls, build: BuildInfo) -> "Versions":
        """Return the versions of the code that runs."""
        return cls(service_version(), CONTRACT_VERSION, civiqhub_selene.__version__, build)

    def of_job(self) -> dict[str, str]:
        """The versions every job reports."""
        return {
            "service_version": self.service,
            "contract_version": self.contract,
            "engine_version": self.engine,
        }
