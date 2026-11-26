# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Configuration of the service, read from environment variables and checked at start-up.

The same image runs locally, in dev and in staging: only this configuration changes.
A configuration that is incomplete or out of bounds stops the start-up with a message
that names the variable, never the value of a secret. The service never starts on the
network without a token, nor without TLS off the loopback interface, unless a
TLS-terminating proxy is declared.
"""

import ipaddress
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass, fields

from selene_api.adapters import InstanceLimits
from selene_api.security import (
    MIN_TOKEN_LENGTH,
    _read_tokens,
    token_form_is_valid,
    token_listed,
)

__all__ = ["CONTRACT_BOUNDS", "ConfigError", "Settings"]

_PREFIX = "SELENE_"


class ConfigError(ValueError):
    """The configuration does not allow the service to start."""


CONTRACT_BOUNDS = {
    "max_body_bytes": (1024, 64 * 1024 * 1024),
    "max_voters": (1, 20_000),
    "max_items": (1, 5_000),
    "max_cells": (1, 2_000_000),
    "max_work": (1, 10**13),
    "max_result_bytes": (1024, 256 * 1024 * 1024),
    "job_timeout_seconds": (1, 3600),
    "result_ttl_seconds": (1, 86_400),
    "max_retained_jobs": (1, 1000),
    "shutdown_grace_seconds": (1, 300),
    "worker_memory_bytes": (64 * 1024 * 1024, 64 * 1024 * 1024 * 1024),
    "port": (1, 65_535),
    "metrics_port": (1, 65_535),
}
"""Bounds of each numeric setting: the limits of a deployment stay inside the contract."""


def _integer(variable: str, text: str, bounds: tuple[int, int]) -> int:
    try:
        number = int(text)
    except ValueError:
        raise ConfigError(f"{variable} is an integer") from None
    low, high = bounds
    if not low <= number <= high:
        raise ConfigError(f"{variable} is between {low} and {high}")
    return number


@dataclass(frozen=True)
class Settings:
    """Every setting of the service. See ``services/api/README.md``."""

    token_file: pathlib.Path | None = None
    token: str | None = None
    tls_cert_file: pathlib.Path | None = None
    tls_key_file: pathlib.Path | None = None
    tls_terminated_by_proxy: bool = False
    host: str = "0.0.0.0"
    port: int = 8443
    metrics_port: int = 9464
    environment: str | None = None
    job_timeout_seconds: int = 120
    result_ttl_seconds: int = 900
    max_retained_jobs: int = 16
    max_body_bytes: int = 8 * 1024 * 1024
    max_voters: int = 2_000
    max_items: int = 500
    max_cells: int = 250_000
    max_work: int = 20_000_000_000
    max_result_bytes: int = 16 * 1024 * 1024
    shutdown_grace_seconds: int = 10
    worker_memory_bytes: int = 1024 * 1024 * 1024
    build_info_file: pathlib.Path = pathlib.Path("/app/build-info.json")
    log_level: str = "info"

    def __repr__(self) -> str:
        parts = []
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "token":
                parts.append("token=<redacted>" if value is not None else "token=None")
            else:
                parts.append(f"{field.name}={value!r}")
        return f"Settings({', '.join(parts)})"

    @property
    def limits(self) -> InstanceLimits:
        """The limits the adapters enforce."""
        return InstanceLimits(
            max_voters=self.max_voters,
            max_items=self.max_items,
            max_cells=self.max_cells,
            max_work=self.max_work,
        )

    @property
    def worker_cpu_seconds(self) -> int:
        """CPU-time limit of a computation process: a backstop behind the timeout."""
        return self.job_timeout_seconds + 30

    @property
    def tls(self) -> bool:
        """Whether the service is served over HTTPS."""
        return self.tls_cert_file is not None

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> "Settings":
        """Read and check the settings from ``SELENE_*`` variables.

        Raises:
            ConfigError: If a setting is missing, malformed or out of bounds.
        """
        known = {f"{_PREFIX}{field.name.upper()}" for field in fields(cls)}
        unknown = sorted(name for name in environ if name.startswith(_PREFIX) and name not in known)
        if unknown:
            raise ConfigError(f"unknown setting {unknown[0]}: check its spelling")
        values: dict[str, object] = {}
        for field in fields(cls):
            variable = f"{_PREFIX}{field.name.upper()}"
            if variable not in environ:
                continue
            text = environ[variable].strip()
            if field.name in CONTRACT_BOUNDS:
                values[field.name] = _integer(variable, text, CONTRACT_BOUNDS[field.name])
            elif field.name.endswith("_file"):
                values[field.name] = pathlib.Path(text) if text else None
            elif field.name == "tls_terminated_by_proxy":
                if not text:
                    continue
                if text != "true":
                    raise ConfigError(f"{variable} is true")
                values[field.name] = True
            elif field.name == "token":
                if not text:
                    continue
                if not token_form_is_valid(text):
                    raise ConfigError(
                        f"{variable} is shorter than {MIN_TOKEN_LENGTH} characters or holds a space"
                    )
                values[field.name] = text
            else:
                values[field.name] = text or None
        if values.get("tls_terminated_by_proxy") and "port" not in values:
            port_text = environ.get("PORT")
            if port_text is not None and port_text.strip() != "":
                values["port"] = _integer("PORT", port_text.strip(), CONTRACT_BOUNDS["port"])
            else:
                values["port"] = 80
        if values.get("token_file") is None and values.get("token") is None:
            raise ConfigError(
                f"{_PREFIX}TOKEN_FILE or {_PREFIX}TOKEN is required: "
                "the service never runs without a token"
            )
        if values.get("host") is None:
            values.pop("host", None)
        if values.get("log_level") is None:
            values.pop("log_level", None)
        if values.get("build_info_file") is None:
            values.pop("build_info_file", None)
        settings = cls(**values)  # type: ignore[arg-type]
        settings.check()
        return settings

    def check(self) -> None:
        """Check the settings together and the files they name.

        Raises:
            ConfigError: If the settings cannot run together.
        """
        if self.token is None and self.token_file is None:
            raise ConfigError(
                f"{_PREFIX}TOKEN_FILE or {_PREFIX}TOKEN is required: "
                "the service never runs without a token"
            )
        if self.token_file is not None and not self.token_file.is_file():
            raise ConfigError(f"{_PREFIX}TOKEN_FILE does not name a readable file")
        if self.token is not None and self.token_file is not None:
            listed = _read_tokens(self.token_file)
            if not token_listed(listed, self.token):
                raise ConfigError(f"{_PREFIX}TOKEN is not a token of {_PREFIX}TOKEN_FILE")
        if (self.tls_cert_file is None) != (self.tls_key_file is None):
            raise ConfigError(f"{_PREFIX}TLS_CERT_FILE and {_PREFIX}TLS_KEY_FILE go together")
        for path in (self.tls_cert_file, self.tls_key_file):
            if path is not None and not path.is_file():
                raise ConfigError("a TLS file does not name a readable file")
        if self.tls_terminated_by_proxy and self.tls:
            raise ConfigError(
                f"{_PREFIX}TLS_TERMINATED_BY_PROXY cannot be true with "
                f"{_PREFIX}TLS_CERT_FILE and {_PREFIX}TLS_KEY_FILE"
            )
        if not self.tls and not _loopback(self.host) and not self.tls_terminated_by_proxy:
            raise ConfigError(
                "without TLS the service only listens on the loopback interface: set "
                f"{_PREFIX}TLS_CERT_FILE and {_PREFIX}TLS_KEY_FILE, or {_PREFIX}HOST=127.0.0.1"
            )
        if self.port == self.metrics_port:
            raise ConfigError(f"{_PREFIX}PORT and {_PREFIX}METRICS_PORT differ")
        if self.environment is not None and not _label(self.environment):
            raise ConfigError(f"{_PREFIX}ENVIRONMENT is a short label such as dev or staging")
        if self.log_level not in {"debug", "info", "warning", "error"}:
            raise ConfigError(f"{_PREFIX}LOG_LEVEL is debug, info, warning or error")
        if self.max_cells > self.max_voters * self.max_items:
            raise ConfigError(f"{_PREFIX}MAX_CELLS is at most MAX_VOTERS x MAX_ITEMS")


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _label(text: str) -> bool:
    return 0 < len(text) <= 64 and all(
        character.isalnum() or character in "._:-" for character in text
    )
