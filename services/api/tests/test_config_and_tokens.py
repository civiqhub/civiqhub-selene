# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Settings come from the environment, inside the contract bounds; tokens fail closed."""

import os
import pathlib
from typing import Any

import pytest
import yaml

from selene_api import logs
from selene_api.config import CONTRACT_BOUNDS, ConfigError, Settings
from selene_api.schemas import CONTRACT_VERSION
from selene_api.security import TokenFileError, TokenStore, bearer_token
from selene_api.versions import BuildInfo, Versions
from servicetest import TOKEN

OTHER = "test-only-next-token-" + "fedcba9876543210" * 2


def write_token(directory: pathlib.Path, text: str = TOKEN + "\n") -> pathlib.Path:
    path = directory / "token"
    path.write_text(text, encoding="ascii")
    return path


def test_the_defaults_are_those_of_the_readme(tmp_path: pathlib.Path) -> None:
    settings = Settings.from_environ(
        {"SELENE_TOKEN_FILE": str(write_token(tmp_path)), "SELENE_HOST": "127.0.0.1"}
    )
    assert (settings.port, settings.metrics_port) == (8443, 9464)
    assert settings.job_timeout_seconds == 120
    assert settings.result_ttl_seconds == 900
    assert settings.max_retained_jobs == 16
    assert settings.max_body_bytes == 8 * 1024 * 1024
    assert (settings.max_voters, settings.max_items, settings.max_cells) == (2000, 500, 250_000)
    assert settings.environment is None
    assert not settings.tls
    assert settings.tls_terminated_by_proxy is False
    assert settings.token is None
    assert TOKEN not in repr(settings)


def test_tls_settings_go_together_and_open_the_network(tmp_path: pathlib.Path) -> None:
    token = str(write_token(tmp_path))
    cert = tmp_path / "tls.crt"
    key = tmp_path / "tls.key"
    cert.write_text("cert")
    key.write_text("key")
    settings = Settings.from_environ(
        {
            "SELENE_TOKEN_FILE": token,
            "SELENE_TLS_CERT_FILE": str(cert),
            "SELENE_TLS_KEY_FILE": str(key),
        }
    )
    assert settings.tls
    assert settings.host == "0.0.0.0"
    with pytest.raises(ConfigError, match="go together"):
        Settings.from_environ({"SELENE_TOKEN_FILE": token, "SELENE_TLS_CERT_FILE": str(cert)})


@pytest.mark.parametrize(
    ("variables", "message"),
    [
        ({"SELENE_PORT": "9464"}, "SELENE_PORT and SELENE_METRICS_PORT differ"),
        ({"SELENE_ENVIRONMENT": "dev staging"}, "short label"),
        ({"SELENE_LOG_LEVEL": "trace"}, "SELENE_LOG_LEVEL"),
        ({"SELENE_MAX_CELLS": "2000000", "SELENE_MAX_VOTERS": "10"}, "MAX_CELLS is at most"),
        ({"SELENE_MAX_BODY_BYTES": "0"}, "between 1024"),
    ],
)
def test_inconsistent_settings_are_refused(
    tmp_path: pathlib.Path, variables: dict[str, str], message: str
) -> None:
    environ = {"SELENE_TOKEN_FILE": str(write_token(tmp_path)), "SELENE_HOST": "::1", **variables}
    with pytest.raises(ConfigError, match=message):
        Settings.from_environ(environ)


def test_the_bounds_never_exceed_the_contract(openapi: dict[str, Any]) -> None:
    payload = openapi["components"]["schemas"]["CoclusteringSingleV1Input"]["properties"]
    assert CONTRACT_BOUNDS["max_voters"][1] == payload["voters"]["maxItems"]
    assert CONTRACT_BOUNDS["max_items"][1] == payload["items"]["maxItems"]
    assert CONTRACT_BOUNDS["max_cells"][1] <= payload["votes"]["maxItems"]


def test_the_contract_version_is_the_one_of_openapi(openapi: dict[str, Any]) -> None:
    assert openapi["info"]["version"] == CONTRACT_VERSION


def test_the_readme_lists_every_setting() -> None:
    readme = (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    for field in Settings.__dataclass_fields__:
        if field in {"build_info_file", "log_level", "tls_key_file", "host", "metrics_port"}:
            continue
        assert f"SELENE_{field.upper()}" in readme, field


class TestTokens:
    def test_a_listed_token_passes_and_others_fail(self, tmp_path: pathlib.Path) -> None:
        store = TokenStore(write_token(tmp_path, f"# rotation\n{TOKEN}\n\n{OTHER}\n"))
        assert store.verify(TOKEN)
        assert store.verify(OTHER)
        assert not store.verify(TOKEN[:-1])
        assert not store.verify(TOKEN + "x")
        assert not store.verify(None)
        assert not store.verify("é" * 40)

    def test_the_file_is_reloaded_when_it_changes(self, tmp_path: pathlib.Path) -> None:
        path = write_token(tmp_path)
        clock = [0.0]
        store = TokenStore(path, clock=lambda: clock[0])
        replacement = tmp_path / "next"
        replacement.write_text(OTHER + "\n", encoding="ascii")
        os.replace(replacement, path)
        assert store.verify(TOKEN), "checked at most once per second"
        clock[0] = 2.0
        assert store.verify(OTHER)
        assert not store.verify(TOKEN)
        assert store.reloads == 1

    def test_an_unreadable_file_fails_closed(self, tmp_path: pathlib.Path) -> None:
        path = write_token(tmp_path)
        clock = [0.0]
        store = TokenStore(path, clock=lambda: clock[0])
        path.write_text("too-short\n", encoding="ascii")
        clock[0] = 2.0
        assert not store.verify(TOKEN)
        assert not store.usable
        path.unlink()
        clock[0] = 4.0
        assert not store.verify(TOKEN)

    @pytest.mark.parametrize("text", ["", "# only a comment\n", "short\n", f"{TOKEN} trailing\n"])
    def test_a_file_without_valid_token_is_refused_at_start(
        self, tmp_path: pathlib.Path, text: str
    ) -> None:
        with pytest.raises(TokenFileError):
            TokenStore(write_token(tmp_path, text))

    @pytest.mark.parametrize(
        ("header", "token"),
        [
            (f"Bearer {TOKEN}", TOKEN),
            (f"bearer {TOKEN}", TOKEN),
            (f"Basic {TOKEN}", None),
            ("Bearer", None),
            ("Bearer   ", None),
            (None, None),
            ("", None),
        ],
    )
    def test_only_bearer_headers_carry_a_token(self, header: str | None, token: str | None) -> None:
        assert bearer_token(header) == token


class TestBuildInfo:
    def test_a_missing_file_gives_no_information(self, tmp_path: pathlib.Path) -> None:
        assert BuildInfo.read(tmp_path / "missing.json") == BuildInfo()

    def test_only_well_formed_values_are_kept(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "build-info.json"
        path.write_text(
            yaml.safe_dump(
                {
                    "commit": "a" * 40,
                    "source": "https://github.com/civiqhub/civiqhub-selene",
                    "engine_wheel_sha256": "nope",
                }
            )
        )
        info = BuildInfo.read(path)
        assert info == BuildInfo()  # YAML is not JSON
        path.write_text('{"commit": "' + "a" * 40 + '", "engine_wheel_sha256": "' + "b" * 64 + '"}')
        info = BuildInfo.read(path)
        assert (info.commit, info.engine_wheel_sha256) == ("a" * 40, "b" * 64)

    def test_the_versions_stay_apart(self) -> None:
        versions = Versions.current(BuildInfo())
        assert versions.contract == CONTRACT_VERSION
        assert versions.service == "0.1.0"
        assert set(versions.of_job()) == {"service_version", "contract_version", "engine_version"}


ENV_TOKEN = "test-token-abcdefghijklmnopqrstuvwxyzabcdef"
OTHER_ENV = "test-token-zyxwvutsrqponmlkjihgfedcbaabcdef"


class TestTlsTerminatedByProxy:
    def test_absent_keeps_refusing_http_off_loopback(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(
            ConfigError, match="without TLS the service only listens on the loopback"
        ):
            Settings.from_environ(
                {"SELENE_TOKEN_FILE": str(write_token(tmp_path)), "SELENE_HOST": "0.0.0.0"}
            )

    @pytest.mark.parametrize("value", ["TRUE", "false", "1", "yes", "True"])
    def test_a_value_other_than_true_is_refused(self, tmp_path: pathlib.Path, value: str) -> None:
        with pytest.raises(ConfigError, match="SELENE_TLS_TERMINATED_BY_PROXY is true"):
            Settings.from_environ(
                {
                    "SELENE_TOKEN_FILE": str(write_token(tmp_path)),
                    "SELENE_HOST": "127.0.0.1",
                    "SELENE_TLS_TERMINATED_BY_PROXY": value,
                }
            )

    def test_true_allows_http_off_loopback_without_tls(self, tmp_path: pathlib.Path) -> None:
        settings = Settings.from_environ(
            {
                "SELENE_TOKEN_FILE": str(write_token(tmp_path)),
                "SELENE_HOST": "0.0.0.0",
                "SELENE_TLS_TERMINATED_BY_PROXY": "true",
            }
        )
        assert settings.tls_terminated_by_proxy is True
        assert not settings.tls
        assert settings.port == 80

    def test_true_with_both_tls_files_is_refused(self, tmp_path: pathlib.Path) -> None:
        cert = tmp_path / "tls.crt"
        key = tmp_path / "tls.key"
        cert.write_text("cert")
        key.write_text("key")
        with pytest.raises(ConfigError, match="SELENE_TLS_TERMINATED_BY_PROXY cannot be true"):
            Settings.from_environ(
                {
                    "SELENE_TOKEN_FILE": str(write_token(tmp_path)),
                    "SELENE_TLS_CERT_FILE": str(cert),
                    "SELENE_TLS_KEY_FILE": str(key),
                    "SELENE_TLS_TERMINATED_BY_PROXY": "true",
                }
            )

    def test_true_with_one_tls_file_still_requires_the_pair(self, tmp_path: pathlib.Path) -> None:
        cert = tmp_path / "tls.crt"
        cert.write_text("cert")
        with pytest.raises(ConfigError, match="go together"):
            Settings.from_environ(
                {
                    "SELENE_TOKEN_FILE": str(write_token(tmp_path)),
                    "SELENE_TLS_CERT_FILE": str(cert),
                    "SELENE_TLS_TERMINATED_BY_PROXY": "true",
                }
            )

    def test_port_comes_from_port_then_80_and_selene_port_wins(
        self, tmp_path: pathlib.Path
    ) -> None:
        token = str(write_token(tmp_path))
        base = {
            "SELENE_TOKEN_FILE": token,
            "SELENE_HOST": "0.0.0.0",
            "SELENE_TLS_TERMINATED_BY_PROXY": "true",
        }
        assert Settings.from_environ(base).port == 80
        assert Settings.from_environ({**base, "PORT": "8080"}).port == 8080
        assert Settings.from_environ({**base, "PORT": "8080", "SELENE_PORT": "9000"}).port == 9000

    def test_an_invalid_port_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ConfigError, match="PORT is an integer"):
            Settings.from_environ(
                {
                    "SELENE_TOKEN_FILE": str(write_token(tmp_path)),
                    "SELENE_HOST": "0.0.0.0",
                    "SELENE_TLS_TERMINATED_BY_PROXY": "true",
                    "PORT": "soon",
                }
            )
        with pytest.raises(ConfigError, match="PORT is between"):
            Settings.from_environ(
                {
                    "SELENE_TOKEN_FILE": str(write_token(tmp_path)),
                    "SELENE_HOST": "0.0.0.0",
                    "SELENE_TLS_TERMINATED_BY_PROXY": "true",
                    "PORT": "0",
                }
            )

    def test_port_is_ignored_outside_proxy_mode(self, tmp_path: pathlib.Path) -> None:
        settings = Settings.from_environ(
            {
                "SELENE_TOKEN_FILE": str(write_token(tmp_path)),
                "SELENE_HOST": "127.0.0.1",
                "PORT": "80",
            }
        )
        assert settings.port == 8443

    def test_the_proxy_mode_is_an_allowed_log_field(self) -> None:
        logs.event("service_starting", tls_terminated_by_proxy=True)
        with pytest.raises(ValueError, match="field not allowed"):
            logs.event("service_starting", token=ENV_TOKEN)


class TestEnvironmentToken:
    def test_a_valid_environment_token_starts_without_a_file(self) -> None:
        settings = Settings.from_environ({"SELENE_TOKEN": ENV_TOKEN, "SELENE_HOST": "127.0.0.1"})
        assert settings.token_file is None
        assert settings.token == ENV_TOKEN
        assert ENV_TOKEN not in repr(settings)
        store = TokenStore(settings.token_file, token=settings.token)
        assert store.path is None
        assert store.verify(ENV_TOKEN)
        assert not store.verify(OTHER_ENV)
        assert store.usable

    def test_an_environment_token_is_never_written(self, tmp_path: pathlib.Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        TokenStore(token=ENV_TOKEN)
        assert list(empty.iterdir()) == []

    def test_both_sources_accept_every_file_token(self, tmp_path: pathlib.Path) -> None:
        path = write_token(tmp_path, f"{TOKEN}\n{OTHER}\n")
        settings = Settings.from_environ(
            {
                "SELENE_TOKEN_FILE": str(path),
                "SELENE_TOKEN": TOKEN,
                "SELENE_HOST": "127.0.0.1",
            }
        )
        store = TokenStore(settings.token_file, token=settings.token)
        assert store.verify(TOKEN)
        assert store.verify(OTHER)
        assert TOKEN not in repr(settings)

    def test_an_environment_token_missing_from_the_file_is_refused(
        self, tmp_path: pathlib.Path
    ) -> None:
        with pytest.raises(
            ConfigError, match="SELENE_TOKEN is not a token of SELENE_TOKEN_FILE"
        ) as caught:
            Settings.from_environ(
                {
                    "SELENE_TOKEN_FILE": str(write_token(tmp_path)),
                    "SELENE_TOKEN": OTHER_ENV,
                    "SELENE_HOST": "127.0.0.1",
                }
            )
        assert OTHER_ENV not in str(caught.value)
        assert TOKEN not in str(caught.value)

    @pytest.mark.parametrize(
        "value",
        ["test-token-short", "test-token-abcdefghijklmnopqrstuvwxyz abc"],
    )
    def test_a_malformed_environment_token_is_refused(self, value: str) -> None:
        with pytest.raises(ConfigError, match="SELENE_TOKEN is shorter than 32") as caught:
            Settings.from_environ({"SELENE_TOKEN": value, "SELENE_HOST": "127.0.0.1"})
        assert value not in str(caught.value)
