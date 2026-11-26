# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Tests of the SELENE service: its contract, its server and its image.

Run them from the root of the repository with ``pytest services/api/tests``. They use
synthetic data only. The contract client and the fixture generator live under
``services/api/contract`` and are imported from there, as a consumer would.
"""

import json
import pathlib
import sys
from collections.abc import Callable
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator
from referencing import Registry
from referencing.jsonschema import DRAFT202012

SERVICE = pathlib.Path(__file__).resolve().parents[1]
CONTRACT_DIR = SERVICE / "contract"
FIXTURES = CONTRACT_DIR / "fixtures" / "v1"
OPENAPI = SERVICE / "openapi.yaml"
OPENAPI_URI = "urn:selene:openapi:v1"

# The contract client and the fixture generator, imported as a consumer would, and the
# helpers of these tests (servicetest.py).
for directory in (CONTRACT_DIR, pathlib.Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


@pytest.fixture(scope="session")
def openapi() -> dict[str, Any]:
    """The contract document."""
    document: dict[str, Any] = yaml.safe_load(OPENAPI.read_text(encoding="utf-8"))
    return document


@pytest.fixture(scope="session")
def schema_validator(openapi: dict[str, Any]) -> Callable[[str], Draft202012Validator]:
    """Return a validator of one component schema of the contract, by name."""
    resource = DRAFT202012.create_resource(openapi)
    registry: Registry[Any] = Registry().with_resource(OPENAPI_URI, resource)

    def validator(name: str) -> Draft202012Validator:
        return Draft202012Validator(
            {"$ref": f"{OPENAPI_URI}#/components/schemas/{name}"},
            registry=registry,
            format_checker=Draft202012Validator.FORMAT_CHECKER,
        )

    return validator


@pytest.fixture(scope="session")
def fixtures_dir() -> pathlib.Path:
    """The fixtures of contract v1."""
    return FIXTURES


@pytest.fixture(scope="session")
def read_fixture() -> Callable[[str], Any]:
    """Return the decoded content of a fixture, by its path under ``fixtures/v1``."""

    def read(relative: str) -> Any:
        return json.loads((FIXTURES / relative).read_text(encoding="utf-8"))

    return read
