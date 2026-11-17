# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import importlib
import pathlib
import re
import subprocess
import sys
import tomllib
from importlib.metadata import version
from importlib.resources import files

import pytest

import civiqhub_selene

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_version_matches_installed_metadata() -> None:
    assert civiqhub_selene.__version__ == version("civiqhub-selene")


def test_version_matches_pyproject() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == civiqhub_selene.__version__


def test_version_matches_citation() -> None:
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r'^version:\s*"?([^"\s]+)"?\s*$', text, re.MULTILINE)
    assert match is not None
    assert match.group(1) == civiqhub_selene.__version__


@pytest.mark.parametrize("name", ["types", "ordinal", "semantic", "landscape"])
def test_subpackage_imports(name: str) -> None:
    importlib.import_module(f"civiqhub_selene.{name}")


def test_package_is_typed() -> None:
    assert files("civiqhub_selene").joinpath("py.typed").is_file()


CORE_NAMES = [
    "Estimator",
    "Result",
    "Unavailable",
    "UnavailableReason",
    "Provenance",
    "Method",
    "Randomness",
    "Uncertainty",
    "StandardError",
    "Interval",
    "Registry",
    "EstimatorSpec",
    "default_registry",
]

ERROR_NAMES = [
    "SeleneError",
    "InvalidInputError",
    "InvalidParameterError",
    "UnknownEstimatorError",
    "UnavailableError",
    "MissingExtraError",
    "EstimatorDefinitionError",
    "ContractError",
]


def test_public_names_of_the_core() -> None:
    assert sorted(civiqhub_selene.__all__) == sorted([*CORE_NAMES, *ERROR_NAMES, "__version__"])
    for name in civiqhub_selene.__all__:
        assert hasattr(civiqhub_selene, name)


@pytest.mark.parametrize("module", ["civiqhub_selene", "civiqhub_selene.ordinal"])
def test_core_imports_neither_numpy_nor_the_sdk(module: str) -> None:
    script = (
        "import sys\n"
        f"import {module}\n"
        "loaded = [name for name in ('numpy', 'civiqhub', 'pytest') if name in sys.modules]\n"
        "assert not loaded, f'{loaded} loaded by the core'\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
