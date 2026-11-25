# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""The service is packaged apart from the engine, with locked and hashed dependencies."""

import pathlib
import re
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

SERVICE = pathlib.Path(__file__).resolve().parents[1]
ROOT = SERVICE.parents[1]


def project(path: pathlib.Path) -> dict[str, object]:
    data: dict[str, object] = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
    return data


def requirements(path: pathlib.Path) -> list[Requirement]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Requirement(line) for line in lines if line and not line.startswith("#")]


def locked() -> dict[str, tuple[str, int]]:
    """Return each locked package with its version and its number of hashes."""
    text = (SERVICE / "requirements.lock").read_text(encoding="utf-8")
    pins: dict[str, tuple[str, int]] = {}
    for block in re.split(r"\n(?=[A-Za-z])", text):
        match = re.match(r"([A-Za-z0-9._-]+)==([^\s\\]+)", block)
        if match:
            pins[canonicalize_name(match.group(1))] = (
                match.group(2),
                block.count("--hash=sha256:"),
            )
    return pins


def test_the_service_is_never_published_on_pypi() -> None:
    service = project(SERVICE / "pyproject.toml")
    assert "Private :: Do Not Upload" in service["classifiers"]  # type: ignore[operator]


def test_the_engine_package_holds_no_service_code() -> None:
    build = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["hatch"]
    assert build["build"]["targets"]["wheel"]["packages"] == ["src/civiqhub_selene"]
    assert not any("services" in entry for entry in build["build"]["targets"]["sdist"]["include"])


def test_the_lock_lists_exactly_the_runtime_dependencies() -> None:
    service = [Requirement(text) for text in project(SERVICE / "pyproject.toml")["dependencies"]]  # type: ignore[attr-defined]
    engine = project(ROOT / "pyproject.toml")
    blocks = [Requirement(text) for text in engine["optional-dependencies"]["blocks"]]  # type: ignore[index]
    expected = {
        canonicalize_name(requirement.name)
        for requirement in [*service, *blocks]
        if canonicalize_name(requirement.name) != "civiqhub-selene"
    }
    listed = {
        canonicalize_name(requirement.name)
        for requirement in requirements(SERVICE / "requirements.in")
    }
    assert listed == expected


def test_every_pin_satisfies_every_declared_range_and_has_hashes() -> None:
    pins = locked()
    declared = [
        *requirements(SERVICE / "requirements.in"),
        *(Requirement(text) for text in project(SERVICE / "pyproject.toml")["dependencies"]),  # type: ignore[attr-defined]
    ]
    for requirement in declared:
        name = canonicalize_name(requirement.name)
        if name == "civiqhub-selene":
            continue
        version, hashes = pins[name]
        assert requirement.specifier.contains(version, prereleases=False), (name, version)
        assert hashes >= 1, name
    assert all(hashes >= 1 for _version, hashes in pins.values())
    assert "civiqhub-selene" not in pins, "the engine comes from the wheel of the same commit"
