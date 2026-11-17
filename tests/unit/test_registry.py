# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types
from dataclasses import dataclass
from importlib import metadata

import pytest

from civiqhub_selene import registry as registry_module
from civiqhub_selene.errors import (
    EstimatorDefinitionError,
    InvalidParameterError,
    MissingExtraError,
    UnknownEstimatorError,
)
from civiqhub_selene.estimator import Estimator
from civiqhub_selene.registry import GROUP, EstimatorSpec, Registry, default_registry


@dataclass(frozen=True)
class Numbers:
    values: tuple[float, ...]

    def fingerprint(self) -> str:
        return repr(self.values)


@dataclass(frozen=True, kw_only=True)
class One(Estimator[Numbers, float]):
    name = "one"
    version = "1.0.0"
    reference = "r"
    input_type = Numbers

    def _compute(self, data: Numbers, *, seed: int, computation: str) -> float:
        return 1.0


@dataclass(frozen=True, kw_only=True)
class Two(One):
    name = "two"


def install_plugin(monkeypatch: pytest.MonkeyPatch, entries: dict[str, str]) -> None:
    """Install a fake module and fake entry points that point to it."""
    plugin = types.ModuleType("selene_fake_plugin")
    plugin.One = One  # type: ignore[attr-defined]
    plugin.Two = Two  # type: ignore[attr-defined]
    plugin.not_an_estimator = dict  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "selene_fake_plugin", plugin)
    installed = [
        metadata.EntryPoint(name=key, value=target, group=GROUP) for key, target in entries.items()
    ]

    def fake_entry_points(*, group: str) -> list[metadata.EntryPoint]:
        return [entry_point for entry_point in installed if entry_point.group == group]

    monkeypatch.setattr(metadata, "entry_points", fake_entry_points)


def test_group_name() -> None:
    assert GROUP == "civiqhub_selene.estimators"


def test_from_classes_loads_the_class_itself() -> None:
    registry = Registry.from_classes({"demo.one": One})
    assert registry["demo.one"].load() is One
    assert registry["demo.one"].key == "demo.one"


def test_family_and_choice() -> None:
    registry = Registry.from_classes({"demo.one": One, "demo.two": Two, "other.one": One})
    spec = registry["demo.one"]
    assert spec.family == "demo"
    assert spec.choice == "one"
    assert registry.family("demo") == {"one": spec, "two": registry["demo.two"]}
    assert registry.family("missing") == {}


def test_an_unknown_key_lists_the_choices_of_its_family() -> None:
    registry = Registry.from_classes({"demo.one": One, "demo.two": Two, "other.one": One})
    with pytest.raises(UnknownEstimatorError, match=r"'demo' offers: one, two") as caught:
        registry["demo.three"]
    assert isinstance(caught.value, LookupError)
    assert "'demo.three'" in str(caught.value)
    with pytest.raises(UnknownEstimatorError, match=r"no family 'absent' among: demo, other"):
        registry["absent.one"]
    with pytest.raises(UnknownEstimatorError, match="among: none"):
        Registry()["absent.one"]


def test_membership_and_get_do_not_raise() -> None:
    registry = Registry.from_classes({"demo.one": One})
    assert "demo.one" in registry
    assert "demo.two" not in registry
    not_a_key: object = 3
    assert not_a_key not in registry
    assert registry.get("demo.two") is None
    assert registry.get("demo.one") is registry["demo.one"]
    assert registry.get("demo.two", registry["demo.one"]) is registry["demo.one"]


def test_keys_are_sorted_and_counted() -> None:
    registry = Registry.from_classes({"demo.two": Two, "alpha.one": One, "demo.one": One})
    assert list(registry) == ["alpha.one", "demo.one", "demo.two"]
    assert len(registry) == 3
    assert repr(registry) == "Registry(alpha.one, demo.one, demo.two)"


def test_from_entry_points_on_an_empty_group_is_empty() -> None:
    registry = Registry.from_entry_points("civiqhub_selene.a_group_nobody_declares")
    assert len(registry) == 0
    assert list(registry) == []


def test_an_installed_entry_point_is_found_then_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    install_plugin(monkeypatch, {"demo.one": "selene_fake_plugin:One"})
    registry = Registry.from_entry_points()
    assert list(registry) == ["demo.one"]
    spec = registry["demo.one"]
    assert spec.target == "selene_fake_plugin:One"
    assert spec.load() is One


def test_reading_entry_points_imports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    install_plugin(monkeypatch, {"demo.one": "a_module_that_does_not_exist:One"})

    def refuse(name: str) -> types.ModuleType:
        raise AssertionError(f"{name} was imported while reading the registry")

    monkeypatch.setattr(importlib, "import_module", refuse)
    assert list(Registry.from_entry_points()) == ["demo.one"]


def test_load_keeps_the_class(monkeypatch: pytest.MonkeyPatch) -> None:
    install_plugin(monkeypatch, {"demo.one": "selene_fake_plugin:One"})
    spec = Registry.from_entry_points()["demo.one"]
    assert spec.load() is One
    monkeypatch.delitem(sys.modules, "selene_fake_plugin")
    assert spec.load() is One


def test_load_names_a_missing_module_or_class(monkeypatch: pytest.MonkeyPatch) -> None:
    install_plugin(
        monkeypatch,
        {
            "demo.module": "a_module_that_does_not_exist.inner:One",
            "demo.attribute": "selene_fake_plugin:Three",
        },
    )
    registry = Registry.from_entry_points()
    with pytest.raises(UnknownEstimatorError, match="a_module_that_does_not_exist"):
        registry["demo.module"].load()
    with pytest.raises(UnknownEstimatorError, match="has no 'Three'"):
        registry["demo.attribute"].load()


def test_load_refuses_a_target_that_is_not_an_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    install_plugin(monkeypatch, {"demo.other": "selene_fake_plugin:not_an_estimator"})
    with pytest.raises(EstimatorDefinitionError, match="not a subclass of Estimator"):
        Registry.from_entry_points()["demo.other"].load()
    with pytest.raises(EstimatorDefinitionError, match="not a subclass of Estimator"):
        Registry.from_classes({"demo.other": dict})  # type: ignore[dict-item]


def test_a_missing_extra_goes_through_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_extra(name: str) -> types.ModuleType:
        raise MissingExtraError(f"{name} needs NumPy, which is not installed")

    monkeypatch.setattr(importlib, "import_module", missing_extra)
    with pytest.raises(MissingExtraError, match="needs NumPy") as caught:
        EstimatorSpec("coclustering.single", "some.module:Model").load()
    assert "civiqhub-selene[blocks]" in caught.value.hint


def test_a_missing_dependency_of_the_target_is_not_an_unknown_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_dependency(name: str) -> types.ModuleType:
        raise ModuleNotFoundError("No module named 'heavy'", name="heavy")

    monkeypatch.setattr(importlib, "import_module", missing_dependency)
    with pytest.raises(ModuleNotFoundError, match="heavy") as caught:
        EstimatorSpec("demo.one", "some.module:Model").load()
    assert not isinstance(caught.value, UnknownEstimatorError)


@pytest.mark.parametrize(
    "key", ["Bad Key", "nodot", "a.b.c", "A.b", "a. b", ".b", "a.", "a-b.c", "1a.b", ""]
)
def test_a_malformed_key_is_refused(key: str) -> None:
    with pytest.raises(InvalidParameterError, match=r"family\.choice"):
        Registry.from_classes({key: One})
    with pytest.raises(InvalidParameterError, match=r"family\.choice"):
        EstimatorSpec(key, "module:Class")


@pytest.mark.parametrize("target", ["module", "module:", ":Class", "module:Class:x", "a b:C"])
def test_a_malformed_target_is_refused(target: str) -> None:
    with pytest.raises(InvalidParameterError, match="module:Class"):
        EstimatorSpec("demo.one", target)


def test_the_registry_is_immutable() -> None:
    registry = Registry.from_classes({"demo.one": One})
    with pytest.raises(TypeError, match="item assignment"):
        registry["demo.two"] = registry["demo.one"]  # type: ignore[index]
    with pytest.raises(TypeError, match="item deletion"):
        del registry["demo.one"]  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        registry.extra = 1  # type: ignore[attr-defined]


def test_union_gives_a_new_registry_where_the_right_side_wins() -> None:
    left = Registry.from_classes({"demo.one": One, "demo.two": Two})
    right = Registry.from_classes({"demo.one": Two, "other.one": One})
    merged = left | right
    assert list(merged) == ["demo.one", "demo.two", "other.one"]
    assert merged["demo.one"].load() is Two
    assert left["demo.one"].load() is One
    assert (right | left)["demo.one"].load() is One
    with pytest.raises(TypeError, match="unsupported operand"):
        left | {"demo.one": One}  # type: ignore[operator]


def test_two_packages_declaring_one_key_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    installed = [
        metadata.EntryPoint(name="demo.one", value="first.module:One", group=GROUP),
        metadata.EntryPoint(name="demo.one", value="first.module:One", group=GROUP),
        metadata.EntryPoint(name="demo.one", value="second.module:One", group=GROUP),
    ]
    monkeypatch.setattr(metadata, "entry_points", lambda *, group: installed)
    with pytest.raises(EstimatorDefinitionError, match="two installed packages") as caught:
        Registry.from_entry_points()
    assert "Uninstall" in caught.value.hint

    monkeypatch.setattr(metadata, "entry_points", lambda *, group: installed[:2])
    assert list(Registry.from_entry_points()) == ["demo.one"]


def test_default_registry_is_read_once(monkeypatch: pytest.MonkeyPatch) -> None:
    install_plugin(monkeypatch, {"demo.one": "selene_fake_plugin:One"})
    default_registry.cache_clear()
    try:
        first = default_registry()
        assert isinstance(first, Registry)
        assert list(first) == ["demo.one"]
        install_plugin(monkeypatch, {})
        assert default_registry() is first
    finally:
        default_registry.cache_clear()


def test_module_exports() -> None:
    assert set(registry_module.__all__) == {
        "GROUP",
        "EstimatorSpec",
        "Registry",
        "default_registry",
    }
