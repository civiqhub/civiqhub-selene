# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

import pytest

from civiqhub_selene.types import (
    ADDED_SETTING,
    Displayable,
    Method,
    Provenance,
    Randomness,
    canonical,
    settings_of,
)

METHOD = Method("ordinal_dissensus", "0.1.0", "Research note, 2026")


def make_provenance(**changes: object) -> Provenance:
    base = Provenance(
        method=METHOD,
        engine_version="0.0.1",
        parameters=(("moderate_position", 0.5), ("tolerance", 1e-6)),
        input_fingerprint="0123456789abcdef",
        dependencies=(("python", "3.13.0"),),
        randomness=Randomness("PCG64", "7"),
        lineage=("aaaaaaaaaaaaaaaa",),
    )
    return dataclasses.replace(base, **changes)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("instance", "field_name"),
    [
        (METHOD, "name"),
        (Randomness("PCG64", "0"), "seed"),
        (make_provenance(), "engine_version"),
    ],
)
def test_types_are_frozen(instance: object, field_name: str) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field_name, "changed")


def test_method_prints_as_name_and_version() -> None:
    assert str(METHOD) == "ordinal_dissensus 0.1.0"
    assert Method("m", "1.0.0").reference == ""


def test_seed_is_an_integer_of_any_size_or_none() -> None:
    big = 2**200 + 1
    provenance = make_provenance(randomness=Randomness("PCG64", str(big)))
    assert provenance.seed == big
    assert json.loads(json.dumps(provenance.to_dict()))["randomness"]["seed"] == str(big)
    assert make_provenance(randomness=None).seed is None


def test_fingerprint_ignores_engine_version_and_dependencies() -> None:
    reference = make_provenance().fingerprint()
    assert make_provenance(engine_version="9.9.9").fingerprint() == reference
    assert (
        make_provenance(dependencies=(("python", "3.14.0"), ("numpy", "2.4.0"))).fingerprint()
        == reference
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"parameters": (("moderate_position", 0.6), ("tolerance", 1e-6))},
        {"randomness": Randomness("PCG64", "8")},
        {"randomness": None},
        {"input_fingerprint": "fedcba9876543210"},
        {"lineage": ()},
        {"method": Method("ordinal_dissensus", "0.2.0")},
        {"method": Method("collective_regime", "0.1.0")},
    ],
)
def test_fingerprint_changes_with_what_defines_the_computation(
    changes: dict[str, object],
) -> None:
    assert make_provenance(**changes).fingerprint() != make_provenance().fingerprint()


def test_fingerprint_is_sixteen_hexadecimal_characters() -> None:
    assert re.fullmatch(r"[0-9a-f]{16}", make_provenance().fingerprint())


def test_fingerprint_is_stable_across_processes() -> None:
    script = (
        "from civiqhub_selene.types import Method, Provenance, Randomness\n"
        "p = Provenance(Method('m', '1.0.0'), '0.0.1', (('a', 1), ('b', ('x', 2.5))),"
        " 'abc', (), Randomness('PCG64', '3'), ('l',))\n"
        "print(p.fingerprint())\n"
    )
    outputs = set()
    for hash_seed in ("0", "1", "4242"):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        outputs.add(completed.stdout.strip())
    assert len(outputs) == 1
    expected = Provenance(
        Method("m", "1.0.0"),
        "0.0.1",
        (("a", 1), ("b", ("x", 2.5))),
        "abc",
        (),
        Randomness("PCG64", "3"),
        ("l",),
    ).fingerprint()
    assert outputs == {expected}


def test_to_dict_is_json_ready_and_carries_the_fingerprint() -> None:
    provenance = make_provenance()
    data = provenance.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert data["fingerprint"] == provenance.fingerprint()
    assert data["method"] == {
        "name": "ordinal_dissensus",
        "version": "0.1.0",
        "reference": "Research note, 2026",
    }
    assert data["parameters"] == {"moderate_position": 0.5, "tolerance": 1e-6}
    assert data["dependencies"] == {"python": "3.13.0"}
    assert data["randomness"] == {"generator": "PCG64", "seed": "7"}
    assert data["lineage"] == ["aaaaaaaaaaaaaaaa"]
    assert make_provenance(randomness=None).to_dict()["randomness"] is None


@dataclass(frozen=True)
class Inner:
    spread: float = 0.3
    levels: tuple[int, ...] = (1, 2)


@dataclass(frozen=True)
class Outer:
    zeta: int = 2
    alpha: str = "a"
    inner: Inner = Inner()
    derived: int = field(default=0, init=False)


def test_settings_are_sorted_and_nest_dataclasses_with_their_type_first() -> None:
    assert settings_of(Outer()) == (
        ("alpha", "a"),
        ("inner", (("type", "Inner"), ("levels", (1, 2)), ("spread", 0.3))),
        ("zeta", 2),
    )


def test_settings_of_a_non_dataclass_are_empty() -> None:
    assert settings_of(object()) == ()
    assert settings_of(3) == ()
    assert settings_of(Outer) == ()


def test_nested_settings_become_nested_dictionaries() -> None:
    provenance = make_provenance(parameters=settings_of(Outer()))
    assert provenance.to_dict()["parameters"] == {
        "alpha": "a",
        "inner": {"type": "Inner", "levels": [1, 2], "spread": 0.3},
        "zeta": 2,
    }


def test_canonical_sorts_keys_refuses_nan_and_writes_unknown_objects_as_repr() -> None:
    assert canonical({"b": 1, "a": (1, 2)}) == '{"a": [1, 2], "b": 1}'
    assert canonical(Inner()) == json.dumps(repr(Inner()))
    with pytest.raises(ValueError, match="JSON"):
        canonical(float("nan"))


@dataclass(frozen=True, repr=False)
class Shown(Displayable):
    label: str

    def _describe(self) -> list[str]:
        return [f"Shown({self.label})", "first <detail>", "second detail"]


def test_displayable_derives_its_displays_from_one_description() -> None:
    shown = Shown("x")
    assert repr(shown) == "Shown(x)"
    assert "\n" not in repr(shown)
    assert str(shown) == "Shown(x)\nfirst <detail>\nsecond detail"
    assert shown.summary() == str(shown)
    assert shown._repr_html_() == "<pre>Shown(x)\nfirst &lt;detail&gt;\nsecond detail</pre>"


def test_displayable_is_stateless_and_asks_for_a_description() -> None:
    assert Displayable.__slots__ == ()
    with pytest.raises(NotImplementedError, match="_describe"):
        repr(Displayable())


@dataclass(frozen=True)
class Grown:
    alpha: str = "a"
    later: bool = field(default=False, metadata={ADDED_SETTING: True})
    count: int = field(default=0, metadata={ADDED_SETTING: True})


def test_an_added_setting_is_left_out_at_its_default_only() -> None:
    assert settings_of(Grown()) == (("alpha", "a"),)
    assert settings_of(Grown(later=True)) == (("alpha", "a"), ("later", True))
    assert settings_of(Grown(count=3)) == (("alpha", "a"), ("count", 3))
    # Equal but of another type is not the default: False == 0, yet it is written.
    assert settings_of(Grown(count=False)) == (("alpha", "a"), ("count", False))
