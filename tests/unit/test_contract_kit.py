# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import itertools
import re
import subprocess
import sys
from dataclasses import dataclass

import pytest

from civiqhub_selene import testing
from civiqhub_selene.errors import ContractError
from civiqhub_selene.estimator import Estimator
from civiqhub_selene.testing import EstimatorContract, fingerprint_of
from civiqhub_selene.types import Unavailable, UnavailableReason, canonical


@dataclass(frozen=True)
class Numbers:
    values: tuple[float, ...]

    def fingerprint(self) -> str:
        return fingerprint_of(["Numbers", self.values])


@dataclass(frozen=True, kw_only=True)
class Mean(Estimator[Numbers, float]):
    """A conforming example, written without touching a class of the package."""

    name = "mean"
    version = "1.0.0"
    reference = "Arithmetic mean."
    input_type = Numbers

    scale: float = 1.0

    def _why_unavailable(self, data: Numbers) -> Unavailable | None:
        if not data.values:
            return Unavailable(UnavailableReason.NO_VOTES, "0 values, at least 1 needed")
        return None

    def _compute(self, data: Numbers, *, seed: int, computation: str) -> float:
        return self.scale * sum(data.values) / len(data.values)


class TestExample(EstimatorContract):
    def make_estimator(self) -> Mean:
        return Mean(scale=2.0)

    def make_data(self) -> Numbers:
        return Numbers((1.0, 2.0, 3.0))

    def make_empty_data(self) -> Numbers:
        return Numbers(())


@dataclass(frozen=True, kw_only=True)
class Constant(Mean):
    """A conforming estimator that always answers."""

    name = "constant"

    def _why_unavailable(self, data: Numbers) -> Unavailable | None:
        return None

    def _compute(self, data: Numbers, *, seed: int, computation: str) -> float:
        return 1.0


class TestAlwaysAnswers(EstimatorContract):
    def make_estimator(self) -> Constant:
        return Constant()

    def make_data(self) -> Numbers:
        return Numbers((1.0,))

    def make_empty_data(self) -> None:
        return None


_calls = itertools.count()


@dataclass(frozen=True, kw_only=True)
class Drifting(Mean):
    """Faulty: the value changes from one call to the next."""

    name = "drifting"

    def _compute(self, data: Numbers, *, seed: int, computation: str) -> float:
        return float(next(_calls))


@dataclass(frozen=True, kw_only=True)
class Silent(Mean):
    """Faulty: returns None instead of an explained absence."""

    name = "silent"

    def _why_unavailable(self, data: Numbers) -> Unavailable | None:
        return None

    def _compute(self, data: Numbers, *, seed: int, computation: str) -> float:
        if not data.values:
            return None  # type: ignore[return-value]
        return super()._compute(data, seed=seed, computation=computation)


@dataclass(frozen=True, kw_only=True)
class Unreferenced(Mean):
    """Faulty: no reference."""

    name = "unreferenced"
    reference = ""


@dataclass(frozen=True, kw_only=True)
class Permissive(Estimator[Numbers, float]):
    """Faulty: reads anything."""

    name = "permissive"
    version = "1.0.0"
    reference = "r"
    input_type = object

    def _compute(self, data: Numbers, *, seed: int, computation: str) -> float:
        return 0.0


def contract_for(estimator: Estimator[Numbers, float]) -> EstimatorContract:
    class Contract(EstimatorContract):
        def make_estimator(self) -> Estimator[Numbers, float]:
            return estimator

        def make_data(self) -> Numbers:
            return Numbers((1.0, 2.0, 3.0))

        def make_empty_data(self) -> Numbers:
            return Numbers(())

    return Contract()


def test_a_drifting_estimator_breaks_determinism() -> None:
    with pytest.raises(ContractError, match="Drifting breaks the rule 'deterministic'"):
        contract_for(Drifting()).test_deterministic()


def test_none_instead_of_an_absence_breaks_explicit_absence() -> None:
    with pytest.raises(ContractError, match="Silent breaks the rule 'absence is explicit'"):
        contract_for(Silent()).test_absence_is_explicit()


def test_an_empty_reference_breaks_the_declared_identity() -> None:
    with pytest.raises(ContractError, match="'identity is declared': reference is empty"):
        contract_for(Unreferenced()).test_identity_is_declared()


def test_accepting_any_input_breaks_the_rejection_of_foreign_input() -> None:
    with pytest.raises(ContractError, match="Permissive breaks the rule 'rejects foreign"):
        contract_for(Permissive()).test_rejects_foreign_input()


def test_a_contract_error_is_an_assertion_error_with_a_hint() -> None:
    with pytest.raises(AssertionError, match="Unreferenced") as caught:
        contract_for(Unreferenced()).test_identity_is_declared()
    assert isinstance(caught.value, ContractError)
    assert "EstimatorContract" in caught.value.hint


def test_an_estimator_that_is_not_a_frozen_dataclass_breaks_the_contract() -> None:
    @dataclass(kw_only=True)
    class Thawed(Estimator[Numbers, float]):
        name = "thawed"
        version = "1.0.0"
        reference = "r"
        input_type = Numbers

        def _compute(self, data: Numbers, *, seed: int, computation: str) -> float:
            return 0.0

    class Plain(Estimator[Numbers, float]):
        name = "plain"
        version = "1.0.0"
        reference = "r"
        input_type = Numbers

        def _compute(self, data: Numbers, *, seed: int, computation: str) -> float:
            return 0.0

    with pytest.raises(ContractError, match="not frozen"):
        contract_for(Thawed()).test_estimator_is_a_frozen_value()
    with pytest.raises(ContractError, match="not a dataclass"):
        contract_for(Plain()).test_estimator_is_a_frozen_value()


def test_a_value_that_is_none_or_nan_is_not_a_result() -> None:
    @dataclass(frozen=True, kw_only=True)
    class Hollow(Mean):
        name = "hollow"

        def _compute(self, data: Numbers, *, seed: int, computation: str) -> float:
            return None  # type: ignore[return-value]

    @dataclass(frozen=True, kw_only=True)
    class Undefined(Mean):
        name = "undefined"

        def _compute(self, data: Numbers, *, seed: int, computation: str) -> float:
            return float("nan")

    with pytest.raises(ContractError, match="the value is None"):
        contract_for(Hollow()).test_returns_a_result()
    with pytest.raises(ContractError, match="the value is NaN"):
        contract_for(Undefined()).test_returns_a_result()


def test_a_contract_without_its_three_methods_says_what_to_write() -> None:
    with pytest.raises(NotImplementedError, match="make_estimator"):
        EstimatorContract().test_identity_is_declared()


def test_fingerprint_of_uses_the_fingerprint_of_the_value_when_it_has_one() -> None:
    numbers = Numbers((1.0, 2.0))
    assert fingerprint_of(numbers) == numbers.fingerprint()
    assert re.fullmatch(r"[0-9a-f]{16}", fingerprint_of(0.25))
    assert fingerprint_of((1, 2)) == fingerprint_of([1, 2])
    assert fingerprint_of(0.25) != fingerprint_of(0.5)
    assert canonical(0.25) == "0.25"


def test_importing_the_kit_does_not_load_pytest() -> None:
    script = (
        "import sys\n"
        "import civiqhub_selene\n"
        "import civiqhub_selene.testing\n"
        "assert 'pytest' not in sys.modules, 'pytest was loaded'\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_the_module_docstring_shows_the_usage_in_under_fifteen_lines() -> None:
    docstring = testing.__doc__ or ""
    example = docstring.split("::", 1)[1].split("A test of the kit", 1)[0]
    lines = [line for line in example.splitlines() if line.strip()]
    assert "class TestMean(EstimatorContract):" in example
    assert 0 < len(lines) < 15
