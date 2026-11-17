# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import platform
from dataclasses import dataclass, field
from importlib import metadata
from typing import ClassVar

import pytest

import civiqhub_selene
from civiqhub_selene.errors import (
    EstimatorDefinitionError,
    InvalidInputError,
    InvalidParameterError,
)
from civiqhub_selene.estimator import Estimator, Fingerprinted
from civiqhub_selene.types import (
    Interval,
    Method,
    Result,
    StandardError,
    Unavailable,
    UnavailableReason,
    Uncertainty,
    canonical,
)


@dataclass(frozen=True)
class Numbers:
    values: tuple[float, ...]
    lineage: tuple[str, ...] = ()

    def fingerprint(self) -> str:
        return canonical(["Numbers", self.values])


@dataclass(frozen=True)
class Plain:
    """An input without lineage."""

    def fingerprint(self) -> str:
        return "plain"


@dataclass(frozen=True, kw_only=True)
class Mean(Estimator[Numbers, float]):
    name = "mean"
    version = "1.0.0"
    reference = "Arithmetic mean."
    input_type = Numbers

    scale: float = 1.0
    calls: list[str] = field(default_factory=list, init=False, compare=False, repr=False)

    def _why_unavailable(self, data: Numbers) -> Unavailable | None:
        self.calls.append("why")
        if not data.values:
            return Unavailable(UnavailableReason.NO_VOTES, "0 values, at least 1 needed")
        return None

    def _compute(self, data: Numbers, *, seed: int, computation: str) -> float | Unavailable:
        self.calls.append(f"compute:{seed}:{computation}")
        if any(value < 0 for value in data.values):
            return Unavailable(UnavailableReason.UNSUPPORTED_INPUT, "a value is negative")
        return self.scale * sum(data.values) / len(data.values)

    def _uncertainty(self, data: Numbers, value: float) -> Uncertainty | None:
        self.calls.append("uncertainty")
        return StandardError(0.5) if len(data.values) > 1 else None


@dataclass(frozen=True, kw_only=True)
class Noisy(Estimator[Numbers, float]):
    name = "noisy"
    version = "0.1.0"
    reference = "A method with randomness."
    input_type = Numbers
    stochastic = True
    dependencies: ClassVar[tuple[str, ...]] = ("pytest", "a-package-that-is-not-installed")

    def _compute(self, data: Numbers, *, seed: int, computation: str) -> float:
        return float(seed)


NUMBERS = Numbers((1.0, 2.0, 3.0))


def test_an_input_only_needs_a_fingerprint() -> None:
    fingerprinted: Fingerprinted = NUMBERS
    assert fingerprinted.fingerprint() == NUMBERS.fingerprint()


def test_estimate_returns_a_result_with_its_provenance() -> None:
    result = Mean(scale=2.0).estimate(NUMBERS)
    assert isinstance(result, Result)
    assert result.value == 4.0
    provenance = result.provenance
    assert provenance.method == Method("mean", "1.0.0", "Arithmetic mean.")
    assert provenance.parameters == (("scale", 2.0),)
    assert provenance.input_fingerprint == NUMBERS.fingerprint()
    assert provenance.engine_version == civiqhub_selene.__version__
    assert provenance.dependencies == (("python", platform.python_version()),)


def test_estimate_passes_the_fingerprint_of_the_result_to_compute() -> None:
    estimator = Mean()
    result = estimator.estimate(NUMBERS, seed=3)
    assert estimator.calls == ["why", f"compute:3:{result.fingerprint()}", "uncertainty"]


def test_estimate_attaches_the_uncertainty_the_method_defines() -> None:
    assert Mean().estimate(NUMBERS).uncertainty == StandardError(0.5)
    assert Mean().estimate(Numbers((1.0,))).uncertainty is None
    assert Noisy().estimate(NUMBERS).uncertainty is None


@dataclass(frozen=True, kw_only=True)
class Nameless(Estimator[Numbers, float]):
    version = "1.0.0"
    reference = "r"
    input_type = Numbers

    def _compute(self, data: Numbers, *, seed: int, computation: str) -> float:
        return 0.0


@dataclass(frozen=True, kw_only=True)
class BadName(Nameless):
    name = "Bad-Name"


@dataclass(frozen=True, kw_only=True)
class ShortVersion(Nameless):
    name = "short_version"
    version = "1.0"


@dataclass(frozen=True, kw_only=True)
class NoInputType(Estimator[Numbers, float]):
    name = "no_input_type"
    version = "1.0.0"
    reference = "r"

    def _compute(self, data: Numbers, *, seed: int, computation: str) -> float:
        return 0.0


@dataclass(frozen=True, kw_only=True)
class NoReference(Estimator[Numbers, float]):
    name = "no_reference"
    version = "1.0.0"
    input_type = Numbers

    def _compute(self, data: Numbers, *, seed: int, computation: str) -> float:
        return 0.0


@pytest.mark.parametrize(
    ("estimator_class", "named"),
    [
        (Nameless, "Nameless.name"),
        (BadName, "'Bad-Name'"),
        (ShortVersion, "'1.0'"),
        (NoInputType, "NoInputType.input_type"),
    ],
)
def test_an_undeclared_identity_stops_estimate_before_the_input_is_checked(
    estimator_class: type[Estimator[Numbers, float]], named: str
) -> None:
    with pytest.raises(EstimatorDefinitionError, match=named) as caught:
        estimator_class().estimate(object())  # type: ignore[arg-type]
    assert isinstance(caught.value, TypeError)
    with pytest.raises(EstimatorDefinitionError, match=named):
        estimator_class.method()


def test_a_missing_reference_is_an_empty_reference() -> None:
    assert NoReference.method() == Method("no_reference", "1.0.0", "")


def test_an_input_of_another_type_is_refused() -> None:
    with pytest.raises(InvalidInputError, match="mean reads a Numbers, got Plain"):
        Mean().estimate(Plain())  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError, match="got object"):
        Mean().estimate(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [-1, 1.0, "0", True, None])
def test_a_seed_is_a_non_negative_integer(seed: object) -> None:
    with pytest.raises(InvalidParameterError, match="seed must be a non-negative integer"):
        Mean().estimate(NUMBERS, seed=seed)  # type: ignore[arg-type]


def test_an_absence_skips_the_computation_and_keeps_the_provenance() -> None:
    estimator = Mean()
    empty = Numbers(())
    result = estimator.estimate(empty)
    assert estimator.calls == ["why"]
    assert result.is_available is False
    assert result.unavailable == Unavailable(
        UnavailableReason.NO_VOTES, "0 values, at least 1 needed"
    )
    assert result.provenance.input_fingerprint == empty.fingerprint()
    assert result.provenance.method.name == "mean"


def test_an_absence_found_during_the_computation_has_no_uncertainty() -> None:
    estimator = Mean()
    result = estimator.estimate(Numbers((1.0, -1.0)))
    assert result.is_available is False
    assert result.unavailable is not None
    assert result.unavailable.reason is UnavailableReason.UNSUPPORTED_INPUT
    assert result.uncertainty is None
    assert "uncertainty" not in estimator.calls


def test_a_method_without_randomness_leaves_the_seed_out() -> None:
    first = Mean().estimate(NUMBERS, seed=0)
    second = Mean().estimate(NUMBERS, seed=1)
    assert first.provenance.randomness is None
    assert first.provenance.seed is None
    assert first.fingerprint() == second.fingerprint()
    assert first == second


def test_a_stochastic_method_records_the_seed_in_the_fingerprint() -> None:
    first = Noisy().estimate(NUMBERS, seed=0)
    second = Noisy().estimate(NUMBERS, seed=1)
    assert first.provenance.randomness is not None
    assert first.provenance.randomness.generator == "PCG64"
    assert first.provenance.seed == 0
    assert second.provenance.seed == 1
    assert first.fingerprint() != second.fingerprint()
    assert Noisy().estimate(NUMBERS, seed=1) == second


def test_dependencies_are_recorded_with_their_installed_version() -> None:
    provenance = Noisy().estimate(NUMBERS).provenance
    assert provenance.dependencies == (
        ("python", platform.python_version()),
        ("pytest", metadata.version("pytest")),
        ("a-package-that-is-not-installed", "unknown"),
    )


def test_lineage_is_taken_from_the_input_when_it_has_one() -> None:
    with_lineage = Numbers((1.0,), lineage=("aaaa", "bbbb"))
    assert Mean().estimate(with_lineage).provenance.lineage == ("aaaa", "bbbb")
    assert Mean().estimate(NUMBERS).provenance.lineage == ()


def test_settings_enter_the_fingerprint() -> None:
    assert Mean(scale=1.0).estimate(NUMBERS) != Mean(scale=2.0).estimate(NUMBERS)
    assert Mean(scale=2.0).estimate(NUMBERS) == Mean(scale=2.0).estimate(NUMBERS)


def test_estimate_cannot_be_redefined() -> None:
    # The ignore below is needed: mypy refuses the override because estimate is final.
    # With warn_unused_ignores, this file would fail to type-check if it did not.
    with pytest.raises(EstimatorDefinitionError, match=r"Rewritten redefines estimate\(\)"):

        class Rewritten(Mean):
            def estimate(  # type: ignore[misc]
                self, data: Numbers, *, seed: int = 0
            ) -> Result[float]:
                raise AssertionError("never reached")


def test_an_estimator_without_compute_is_abstract() -> None:
    class Incomplete(Estimator[Numbers, float]):
        name = "incomplete"
        version = "1.0.0"
        reference = "r"
        input_type = Numbers

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()  # type: ignore[abstract]


def test_uncertainty_can_be_an_interval() -> None:
    @dataclass(frozen=True, kw_only=True)
    class Bounded(Mean):
        name = "bounded"

        def _uncertainty(self, data: Numbers, value: float) -> Uncertainty | None:
            return Interval(value - 1, value + 1, 0.9)

    result = Bounded().estimate(NUMBERS)
    assert result.uncertainty == Interval(1.0, 3.0, 0.9)
    assert result.method.name == "bounded"
