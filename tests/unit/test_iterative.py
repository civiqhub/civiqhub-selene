# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass
from importlib import metadata

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene.blocks._numerics import History
from civiqhub_selene.blocks.iterative import HasObjective, IterativeEstimator
from civiqhub_selene.errors import EstimatorDefinitionError
from civiqhub_selene.testing import EstimatorContract

pytestmark = pytest.mark.blocks


@dataclass(frozen=True)
class Target:
    value: float

    def fingerprint(self) -> str:
        return f"target:{self.value!r}"


@dataclass(frozen=True)
class Position:
    """The state of the test model: a number and its objective."""

    number: float
    objective: float


@dataclass(frozen=True)
class Outcome:
    number: float
    objective: float
    history: History
    computation: str

    def fingerprint(self) -> str:
        return f"{self.number!r}:{self.history.objectives!r}"


STARTS: list[float] = []


@dataclass(frozen=True, kw_only=True)
class Approach(IterativeEstimator[Target, Position, Outcome]):
    """Each step moves the number half of the way to the target."""

    name = "approach"
    version = "0.1.0"
    reference = "A test model."
    input_type = Target

    n_starts: int = 4
    max_iterations: int = 100
    tolerance: float = 1e-9

    def _start(self, data: Target, rng: np.random.Generator) -> Position:
        number = float(rng.uniform(-100.0, 100.0))
        STARTS.append(number)
        return Position(number, -abs(number - data.value))

    def _step(self, data: Target, state: Position) -> Position:
        number = (state.number + data.value) / 2
        return Position(number, -abs(number - data.value))

    def _finish(self, data: Target, state: Position, history: History, computation: str) -> Outcome:
        return Outcome(state.number, state.objective, history, computation)


def run(estimator: Approach, seed: int = 0) -> tuple[Outcome, list[float]]:
    STARTS.clear()
    outcome = estimator.estimate(Target(10.0), seed=seed).value
    return outcome, list(STARTS)


def test_a_state_only_needs_an_objective() -> None:
    state: HasObjective = Position(1.0, -9.0)
    assert state.objective == -9.0


def test_each_start_has_its_own_generator_and_the_seed_fixes_them_all() -> None:
    _, starts = run(Approach(n_starts=5), seed=7)
    assert len(starts) == 5
    assert len(set(starts)) == 5
    _, again = run(Approach(n_starts=5), seed=7)
    assert again == starts
    _, other = run(Approach(n_starts=5), seed=8)
    assert other != starts
    # The first starts do not depend on how many starts follow.
    _, fewer = run(Approach(n_starts=2), seed=7)
    assert fewer == starts[:2]


def test_the_start_with_the_largest_final_objective_is_kept() -> None:
    estimator = Approach(n_starts=6, max_iterations=3)
    outcome, starts = run(estimator)
    finals = [10.0 + (start - 10.0) / 8 for start in starts]
    best = min(finals, key=lambda number: abs(number - 10.0))
    assert outcome.number == pytest.approx(best)
    assert outcome.objective == pytest.approx(-abs(best - 10.0))
    assert outcome.objective == max(-abs(number - 10.0) for number in finals)


def test_on_a_tie_the_first_start_is_kept() -> None:
    @dataclass(frozen=True, kw_only=True)
    class Flat(Approach):
        name = "flat"

        def _step(self, data: Target, state: Position) -> Position:
            return Position(state.number, 0.0)

    outcome, starts = run(Flat(n_starts=4))
    assert outcome.number == starts[0]


def test_a_start_whose_objective_is_not_a_number_never_wins() -> None:
    @dataclass(frozen=True, kw_only=True)
    class Broken(Approach):
        name = "broken"

        def _step(self, data: Target, state: Position) -> Position:
            lost = state.number < 0
            return Position(state.number, math.nan if lost else -abs(state.number - data.value))

    outcome, starts = run(Broken(n_starts=6, max_iterations=2))
    assert any(start < 0 for start in starts)
    assert not math.isnan(outcome.objective)
    assert outcome.number >= 0


def test_a_start_stops_when_its_gain_goes_under_the_tolerance() -> None:
    outcome, _ = run(Approach(n_starts=1, tolerance=1e-3, max_iterations=100))
    history = outcome.history
    assert history.converged is True
    assert history.n_iterations < 100
    gains = np.diff(history.objectives)
    assert gains[-1] < 1e-3
    assert (gains[:-1] >= 1e-3).all()
    assert history.objectives[-1] == outcome.objective


def test_a_start_stops_at_the_largest_number_of_iterations() -> None:
    outcome, _ = run(Approach(n_starts=1, tolerance=1e-12, max_iterations=4))
    assert outcome.history.converged is False
    assert outcome.history.n_iterations == 4


def test_an_infinite_starting_objective_does_not_stop_the_loop() -> None:
    @dataclass(frozen=True, kw_only=True)
    class Cold(Approach):
        name = "cold"

        def _start(self, data: Target, rng: np.random.Generator) -> Position:
            return Position(float(rng.uniform(-100.0, 100.0)), -math.inf)

    outcome = Cold(n_starts=2, tolerance=1e-3).estimate(Target(10.0)).value
    assert outcome.history.converged is True
    assert outcome.history.n_iterations > 1
    assert math.isfinite(outcome.objective)


def test_provenance_records_the_seed_and_numpy() -> None:
    result = Approach().estimate(Target(10.0), seed=5)
    provenance = result.provenance
    assert provenance.randomness is not None
    assert provenance.randomness.generator == "PCG64"
    assert provenance.seed == 5
    assert dict(provenance.dependencies)["numpy"] == metadata.version("numpy")
    assert provenance.parameters == (
        ("max_iterations", 100),
        ("n_starts", 4),
        ("tolerance", 1e-9),
    )
    assert result.value.computation == result.fingerprint()
    assert Approach().estimate(Target(10.0), seed=6).fingerprint() != result.fingerprint()


def test_compute_cannot_be_redefined() -> None:
    with pytest.raises(EstimatorDefinitionError, match=r"Rewritten redefines _compute\(\)"):

        @dataclass(frozen=True, kw_only=True)
        class Rewritten(Approach):
            def _compute(  # type: ignore[misc]
                self, data: Target, *, seed: int, computation: str
            ) -> Outcome:
                raise AssertionError("never reached")


def test_the_three_hooks_are_required() -> None:
    @dataclass(frozen=True, kw_only=True)
    class Hookless(IterativeEstimator[Target, Position, Outcome]):
        name = "hookless"
        version = "0.1.0"
        reference = "r"
        input_type = Target
        n_starts: int = 1
        max_iterations: int = 1
        tolerance: float = 0.0

    with pytest.raises(TypeError, match="abstract"):
        Hookless()  # type: ignore[abstract]


class TestApproach(EstimatorContract):
    def make_estimator(self) -> Approach:
        return Approach(n_starts=2)

    def make_data(self) -> Target:
        return Target(10.0)

    def make_empty_data(self) -> None:
        return None
