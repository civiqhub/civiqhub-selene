# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Base class of the estimators that run an iterative algorithm.

:class:`IterativeEstimator` knows how to run an iterative algorithm: several starts,
one seed per start, convergence, and the choice of the best start. A concrete
estimator writes three hooks and never writes a loop nor draws a seed::

    _start(data, rng)                           -> a starting state
    _step(data, state)                          -> the state after one full iteration
    _finish(data, state, history, computation)  -> the public value
"""

import math
from abc import abstractmethod
from typing import ClassVar, Generic, Protocol, TypeVar, final

import numpy as np

from civiqhub_selene.blocks._numerics import History, iterate
from civiqhub_selene.errors import EstimatorDefinitionError
from civiqhub_selene.estimator import DataT_contra, Estimator, ValueT_co

__all__ = ["HasObjective", "IterativeEstimator"]


class HasObjective(Protocol):
    """Any state that carries the quantity its algorithm maximises."""

    @property
    def objective(self) -> float:
        """The quantity the algorithm maximises, at this state."""
        ...


StateT = TypeVar("StateT", bound=HasObjective)


def _comparable(objective: float) -> float:
    """Read an objective for a comparison: a value that is not a number never wins."""
    value = float(objective)
    return -math.inf if math.isnan(value) else value


class IterativeEstimator(
    Estimator[DataT_contra, ValueT_co], Generic[DataT_contra, StateT, ValueT_co]
):
    """An estimator that maximises an objective from several random starts.

    ``_compute`` is written once, here: it derives one seed per start from the seed
    received, runs every start to convergence with ``iterate``, the only convergence
    loop of the package, keeps the start whose final objective is the largest (the
    first one on a tie) and hands it to :meth:`_finish`.

    A step must not lower the objective. A start stops as soon as its gain of
    objective goes under ``tolerance``, or after ``max_iterations`` steps.

    See the mathematical model: :doc:`/model/estimation`.

    Fields a concrete subclass declares:

    Attributes:
        n_starts: Number of random starts; the best one is kept.
        max_iterations: Largest number of iterations of a start.
        tolerance: Gain of objective under which a start stops.
    """

    stochastic: ClassVar[bool] = True
    dependencies: ClassVar[tuple[str, ...]] = ("numpy",)

    n_starts: int
    max_iterations: int
    tolerance: float

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "_compute" in cls.__dict__:
            raise EstimatorDefinitionError(
                f"{cls.__name__} redefines _compute(), which is written once in IterativeEstimator",
                hint="Write the algorithm in _start(), _step() and _finish().",
            )

    @final
    def _compute(self, data: DataT_contra, *, seed: int, computation: str) -> ValueT_co:
        best: tuple[StateT, History] | None = None
        for start_seed in np.random.SeedSequence(seed).spawn(self.n_starts):
            rng = np.random.default_rng(start_seed)
            state, history = iterate(
                lambda current: self._step(data, current),
                self._start(data, rng),
                gap=lambda old, new: float(new.objective) - float(old.objective),
                value=lambda current: float(current.objective),
                tolerance=self.tolerance,
                max_iterations=self.max_iterations,
            )
            if best is None or _comparable(state.objective) > _comparable(best[0].objective):
                best = (state, history)
        assert best is not None, "n_starts is at least 1"
        return self._finish(data, best[0], best[1], computation)

    @abstractmethod
    def _start(self, data: DataT_contra, rng: np.random.Generator) -> StateT:
        """Return a starting state. Called once per start, with the generator of the start."""

    @abstractmethod
    def _step(self, data: DataT_contra, state: StateT) -> StateT:
        """Return the state after one full iteration. It must not lower ``objective``."""

    @abstractmethod
    def _finish(
        self, data: DataT_contra, state: StateT, history: History, computation: str
    ) -> ValueT_co:
        """Return the public value drawn from the best final state.

        Args:
            data: The input.
            state: The final state of the start that was kept.
            history: The objective after each iteration of that start, and whether it
                converged.
            computation: Fingerprint the result will carry.
        """
