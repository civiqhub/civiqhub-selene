# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""The choice of the shape of a block model: how many groups and how many blocks.

A block model is estimated at a fixed shape: ``n_groups`` groups of voters and
``n_blocks`` blocks of items. :class:`ShapeSelection` estimates it at every shape of a
grid, scores each shape with a criterion, and returns the co-clustering at the best
shape, with the trace of the choice in its ``selection``::

    result = select_shape(votes, "single", n_groups=(1, 2, 3, 4), n_blocks=(1, 2, 3))
    result.value.selection.best      # (3, 2)
    print(result.value.selection)    # the grid of scores

Two criteria are known, by key:

``"integrated_likelihood"``
    The integrated completed likelihood (ICL) of the value (Biernacki, Celeux and
    Govaert, 2000): the classified log-likelihood minus a penalty for the parameters.
    Read on values that carry it (:class:`~civiqhub_selene.blocks.results.HasIntegratedLikelihood`),
    such as the single-membership model. One estimation per shape.
``"held_out_likelihood"``
    The mean log-probability of held-out cells, by cell-wise cross-validation: the shown
    cells are split at random into ``n_folds`` folds; for each fold the model is
    estimated with the fold held out (:meth:`~civiqhub_selene.blocks.data.VoteMatrix.hold_out`)
    and scores the states of the held-out cells
    (:class:`~civiqhub_selene.blocks.results.PredictsVotes`). The score is the mean over
    the folds, with its standard error. It suits every model, the mixed-membership one
    included, whose memberships are incidental parameters for which asymptotic penalties
    give no guarantee. ``n_folds`` estimations per shape.

Each model names its default criterion (``default_criterion``): the integrated likelihood
for the single-membership model, the held-out likelihood for the mixed-membership one.
"""

import dataclasses
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np

from civiqhub_selene.blocks._arrays import FloatArray
from civiqhub_selene.blocks.base import CoclusteringEstimator, availability, estimator_for
from civiqhub_selene.blocks.data import VoteMatrix
from civiqhub_selene.blocks.results import (
    Coclustering,
    HasIntegratedLikelihood,
    PredictsVotes,
    Selection,
)
from civiqhub_selene.blocks.single import LatentBlockModel
from civiqhub_selene.errors import InvalidParameterError
from civiqhub_selene.estimator import Estimator
from civiqhub_selene.registry import Registry
from civiqhub_selene.types import Result, Unavailable, UnavailableReason

__all__ = ["CRITERIA", "ShapeSelection", "select_shape"]

CRITERIA: dict[str, str] = {
    "integrated_likelihood": (
        "the integrated completed likelihood of the value; one estimation per shape"
    ),
    "held_out_likelihood": (
        "the mean log-probability of held-out cells, by cell-wise cross-validation; "
        "n_folds estimations per shape"
    ),
}
"""The criteria :class:`ShapeSelection` knows, by key, with their meaning."""

_SEARCHES = ("grid", "greedy")


def _shapes(received: object, name: str) -> tuple[int, ...]:
    """Check a collection of sizes and return it sorted, without repeat."""
    if isinstance(received, (str, bytes)) or not isinstance(received, Iterable):
        raise InvalidParameterError(f"{name} is a collection of integers, got {received!r}")
    sizes = tuple(received)
    if not sizes or not all(
        isinstance(size, int) and not isinstance(size, bool) and size >= 1 for size in sizes
    ):
        raise InvalidParameterError(
            f"{name} holds at least one integer, each at least 1, got {received!r}"
        )
    return tuple(sorted(set(sizes)))


@dataclass(frozen=True, kw_only=True)
class ShapeSelection(Estimator[VoteMatrix, Coclustering]):
    """Choose the number of groups and of blocks of a block model, and estimate it there.

    Definition. For every shape ``(Q, L)`` of the grid ``n_groups x n_blocks`` on which
    a co-clustering exists (see :func:`~civiqhub_selene.blocks.base.availability`), the
    model is estimated at that shape and scored by the criterion:

    - ``"integrated_likelihood"``: ``score(Q, L)`` is the integrated likelihood of the
      value estimated on all the votes;
    - ``"held_out_likelihood"``: the shown cells are split at random into F folds; for the
      fold f, the model is estimated on the votes where the cells of f are held out, and
      ``s_f(Q, L)`` is the mean log-probability of the states of the cells of f.
      ``score(Q, L)`` is the mean of the ``s_f`` and its standard error is their standard
      deviation divided by ``sqrt(F)``.

    The best shape has the largest score (the first in the order of the grid on a tie).
    The alternatives are the other shapes whose score lies within one standard error of
    the best one; the integrated likelihood gives no standard error, hence no
    alternative. A shape on which no co-clustering exists scores ``-inf``.

    With ``search="grid"`` every shape is scored. With ``search="greedy"`` the search
    starts from the smallest shape and, at each step, scores the shapes with one more
    group and with one more block, moves to the better one if it improves the current
    score, and stops otherwise; the shapes it does not visit score NaN. It visits about
    ``len(n_groups) + len(n_blocks)`` shapes instead of their product.

    The value is the co-clustering of the model at the best shape, estimated on all the
    votes with the seed received, with its ``selection`` set and its ``computation`` set
    to this computation. Every estimation uses the same seed; the folds are drawn with
    ``numpy.random.default_rng(seed)``.

    References: Biernacki, Celeux and Govaert (2000), Assessing a mixture model for
    clustering with the integrated completed likelihood, IEEE TPAMI 22(7); cross-validation
    by held-out entries as in Li, Levina and Zhu (2020), Network cross-validation by edge
    sampling, Biometrika 107(2).

    Attributes:
        model: The block model whose shape is chosen; its own ``n_groups`` and
            ``n_blocks`` are ignored. Defaults to ``LatentBlockModel()``.
        n_groups: The numbers of groups compared, integers of at least 1. Defaults to
            1 to 4.
        n_blocks: The numbers of blocks compared, likewise. Defaults to 1 to 3.
        criterion: A key of :data:`CRITERIA`, or ``None`` for the default criterion of
            the model. Defaults to ``None``.
        n_folds: The number of folds of the held-out likelihood, at least 2. Defaults
            to 5. Unused by the integrated likelihood.
        search: ``"grid"`` (every shape) or ``"greedy"``. Defaults to ``"grid"``.

    Raises:
        InvalidParameterError: If a setting is out of range, or if the criterion does not
            suit the values of the model.
    """

    name: ClassVar[str] = "shape_selection"
    version: ClassVar[str] = "0.1.0"
    reference: ClassVar[str] = "Biernacki, Celeux, Govaert 2000; Li, Levina, Zhu 2020"
    input_type: ClassVar[type] = VoteMatrix
    stochastic: ClassVar[bool] = True
    dependencies: ClassVar[tuple[str, ...]] = ("numpy",)

    model: CoclusteringEstimator[Any] = field(default_factory=LatentBlockModel)
    n_groups: tuple[int, ...] = (1, 2, 3, 4)
    n_blocks: tuple[int, ...] = (1, 2, 3)
    criterion: str | None = None
    n_folds: int = 5
    search: Literal["grid", "greedy"] = "grid"

    def __post_init__(self) -> None:
        if not isinstance(self.model, CoclusteringEstimator):
            raise InvalidParameterError(
                f"model is a block model such as LatentBlockModel(), got {self.model!r}"
            )
        object.__setattr__(self, "n_groups", _shapes(self.n_groups, "n_groups"))
        object.__setattr__(self, "n_blocks", _shapes(self.n_blocks, "n_blocks"))
        if self.criterion is not None and self.criterion not in CRITERIA:
            raise InvalidParameterError(
                f"criterion is one of {sorted(CRITERIA)} or None, got {self.criterion!r}"
            )
        folds = self.n_folds
        if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
            raise InvalidParameterError(f"n_folds is an integer of at least 2, got {folds!r}")
        if self.search not in _SEARCHES:
            raise InvalidParameterError(f"search is one of {_SEARCHES}, got {self.search!r}")
        # Every shape of the grid must be a valid setting of the model.
        for n_groups in self.n_groups:
            for n_blocks in self.n_blocks:
                self.model.with_shape(n_groups, n_blocks)

    @property
    def effective_criterion(self) -> str:
        """The criterion used: the one given, or the default criterion of the model."""
        if self.criterion is not None:
            return self.criterion
        default = type(self.model).default_criterion
        if default not in CRITERIA:
            raise InvalidParameterError(
                f"{type(self.model).__name__} names the default criterion {default!r}, "
                f"which is not one of {sorted(CRITERIA)}: give the criterion"
            )
        return default

    def _why_unavailable(self, votes: VoteMatrix) -> Unavailable | None:
        if votes.n_votes == 0:
            return availability(votes)
        if all(
            availability(votes, n_groups, n_blocks) is not None
            for n_groups in self.n_groups
            for n_blocks in self.n_blocks
        ):
            return Unavailable(
                UnavailableReason.NOT_ENOUGH_DATA,
                f"no shape of the grid has enough voters and items with a vote: "
                f"{votes.n_active_voters} voters and {votes.n_active_items} items with a vote",
            )
        return None

    def _compute(self, votes: VoteMatrix, *, seed: int, computation: str) -> Coclustering:
        criterion = self.effective_criterion
        grid = (len(self.n_groups), len(self.n_blocks))
        scores = np.full(grid, np.nan)
        errors = np.full(grid, np.nan)
        folds = np.random.default_rng(seed).integers(0, self.n_folds, size=votes.shape)

        def score(row: int, column: int) -> float:
            if math.isnan(scores[row, column]):
                value, error = self._score(
                    votes, self.n_groups[row], self.n_blocks[column], criterion, folds, seed
                )
                scores[row, column], errors[row, column] = value, error
            return float(scores[row, column])

        if self.search == "grid":
            for row in range(grid[0]):
                for column in range(grid[1]):
                    score(row, column)
        else:
            row, column = 0, 0
            current = score(row, column)
            while True:
                moves = [
                    (row + 1, column) if row + 1 < grid[0] else None,
                    (row, column + 1) if column + 1 < grid[1] else None,
                ]
                candidates = [(score(*move), move) for move in moves if move is not None]
                if not candidates:
                    break
                best_value, best_move = max(candidates, key=lambda pair: pair[0])
                if not best_value > current:
                    break
                current, (row, column) = best_value, best_move

        visited = np.where(np.isnan(scores), -np.inf, scores)
        best_row, best_column = (
            int(index) for index in np.unravel_index(int(np.argmax(visited)), grid)
        )
        best = (self.n_groups[best_row], self.n_blocks[best_column])
        best_error = errors[best_row, best_column]
        alternatives: tuple[tuple[int, int], ...] = ()
        if criterion == "held_out_likelihood" and math.isfinite(best_error):
            floor = scores[best_row, best_column] - best_error
            alternatives = tuple(
                (self.n_groups[row], self.n_blocks[column])
                for row in range(grid[0])
                for column in range(grid[1])
                if (row, column) != (best_row, best_column)
                and math.isfinite(scores[row, column])
                and scores[row, column] >= floor
            )
        selection = Selection(
            criterion=criterion,
            n_groups=self.n_groups,
            n_blocks=self.n_blocks,
            scores=scores,
            standard_errors=errors if criterion == "held_out_likelihood" else None,
            best=best,
            alternatives=alternatives,
        )
        final = self.model.with_shape(*best).estimate(votes, seed=seed).value
        chosen: Coclustering = dataclasses.replace(
            final, selection=selection, computation=computation
        )
        return chosen

    def _score(
        self,
        votes: VoteMatrix,
        n_groups: int,
        n_blocks: int,
        criterion: str,
        folds: Any,
        seed: int,
    ) -> tuple[float, float]:
        """Return the score of one shape and its standard error (NaN when there is none)."""
        if availability(votes, n_groups, n_blocks) is not None:
            return -math.inf, math.nan
        model = self.model.with_shape(n_groups, n_blocks)
        if criterion == "integrated_likelihood":
            value = model.estimate(votes, seed=seed).value
            if not isinstance(value, HasIntegratedLikelihood):
                raise InvalidParameterError(
                    f"the criterion 'integrated_likelihood' reads values that carry one; "
                    f"{type(model).__name__} returns {type(value).__name__}: use "
                    f"'held_out_likelihood'"
                )
            return float(value.integrated_likelihood), math.nan
        per_fold: list[float] = []
        for fold in range(self.n_folds):
            held = votes.shown & (folds == fold)
            if not held.any():
                continue
            result = model.estimate(votes.hold_out(held), seed=seed)
            if not result.is_available:
                return -math.inf, math.nan
            value = result.value
            if not isinstance(value, PredictsVotes):
                raise InvalidParameterError(
                    f"the criterion 'held_out_likelihood' reads values that predict votes; "
                    f"{type(model).__name__} returns {type(value).__name__}"
                )
            logs: FloatArray = value.log_probability(votes)
            per_fold.append(float(logs[held].mean()))
        if not per_fold:
            return -math.inf, math.nan
        error = (
            float(np.std(per_fold, ddof=1) / math.sqrt(len(per_fold)))
            if len(per_fold) > 1
            else math.nan
        )
        return float(np.mean(per_fold)), error


def select_shape(
    votes: VoteMatrix,
    model: CoclusteringEstimator[Any] | str = "single",
    *,
    n_groups: Iterable[int] = (1, 2, 3, 4),
    n_blocks: Iterable[int] = (1, 2, 3),
    criterion: str | None = None,
    n_folds: int = 5,
    search: Literal["grid", "greedy"] = "grid",
    seed: int = 0,
    registry: Registry | None = None,
) -> Result[Coclustering]:
    """Choose the shape of a block model on ``votes`` and return the model at that shape.

    A shortcut for ``ShapeSelection(model=..., ...).estimate(votes, seed=seed)``; see
    :class:`ShapeSelection` for the definition.

    Args:
        votes: The vote matrix.
        model: A block model, or a plain word of the registry such as ``"single"`` or
            ``"mixed"`` for its default estimator.
        n_groups: The numbers of groups compared.
        n_blocks: The numbers of blocks compared.
        criterion: A key of :data:`CRITERIA`, or ``None`` for the default criterion of
            the model.
        n_folds: The number of folds of the held-out likelihood, at least 2.
        search: ``"grid"`` or ``"greedy"``.
        seed: Seed of every draw, a non-negative integer.
        registry: Where to look a plain word up. Defaults to the installed entry points.

    Returns:
        The result of :class:`ShapeSelection`: the co-clustering at the best shape, whose
        ``selection`` holds the scores of every shape.

    Raises:
        InvalidParameterError: If a setting is out of range, or the plain word unknown.
    """
    estimator = estimator_for(model, registry) if isinstance(model, str) else model
    return ShapeSelection(
        model=estimator,
        n_groups=tuple(n_groups),
        n_blocks=tuple(n_blocks),
        criterion=criterion,
        n_folds=n_folds,
        search=search,
    ).estimate(votes, seed=seed)
