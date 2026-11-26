# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Response models: how the cells of a co-cluster spread over the states of a cell.

A block model is made of two independent things: a latent structure (who belongs to
which group, which item to which block) and a response model, which says how the cells
of a co-cluster spread between "shown without a vote" and each score.

A shown cell has K + 1 states: state 0 (shown without a vote) and state ``k`` (vote at
rank ``k``, from 1 to K). A response model reads counts of cells per state and does
not know whether they are integers (single membership) or fractional (mixed
membership).

When the items use several scales, a cell of an item of the scale ``c`` has the states
0 to ``K_c``. A response model then reads its counts and gives its probabilities scale
by scale, with an axis of scales before the axis of states: ``counts[q, l, c, s]`` and
``log_probabilities_by_scale(params)[q, l, c, s]``. A co-cluster keeps one set of
parameters: the scales only change how a vote is observed.

A response model is a frozen dataclass: its settings enter the provenance of the
estimator that carries it as they are.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise
from typing import Literal, Protocol, TypeAlias, TypeVar, cast

import numpy as np
from numpy.typing import ArrayLike

from civiqhub_selene.blocks._arrays import BoolArray, FloatArray
from civiqhub_selene.blocks._numerics import (
    fit_latent_normal_by_scale,
    fit_thresholds,
    level_probabilities_by_scale,
    scale_bounds,
)
from civiqhub_selene.blocks.coclusters import CoclusterTable, state_log_probabilities
from civiqhub_selene.blocks.data import OrdinalScale, VoteMatrix
from civiqhub_selene.errors import InvalidInputError, InvalidParameterError

__all__ = [
    "Categorical",
    "CategoricalParameters",
    "OrdinalProbit",
    "OrdinalProbitParameters",
    "ResponseModel",
    "Thresholds",
]

ParametersT = TypeVar("ParametersT")

_EMPTY_COCLUSTER = 1e-12
_MIN_VOTE_SHARE = 1e-12

ScalesOfTable: TypeAlias = OrdinalScale | Sequence[OrdinalScale]
"""The scales of a table: one scale, or the scales of the items in their order."""


class ResponseModel(Protocol[ParametersT]):
    """What a block model asks of the response model of its co-clusters.

    ``ParametersT`` is the type of the parameters of the model, for Q x L co-clusters.
    Every response model writes the probability of the state ``s`` of a shown cell of
    the co-cluster ``(q, l)`` with a vote rate ``nu[q, l]`` and shares of the votes
    ``p[q, l, k]`` that sum to 1 over the ranks::

        F[q, l, 0] = 1 - nu[q, l]                (shown without a vote)
        F[q, l, k] = nu[q, l] * p[q, l, k]       (vote at rank k, from 1 to K)

    Counts and probabilities carry an axis of scales: with C scales of K levels at
    most, ``counts[q, l, c, s]`` and the log-probabilities have the shape
    (Q, L, C, K + 1). With one scale, the counts may also be given without that axis.

    See the mathematical model: :doc:`/model/response-models`, equations
    :eq:`eq-response-states`, :eq:`eq-response-expected` and :eq:`eq-response-penalty`.
    """

    def initial_parameters(
        self, shape: tuple[int, int], votes: VoteMatrix, rng: np.random.Generator
    ) -> ParametersT:
        """Return starting parameters for ``shape = (Q, L)`` co-clusters on ``votes``.

        The scales of the items of ``votes`` fix the scales of the parameters.
        """
        ...

    def log_probabilities(self, params: ParametersT) -> FloatArray:
        """Return ``log F[q, l, s]`` for each state in each co-cluster, shape (Q, L, K + 1).

        Defined for parameters of one scale.
        """
        ...

    def log_probabilities_by_scale(self, params: ParametersT) -> FloatArray:
        """Return the log-probability of each state of each scale, shape (Q, L, C, K + 1).

        A state past the levels of a scale has a probability floored at 1e-300, so its
        logarithm is finite; a block model never reads it, since no cell is in it.
        """
        ...

    def update(self, counts: FloatArray, previous: ParametersT) -> ParametersT:
        """Return the parameters that maximise the expected likelihood.

        ``counts[q, l, c, s]``, written ``N[q, l, c, s]``, is the number, integer or
        fractional, of cells of the items of the scale ``c`` in the co-cluster
        ``(q, l)`` in the state ``s``; with one scale, ``counts[q, l, s]`` is accepted
        too. The expected likelihood of each co-cluster is
        ``sum_c sum_s N[q, l, c, s] * log F[q, l, c, s]``.
        """
        ...

    def complexity_penalty(self, shape: tuple[int, int], votes: VoteMatrix) -> float:
        """Return the penalty of the parameters in the integrated likelihood.

        Half a logarithm of the sample size per free parameter.

        See the mathematical model: :doc:`/model/response-models`, equation
        :eq:`eq-response-penalty`.
        """
        ...

    def to_table(
        self, params: ParametersT, counts: FloatArray, scale: ScalesOfTable
    ) -> CoclusterTable:
        """Return the readable table of the co-clusters, on ``scale`` or on the scales given."""
        ...


def _by_scale(counts: FloatArray) -> FloatArray:
    """Return counts with an axis of scales: (Q, L, S) becomes (Q, L, 1, S)."""
    array = np.asarray(counts, dtype=np.float64)
    if array.ndim == 3:
        return array[:, :, np.newaxis, :]
    if array.ndim != 4:
        raise InvalidInputError(
            f"counts have the shape (groups, blocks, scales, states), got {array.shape}"
        )
    return array


def _table_scales(scale: ScalesOfTable) -> tuple[OrdinalScale, ...]:
    if isinstance(scale, OrdinalScale):
        return (scale,)
    return tuple(scale)


def _vote_rate(counts: FloatArray, previous: FloatArray, min_probability: float) -> FloatArray:
    """Return the vote rate of each co-cluster, bounded away from 0 and 1.

    ``votes / cells`` where the co-cluster has cells (a sum above 1e-12), the previous
    value elsewhere; the whole bounded between ``min_probability`` and
    ``1 - min_probability``. The votes and the cells are summed over the states, then
    over the scales.
    """
    by_scale = _by_scale(counts)
    n_cells = by_scale.sum(axis=-1).sum(axis=-1)
    n_votes = by_scale[..., 1:].sum(axis=-1).sum(axis=-1)
    rate = np.divide(n_votes, n_cells, out=previous.copy(), where=n_cells > _EMPTY_COCLUSTER)
    bounded: FloatArray = np.clip(rate, min_probability, 1.0 - min_probability)
    return bounded


def _overall_vote_rate(votes: VoteMatrix, min_probability: float) -> float:
    """Return the votes over the shown cells of the matrix, bounded away from 0 and 1."""
    weights = votes._likelihood_weights
    if weights is None:
        rate = votes.n_votes / max(votes.n_shown, 1)
    else:
        shown = float(votes.shown.sum(axis=1) @ weights)
        voted = float(votes.voted.sum(axis=1) @ weights)
        rate = voted / shown if shown > 0 else 0.0
    return min(max(rate, min_probability), 1.0 - min_probability)


def _grid_array(grid: ArrayLike) -> FloatArray:
    """Return a grid as a one-dimensional array; an array that already is one is kept."""
    if isinstance(grid, np.ndarray) and grid.dtype == np.float64 and grid.ndim == 1:
        return grid
    array: FloatArray = np.asarray(grid, dtype=np.float64).reshape(-1)
    return array


def _one_scale(n_scales: int, reading: str) -> None:
    """Refuse a reading of one scale on parameters of several."""
    if n_scales != 1:
        raise InvalidInputError(
            f"these parameters have {n_scales} scales, not one",
            hint=f"Read {reading}, or the form that ends with _by_scale.",
        )


def _frozen(array: FloatArray) -> FloatArray:
    array.setflags(write=False)
    return array


def _padded(shares: tuple[FloatArray, ...]) -> FloatArray:
    """Stack shares of shape (Q, L, K_c) into (Q, L, C, K), zeros past each scale."""
    n_levels = max(array.shape[-1] for array in shares)
    padded: FloatArray = np.zeros((*shares[0].shape[:-1], len(shares), n_levels))
    for position, array in enumerate(shares):
        padded[:, :, position, : array.shape[-1]] = array
    return _frozen(padded)


@dataclass(frozen=True, eq=False, init=False)
class CategoricalParameters:
    """Parameters of the :class:`Categorical` response model.

    Args:
        vote_rate: Shape (Q, L): probability that a shown cell carries a vote.
        vote_shares: Shape (Q, L, K): distribution of the votes over the scores; with
            several scales, a tuple of one array of shape (Q, L, K_c) per scale.

    Attributes:
        vote_rate: As given.
        vote_shares_by_scale: One array of shape (Q, L, K_c) per scale.
    """

    vote_rate: FloatArray
    vote_shares_by_scale: tuple[FloatArray, ...]

    def __init__(self, vote_rate: ArrayLike, vote_shares: ArrayLike | Sequence[ArrayLike]) -> None:
        object.__setattr__(self, "vote_rate", np.asarray(vote_rate, dtype=np.float64))
        if isinstance(vote_shares, tuple) and all(isinstance(a, np.ndarray) for a in vote_shares):
            shares = tuple(np.asarray(array, dtype=np.float64) for array in vote_shares)
        else:
            shares = (np.asarray(vote_shares, dtype=np.float64),)
        object.__setattr__(self, "vote_shares_by_scale", shares)

    @property
    def vote_shares(self) -> FloatArray:
        """Shape (Q, L, K): the distribution of the votes, for parameters of one scale.

        Raises:
            InvalidInputError: If the parameters have several scales.
        """
        _one_scale(len(self.vote_shares_by_scale), "vote_shares_by_scale")
        return self.vote_shares_by_scale[0]

    @cached_property
    def _padded_shares(self) -> FloatArray:
        return _padded(self.vote_shares_by_scale)

    @cached_property
    def _possible(self) -> BoolArray:
        """Shape (C, K): the level ``k`` exists on the scale ``c``."""
        n_levels = max(array.shape[-1] for array in self.vote_shares_by_scale)
        levels = np.array([array.shape[-1] for array in self.vote_shares_by_scale])
        possible: BoolArray = np.arange(n_levels)[np.newaxis, :] < levels[:, np.newaxis]
        return possible


@dataclass(frozen=True)
class Categorical:
    """A vote rate per co-cluster, then a free distribution over the K scores of each scale.

    Definition. For a shown cell of the co-cluster ``(q, l)``, with the vote rate
    ``nu[q, l]`` (``vote_rate``) and the shares ``p[q, l, k]`` (``vote_shares``)::

        P(no vote) = F[q, l, 0] = 1 - nu[q, l]
        P(vote at rank k) = F[q, l, k] = nu[q, l] * p[q, l, k]

    with K - 1 free parameters per co-cluster for the distribution and no constraint on
    its shape. The update is the maximum of the expected likelihood: the vote rate is
    the share of cells that carry a vote, and the distribution is the share of votes at
    each rank, after ``smoothing`` is added to each count; the vote rate is then kept
    between ``min_probability`` and ``1 - min_probability``::

        nu[q, l] = sum_{k >= 1} N[q, l, k] / sum_s N[q, l, s]
        p[q, l, k] = (N[q, l, k] + smoothing) / sum_h (N[q, l, h] + smoothing)

    When the items use several scales, each co-cluster has one free distribution per
    scale, over the ``K_c`` scores of that scale, and one vote rate for all: a cell of
    an item of the scale ``c`` votes at rank ``k`` with the probability
    ``vote_rate[q, l] * vote_shares_by_scale[c][q, l, k]``. Nothing ties the
    distributions of two scales together.

    It is the simple reference against which the cost and the benefit of the ordinal
    probit model are judged.

    See the mathematical model: :doc:`/model/response-models`, equations
    :eq:`eq-response-rate` and :eq:`eq-response-categorical`.

    Attributes:
        smoothing: Pseudo-count added to the number of votes at each rank before the
            distribution is normalised, at least 0. Defaults to 0.5.
        min_probability: Bound that keeps a vote rate away from 0 and from 1, strictly
            between 0 and 0.5. Defaults to 1e-6.

    Raises:
        InvalidParameterError: If a setting is outside its range.
    """

    smoothing: float = 0.5
    min_probability: float = 1e-6

    def __post_init__(self) -> None:
        if not self.smoothing >= 0 or math.isinf(self.smoothing):
            raise InvalidParameterError(
                f"smoothing is a finite number of at least 0, got {self.smoothing!r}"
            )
        if not 0 < self.min_probability < 0.5:
            raise InvalidParameterError(
                f"min_probability is strictly between 0 and 0.5, got {self.min_probability!r}"
            )

    def initial_parameters(
        self, shape: tuple[int, int], votes: VoteMatrix, rng: np.random.Generator
    ) -> CategoricalParameters:
        """Return starting parameters: the overall vote rate, and drawn distributions.

        The vote rate is the same everywhere: the votes over the shown cells of the
        matrix, bounded. Each distribution is drawn from a Dirichlet law of
        concentration 5 on each of the K scores of its scale, the scales in their order.
        """
        rate = _overall_vote_rate(votes, self.min_probability)
        shares = tuple(
            rng.dirichlet(np.full(scale.n_levels, 5.0), size=shape) for scale in votes.scales
        )
        return CategoricalParameters(np.full(shape, rate), shares)

    def log_probabilities(self, params: CategoricalParameters) -> FloatArray:
        """Return the log-probability of each state in each co-cluster, shape (Q, L, K + 1).

        Raises:
            InvalidInputError: If the parameters have several scales.
        """
        return state_log_probabilities(params.vote_rate, params.vote_shares)

    def log_probabilities_by_scale(self, params: CategoricalParameters) -> FloatArray:
        """Return the log-probability of each state of each scale, shape (Q, L, C, K + 1)."""
        return state_log_probabilities(params.vote_rate, params._padded_shares)

    def update(self, counts: FloatArray, previous: CategoricalParameters) -> CategoricalParameters:
        """Return the vote rates and the distributions that the counts call for.

        A co-cluster without any cell keeps its previous vote rate. A co-cluster
        without any vote on a scale keeps its previous distribution on that scale when
        ``smoothing`` is 0.
        """
        by_scale = _by_scale(counts)
        shares: list[FloatArray] = []
        for position, before in enumerate(previous.vote_shares_by_scale):
            smoothed = by_scale[:, :, position, 1 : before.shape[-1] + 1] + self.smoothing
            total = smoothed.sum(axis=-1, keepdims=True)
            shares.append(
                np.divide(smoothed, total, out=before.copy(), where=total > _EMPTY_COCLUSTER)
            )
        vote_rate = _vote_rate(by_scale, previous.vote_rate, self.min_probability)
        return CategoricalParameters(vote_rate, tuple(shares))

    def complexity_penalty(self, shape: tuple[int, int], votes: VoteMatrix) -> float:
        """Return half a logarithm of the sample size per free parameter.

        ``0.5 * Q * L * sum_c(K_c - 1) * log(max(n_votes, 1))`` for the distributions,
        ``K_c`` the levels of each scale, plus ``0.5 * Q * L * log(max(n_shown, 1))``
        for the vote rates.

        See the mathematical model: :doc:`/model/response-models`, equation
        :eq:`eq-response-penalty`.
        """
        n_groups, n_blocks = shape
        n_coclusters = n_groups * n_blocks
        free = sum(scale.n_levels - 1 for scale in votes.scales)
        distributions = 0.5 * n_coclusters * free * math.log(max(votes.n_votes, 1))
        vote_rates = 0.5 * n_coclusters * math.log(max(votes.n_shown, 1))
        return distributions + vote_rates

    def to_table(
        self, params: CategoricalParameters, counts: FloatArray, scale: ScalesOfTable
    ) -> CoclusterTable:
        """Return the table of the co-clusters, with the number of votes of the counts."""
        n_votes = _by_scale(counts)[..., 1:].sum(axis=-1).sum(axis=-1)
        if isinstance(scale, OrdinalScale):
            return CoclusterTable(scale, params.vote_rate, params.vote_shares, n_votes=n_votes)
        return CoclusterTable(
            _table_scales(scale), params.vote_rate, params.vote_shares_by_scale, n_votes=n_votes
        )


ThresholdAxis: TypeAlias = Literal["auto", "levels", "unit"]
"""The axis on which the regular thresholds of the ordinal probit are placed."""

ThresholdMode: TypeAlias = Literal["fixed", "estimated"]
"""Whether the inner thresholds of each scale are held or estimated."""


def _check_grid(scale: OrdinalScale, grid: Sequence[float]) -> tuple[float, ...]:
    try:
        values = tuple(float(threshold) for threshold in grid)
    except (TypeError, ValueError):
        raise InvalidParameterError(f"thresholds are numbers, got {grid!r}") from None
    _check_thresholds(values, scale.n_levels)
    return values


@dataclass(frozen=True)
class Thresholds:
    """Where the thresholds of the ordinal probit lie, for each scale of the items.

    The ordinal probit reads a vote as a hidden opinion cut by the thresholds of the
    scale of its item. Every scale has its own grid; the opinion of a co-cluster, its
    mean and its spread, is one and the same on every scale. The grids must therefore
    sit on one common axis, and this object says which:

    - ``axis="levels"``: a scale of ``K_c`` levels has the thresholds ``k + 1/2``, for
      ``k`` from 1 to ``K_c - 1``. The mean reads on the axis of levels; the first
      levels of every scale line up, a short scale covering the bottom of the axis.
    - ``axis="unit"``: a scale of ``K_c`` levels has the thresholds ``k / K_c``, for
      ``k`` from 1 to ``K_c - 1``: every scale covers the axis from 0 to 1, each of its
      levels taking an equal share of it, so that "yes" on two answers and the two top
      answers of four lie on the same side of 1/2. This is a convention, not a
      measurement: it assumes that the answers of two scales that cut the axis at the
      same place mean the same position.
    - ``axis="auto"``, the default: ``"levels"`` when the items share one scale, and
      ``"unit"`` when they use several. A matrix of one scale is then computed exactly
      as before items could have several.

    ``grids`` gives the thresholds of some scales by hand, on the chosen axis, in place
    of the regular ones; the other scales keep their regular grid. A grid named for a
    scale that a matrix does not use is not read.

    The axis also sets the unit of the spread: one level of the longest scale, ``1`` on
    the axis of levels and ``1 / K`` on the unit axis for a longest scale of K levels.
    The floor ``min_spread`` of the ordinal probit is read in that unit, and so is the
    start of the spread, so that the two axes give the same arrangements on a matrix of
    one scale.

    ``mode="estimated"`` estimates the inner thresholds of each scale, those that lie
    between its first and its last: a scale of ``K_c`` levels has ``K_c - 3`` of them,
    so that only scales of four levels or more have any. The grids above give their
    start; the first and the last thresholds of each scale stay where the grid puts
    them. They anchor the scale on the axis, and fix the location and the unit that the
    opinions would otherwise share with the thresholds (two scales whose ends were free
    could slide against each other). The estimate tests the convention of the axis: on
    data where the answers of a scale are not regular, its inner thresholds move away
    from the regular grid. Each update of the ordinal probit then takes, after the
    opinions, Newton steps on the inner thresholds of each scale, at the opinions found,
    each step kept only if it keeps the order and does not lower the likelihood: a
    generalised expectation-maximisation, whose objective never decreases. Each inner
    threshold adds half a logarithm of the number of votes to the penalty of the
    integrated likelihood (a Hypothesis: the thresholds are shared by every co-cluster).

    See the mathematical model: :doc:`/model/response-models`, equations
    :eq:`eq-response-axes` and :eq:`eq-response-thresholds-estimated`.

    Attributes:
        axis: ``"auto"``, ``"levels"`` or ``"unit"``. Defaults to ``"auto"``.
        grids: The thresholds of some scales, given by hand: a mapping from an
            :class:`~civiqhub_selene.blocks.OrdinalScale` to its ``K_c - 1``
            thresholds, strictly increasing. Kept as pairs sorted by scores.
        mode: ``"fixed"``, the default, or ``"estimated"``.

    Raises:
        InvalidParameterError: If the axis or the mode is unknown, or if a grid is not
            a strictly increasing grid of ``K_c - 1`` finite numbers.
    """

    axis: ThresholdAxis = "auto"
    grids: (
        Mapping[OrdinalScale, Sequence[float]] | tuple[tuple[OrdinalScale, tuple[float, ...]], ...]
    ) = ()
    mode: ThresholdMode = "fixed"

    def __post_init__(self) -> None:
        if self.axis not in ("auto", "levels", "unit"):
            raise InvalidParameterError(
                f"the axis of the thresholds is 'auto', 'levels' or 'unit', got {self.axis!r}"
            )
        if self.mode not in ("fixed", "estimated"):
            raise InvalidParameterError(
                f"the mode of the thresholds is 'fixed' or 'estimated', got {self.mode!r}"
            )
        received = self.grids
        pairs = received.items() if isinstance(received, Mapping) else received
        checked: dict[tuple[int, ...], tuple[OrdinalScale, tuple[float, ...]]] = {}
        for pair in pairs:
            try:
                scale, grid = pair
            except (TypeError, ValueError):
                raise InvalidParameterError(
                    f"grids map an OrdinalScale to its thresholds, got {pair!r}"
                ) from None
            if not isinstance(scale, OrdinalScale):
                raise InvalidParameterError(f"grids are keyed by OrdinalScale, got {scale!r}")
            if scale.scores in checked:
                raise InvalidParameterError(f"grids name the scale {scale.scores!r} twice")
            checked[scale.scores] = (scale, _check_grid(scale, grid))
        object.__setattr__(self, "grids", tuple(checked[key] for key in sorted(checked)))

    def grid_of(self, scale: OrdinalScale, axis: Literal["levels", "unit"]) -> tuple[float, ...]:
        """Return the thresholds of ``scale``: its grid given by hand, or the regular one."""
        for named, grid in cast(tuple[tuple[OrdinalScale, tuple[float, ...]], ...], self.grids):
            if named.scores == scale.scores:
                return grid
        levels = scale.n_levels
        if axis == "levels":
            return tuple(level + 0.5 for level in range(1, levels))
        return tuple(level / levels for level in range(1, levels))

    @property
    def is_estimated(self) -> bool:
        """Whether the inner thresholds are estimated."""
        return self.mode == "estimated"

    def axis_for(self, votes: VoteMatrix) -> Literal["levels", "unit"]:
        """Return the axis used on ``votes``: ``"auto"`` resolved by its number of scales."""
        if self.axis == "auto":
            return "levels" if votes.item_scales.is_uniform else "unit"
        return self.axis


@dataclass(frozen=True, eq=False, init=False)
class OrdinalProbitParameters:
    """Parameters of the :class:`OrdinalProbit` response model.

    Args:
        vote_rate: Shape (Q, L): probability that a shown cell carries a vote.
        latent_mean: Shape (Q, L): mean of the latent opinion, on the axis of the
            thresholds.
        latent_spread: Shape (Q, L): standard deviation of the latent opinion.
        thresholds: The K - 1 thresholds between levels, fixed during the computation;
            with several scales, a tuple of one grid per scale.
        spread_unit: One level of the longest scale on the axis of the thresholds: 1 on
            the axis of levels, ``1 / K`` on the unit axis. The floor of the spread is
            ``min_spread`` times it, and a co-cluster whose votes are all on scales of
            two levels keeps a spread of exactly this value. Defaults to 1.

    Attributes:
        vote_rate: As given.
        latent_mean: As given.
        latent_spread: As given.
        thresholds_by_scale: One grid of thresholds per scale.
        spread_unit: As given.
    """

    vote_rate: FloatArray
    latent_mean: FloatArray
    latent_spread: FloatArray
    thresholds_by_scale: tuple[FloatArray, ...]
    spread_unit: float

    def __init__(
        self,
        vote_rate: ArrayLike,
        latent_mean: ArrayLike,
        latent_spread: ArrayLike,
        thresholds: ArrayLike | Sequence[ArrayLike],
        *,
        spread_unit: float = 1.0,
    ) -> None:
        if (
            isinstance(thresholds, tuple)
            and thresholds
            and all(isinstance(grid, np.ndarray) for grid in thresholds)
        ):
            grids = tuple(_grid_array(grid) for grid in thresholds)
        else:
            grids = (_grid_array(thresholds),)
        object.__setattr__(self, "vote_rate", np.asarray(vote_rate, dtype=np.float64))
        object.__setattr__(self, "latent_mean", np.asarray(latent_mean, dtype=np.float64))
        object.__setattr__(self, "latent_spread", np.asarray(latent_spread, dtype=np.float64))
        object.__setattr__(self, "thresholds_by_scale", grids)
        object.__setattr__(self, "spread_unit", float(spread_unit))

    @property
    def thresholds(self) -> FloatArray:
        """The K - 1 thresholds, for parameters of one scale.

        Raises:
            InvalidInputError: If the parameters have several scales.
        """
        _one_scale(len(self.thresholds_by_scale), "thresholds_by_scale")
        return self.thresholds_by_scale[0]

    @cached_property
    def _grid(self) -> tuple[FloatArray, BoolArray]:
        """The bounds (C, K + 1) and the existing levels (C, K) of the scales."""
        return scale_bounds(self.thresholds_by_scale)

    @cached_property
    def _long_scales(self) -> BoolArray:
        """Shape (C,): the scales of three levels or more, which fix a spread."""
        long: BoolArray = np.array([grid.shape[0] >= 2 for grid in self.thresholds_by_scale])
        return long


@dataclass(frozen=True)
class OrdinalProbit:
    """A vote rate per co-cluster, then a latent normal opinion cut into K levels.

    Definition. In the co-cluster ``(q, l)``, a shown cell carries a vote with the
    probability ``nu[q, l]`` (``vote_rate``). The vote comes from a hidden continuous
    opinion, normal with mean ``mu[q, l]`` (``latent_mean``) and standard deviation
    ``sigma[q, l]`` (``latent_spread``), which fixed thresholds ``t_1 < ... < t_{K-1}``
    cut into K levels, with ``t_0 = -inf`` and ``t_K = +inf``::

        P(no vote) = F[q, l, 0] = 1 - nu[q, l]
        P(vote at rank k) = F[q, l, k] = nu[q, l] * p[q, l, k]
        p[q, l, k] = Phi((t_k - mu[q, l]) / sigma[q, l])
                     - Phi((t_{k-1} - mu[q, l]) / sigma[q, l])

    where ``Phi`` is the standard normal distribution function. Two numbers then sum
    up how a group answers a block: where the opinion stands and how spread it is.

    On items of one scale, the thresholds default to ``k + 1/2`` for ``k`` from 1 to
    K - 1: ``latent_mean`` then reads directly on the axis of levels, from 1 to K.
    They are fixed unless :class:`Thresholds` sets ``mode="estimated"``, which
    estimates the inner thresholds of each scale. The mean and the spread are
    estimated by expectation-maximisation on the hidden opinion (see
    ``fit_latent_normal_by_scale``).

    Items on several scales. Each scale ``c`` has its own grid ``t_{c,1} < ... <
    t_{c,K_c-1}`` (see :class:`Thresholds`), and a vote on an item of the scale ``c``
    is cut by that grid; the opinion of the co-cluster, its ``latent_mean`` and its
    ``latent_spread``, stay one and the same for all its items. This assumes that the
    grids sit on one common axis, which the axis of :class:`Thresholds` sets: by default
    the unit axis, where every scale covers the axis from 0 to 1 in equal shares. The
    expectation of the hidden opinion of a vote uses the grid of its scale; the
    maximisation sums the moments over the levels, then over the scales.

    A vote on a scale of two levels only says on which side of one threshold the
    opinion lies: such votes fix ``(threshold - latent_mean) / latent_spread`` and not
    the mean and the spread apart. In a co-cluster whose votes are all on scales of two
    levels, the spread is therefore held at one level of the longest scale
    (``spread_unit`` of the parameters) and only the mean is estimated; any vote on a
    scale of three levels or more lets the spread be estimated.

    Reference: Corneli, Bouveyron, Latouche (2020), Co-clustering of ordinal data via
    latent continuous random variables and not missing at random entries.

    See the mathematical model: :doc:`/model/response-models`, equations
    :eq:`eq-response-probit`, :eq:`eq-response-probit-by-scale`, :eq:`eq-response-axes`,
    :eq:`eq-response-rate`, :eq:`eq-response-moments`, :eq:`eq-response-opinion`,
    :eq:`eq-response-opinion-by-scale` and :eq:`eq-response-thresholds-estimated`.

    Attributes:
        thresholds: Where the thresholds lie. ``None`` for the default ones (see
            :class:`Thresholds`, ``axis="auto"``); the K - 1 thresholds of the one scale
            of the items, strictly increasing; a mapping from each scale to its grid; or
            a :class:`Thresholds`. A mapping is kept as ``Thresholds(grids=...)``.
        min_spread: Lower bound of ``latent_spread``, strictly positive, in levels of
            the longest scale. It keeps a unanimous co-cluster from sending its spread to
            zero. Defaults to 0.3.
        min_probability: Bound that keeps a vote rate away from 0 and from 1, strictly
            between 0 and 0.5. Defaults to 1e-6.
        max_inner_iterations: Largest number of expectation-maximisation steps on the
            hidden opinion at each update, at least 1. Defaults to 10.

    Raises:
        InvalidParameterError: If a setting is outside its range.
    """

    thresholds: tuple[float, ...] | Thresholds | Mapping[OrdinalScale, Sequence[float]] | None = (
        None
    )
    min_spread: float = 0.3
    min_probability: float = 1e-6
    max_inner_iterations: int = 10

    def __post_init__(self) -> None:
        if isinstance(self.thresholds, Mapping):
            object.__setattr__(self, "thresholds", Thresholds(grids=self.thresholds))
        elif self.thresholds is not None and not isinstance(self.thresholds, Thresholds):
            try:
                thresholds = tuple(float(threshold) for threshold in self.thresholds)
            except (TypeError, ValueError):
                raise InvalidParameterError(
                    f"thresholds are numbers, got {self.thresholds!r}"
                ) from None
            _check_thresholds(thresholds, len(thresholds) + 1)
            object.__setattr__(self, "thresholds", thresholds)
        if not self.min_spread > 0 or math.isinf(self.min_spread):
            raise InvalidParameterError(
                f"min_spread is a finite number above 0, got {self.min_spread!r}"
            )
        if not 0 < self.min_probability < 0.5:
            raise InvalidParameterError(
                f"min_probability is strictly between 0 and 0.5, got {self.min_probability!r}"
            )
        steps = self.max_inner_iterations
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise InvalidParameterError(
                f"max_inner_iterations is an integer of at least 1, got {steps!r}"
            )

    def initial_parameters(
        self, shape: tuple[int, int], votes: VoteMatrix, rng: np.random.Generator
    ) -> OrdinalProbitParameters:
        """Return starting parameters: the overall vote rate and opinions near the middle.

        The vote rate is the same everywhere: the votes over the shown cells of the
        matrix, bounded. On the axis of levels, each mean is the middle of the axis,
        ``(K + 1) / 2``, plus a normal draw of standard deviation 0.5, and each spread
        is 1. On the unit axis, the same start is read in units of one level of the
        longest scale: the middle ``1/2`` plus the draw divided by K, and the spread
        ``1 / K``. The grid of each scale of ``votes`` comes from the setting
        ``thresholds``.

        Raises:
            InvalidParameterError: If the thresholds given do not fit the scales of the
                matrix.
        """
        n_levels = votes.n_levels
        axis, grids = self._grids(votes)
        rate = _overall_vote_rate(votes, self.min_probability)
        if axis == "levels":
            latent_mean: FloatArray = (n_levels + 1) / 2 + rng.normal(0.0, 0.5, size=shape)
            spread_unit = 1.0
        else:
            latent_mean = 0.5 + rng.normal(0.0, 0.5, size=shape) / n_levels
            spread_unit = 1.0 / n_levels
        return OrdinalProbitParameters(
            np.full(shape, rate),
            latent_mean,
            np.full(shape, spread_unit),
            tuple(np.array(grid) for grid in grids),
            spread_unit=spread_unit,
        )

    def log_probabilities(self, params: OrdinalProbitParameters) -> FloatArray:
        """Return the log-probability of each state in each co-cluster, shape (Q, L, K + 1).

        Raises:
            InvalidInputError: If the parameters have several scales.
        """
        _one_scale(len(params.thresholds_by_scale), "thresholds_by_scale")
        return state_log_probabilities(params.vote_rate, self._vote_shares(params)[:, :, 0, :])

    def log_probabilities_by_scale(self, params: OrdinalProbitParameters) -> FloatArray:
        """Return the log-probability of each state of each scale, shape (Q, L, C, K + 1)."""
        return state_log_probabilities(params.vote_rate, self._vote_shares(params))

    def update(
        self, counts: FloatArray, previous: OrdinalProbitParameters
    ) -> OrdinalProbitParameters:
        """Return the vote rates and the latent opinions that the counts call for.

        The mean and the spread restart from the previous ones and take at most
        ``max_inner_iterations`` steps. A co-cluster without any cell keeps its
        previous vote rate; a co-cluster without any vote keeps its previous opinion; a
        co-cluster whose votes are all on scales of two levels keeps the spread
        ``spread_unit``.

        See the mathematical model: :doc:`/model/response-models`, equations
        :eq:`eq-response-rate`, :eq:`eq-response-moments`, :eq:`eq-response-opinion`,
        :eq:`eq-response-opinion-by-scale` and :eq:`eq-response-thresholds-estimated`.
        """
        by_scale = _by_scale(counts)
        bounds, possible = previous._grid
        long = previous._long_scales
        estimable = by_scale[:, :, long, 1:].sum(axis=-1).sum(axis=-1) > _EMPTY_COCLUSTER
        latent_mean, latent_spread = fit_latent_normal_by_scale(
            by_scale[..., 1:],
            previous.latent_mean,
            previous.latent_spread,
            bounds,
            possible,
            min_spread=self.min_spread * previous.spread_unit,
            max_iterations=self.max_inner_iterations,
            estimable=estimable,
            fixed_spread=previous.spread_unit,
        )
        vote_rate = _vote_rate(by_scale, previous.vote_rate, self.min_probability)
        grids = previous.thresholds_by_scale
        if self._is_estimated:
            grids = tuple(
                fit_thresholds(
                    by_scale[:, :, position, 1 : grid.shape[0] + 2],
                    latent_mean,
                    latent_spread,
                    grid,
                    max_steps=self.max_inner_iterations,
                )
                for position, grid in enumerate(grids)
            )
        return OrdinalProbitParameters(
            vote_rate,
            latent_mean,
            latent_spread,
            grids,
            spread_unit=previous.spread_unit,
        )

    def complexity_penalty(self, shape: tuple[int, int], votes: VoteMatrix) -> float:
        """Return half a logarithm of the sample size per free parameter.

        Two parameters of opinion and one vote rate per co-cluster:
        ``Q * L * log(max(n_votes, 1)) + 0.5 * Q * L * log(max(n_shown, 1))``. Fixed
        grids add no parameter, whatever the number of scales; estimated ones add
        ``0.5 * log(max(n_votes, 1))`` per inner threshold, ``K_c - 3`` for a scale of
        ``K_c`` levels (a Hypothesis).

        See the mathematical model: :doc:`/model/response-models`, equation
        :eq:`eq-response-penalty`.
        """
        n_groups, n_blocks = shape
        n_coclusters = n_groups * n_blocks
        opinions = n_coclusters * math.log(max(votes.n_votes, 1))
        vote_rates = 0.5 * n_coclusters * math.log(max(votes.n_shown, 1))
        if not self._is_estimated:
            return opinions + vote_rates
        inner = sum(max(scale.n_levels - 3, 0) for scale in votes.scales)
        return opinions + vote_rates + 0.5 * inner * math.log(max(votes.n_votes, 1))

    def to_table(
        self, params: OrdinalProbitParameters, counts: FloatArray, scale: ScalesOfTable
    ) -> CoclusterTable:
        """Return the table of the co-clusters, with their latent opinion and the grids.

        ``has_fixed_spread`` marks the co-clusters whose votes are all on scales of two
        levels, whose spread was held at ``spread_unit`` rather than estimated.
        """
        scales = _table_scales(scale)
        padded = self._vote_shares(params)
        shares = tuple(
            padded[:, :, position, : one.n_levels] for position, one in enumerate(scales)
        )
        by_scale = _by_scale(counts)
        n_votes = by_scale[..., 1:].sum(axis=-1).sum(axis=-1)
        on_long = by_scale[:, :, params._long_scales, 1:].sum(axis=-1).sum(axis=-1)
        fixed = (n_votes > _EMPTY_COCLUSTER) & (on_long <= _EMPTY_COCLUSTER)
        if isinstance(scale, OrdinalScale):
            return CoclusterTable(
                scale,
                params.vote_rate,
                shares[0],
                n_votes=n_votes,
                latent_mean=params.latent_mean,
                latent_spread=params.latent_spread,
                thresholds=params.thresholds_by_scale[0],
                has_fixed_spread=fixed,
            )
        return CoclusterTable(
            scales,
            params.vote_rate,
            shares,
            n_votes=n_votes,
            latent_mean=params.latent_mean,
            latent_spread=params.latent_spread,
            thresholds=params.thresholds_by_scale,
            has_fixed_spread=fixed,
        )

    @property
    def _is_estimated(self) -> bool:
        return isinstance(self.thresholds, Thresholds) and self.thresholds.is_estimated

    def _grids(
        self, votes: VoteMatrix
    ) -> tuple[Literal["levels", "unit"], list[tuple[float, ...]]]:
        """Return the axis and the grid of each scale of ``votes``."""
        setting = self.thresholds
        if setting is None or isinstance(setting, Thresholds):
            rule = Thresholds() if setting is None else setting
            axis = rule.axis_for(votes)
            return axis, [rule.grid_of(scale, axis) for scale in votes.scales]
        grid = cast(tuple[float, ...], setting)
        if not votes.item_scales.is_uniform:
            raise InvalidParameterError(
                f"thresholds {grid!r} are the grid of one scale, but the items use "
                f"{votes.item_scales.n_scales} scales",
                hint="Give one grid per scale: thresholds=Thresholds(grids={scale: grid, ...}).",
            )
        _check_thresholds(grid, votes.n_levels)
        return "levels", [grid]

    def _vote_shares(self, params: OrdinalProbitParameters) -> FloatArray:
        """Return the probability of each level of each scale, floored at 1e-12.

        Shape (Q, L, C, K); the levels past those of a scale are 0.
        """
        bounds, possible = params._grid
        probabilities = level_probabilities_by_scale(
            params.latent_mean, params.latent_spread, bounds
        )
        shares: FloatArray = np.where(possible, np.maximum(probabilities, _MIN_VOTE_SHARE), 0.0)
        return shares


def _check_thresholds(thresholds: tuple[float, ...], n_levels: int) -> None:
    if len(thresholds) != n_levels - 1 or len(thresholds) < 1:
        raise InvalidParameterError(
            f"a scale of {n_levels} scores has {n_levels - 1} thresholds, "
            f"got {len(thresholds)}: {thresholds!r}"
        )
    if not all(math.isfinite(threshold) for threshold in thresholds) or any(
        low >= high for low, high in pairwise(thresholds)
    ):
        raise InvalidParameterError(
            f"thresholds are finite and strictly increasing, got {thresholds!r}"
        )
