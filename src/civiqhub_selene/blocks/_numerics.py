# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Pure functions on arrays shared by the co-clustering modules.

No function of this module draws a random number.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike

from civiqhub_selene.blocks._arrays import BoolArray, FloatArray, IntArray

__all__ = [
    "History",
    "columns_by_scale",
    "count_states",
    "count_states_by_scale",
    "fit_latent_normal",
    "fit_latent_normal_by_scale",
    "interval_mass",
    "iterate",
    "level_probabilities",
    "level_probabilities_by_scale",
    "logsumexp",
    "norm_cdf",
    "norm_pdf",
    "one_hot",
    "scale_bounds",
    "truncated_moments",
    "truncated_moments_by_scale",
]

StateT = TypeVar("StateT")

_MIN_MASS = 1e-300
_NO_VOTE = 1e-12
_erfc = np.vectorize(math.erfc, otypes=[np.float64])


def one_hot(labels: IntArray, n: int) -> FloatArray:
    """Return the indicator matrix of ``labels``, of shape (len(labels), n)."""
    indicator: FloatArray = np.zeros((labels.shape[0], n), dtype=np.float64)
    indicator[np.arange(labels.shape[0]), labels] = 1.0
    return indicator


def count_states(
    states: FloatArray,
    rows: IntArray,
    cols: IntArray,
    shape: tuple[int, int],
    *,
    voter_weights: FloatArray | None = None,
) -> FloatArray:
    """Count the cells of each co-cluster in each state, for hard labels.

    ``counts[q, l, s]`` is the number of cells of the co-cluster ``(q, l)`` in the
    state ``s``. It is the sufficient statistic of every block model.

    Args:
        states: Indicator of the state of each cell, of shape (M, P, S).
        rows: Label of each row, of shape (M,), in ``0..Q-1``.
        cols: Label of each column, of shape (P,), in ``0..L-1``.
        shape: ``(Q, L)``.
        voter_weights: Optional multiplicative weight of each row.

    Returns:
        An array of shape (Q, L, S).
    """
    n_groups, n_blocks = shape
    groups = one_hot(rows, n_groups)
    if voter_weights is not None:
        groups *= voter_weights[:, np.newaxis]
    counts: FloatArray = np.einsum("iq,jl,ijs->qls", groups, one_hot(cols, n_blocks), states)
    return counts


def columns_by_scale(scale_index: IntArray, n_scales: int) -> tuple[IntArray | None, ...]:
    """Return the columns of each scale, in increasing order.

    With one scale, the only entry is ``None``: every column, read without a copy, so
    that a matrix of one scale is computed exactly as before items could have several.
    """
    if n_scales == 1:
        return (None,)
    return tuple(np.flatnonzero(scale_index == scale) for scale in range(n_scales))


def count_states_by_scale(
    states: FloatArray,
    rows: IntArray,
    cols: IntArray,
    shape: tuple[int, int],
    scale_index: IntArray,
    n_scales: int,
    *,
    voter_weights: FloatArray | None = None,
) -> FloatArray:
    """Count the cells of each co-cluster in each state, scale by scale, for hard labels.

    ``counts[q, l, c, s]`` is the number of cells of the items of the scale ``c`` in the
    co-cluster ``(q, l)`` in the state ``s``. It is the sufficient statistic of every
    block model when items have their own scales; with one scale it is
    :func:`count_states` with an axis of length 1.

    Args:
        states: Indicator of the state of each cell, of shape (M, P, S).
        rows: Label of each row, of shape (M,), in ``0..Q-1``.
        cols: Label of each column, of shape (P,), in ``0..L-1``.
        shape: ``(Q, L)``.
        scale_index: The scale of each column, of shape (P,), in ``0..C-1``.
        n_scales: ``C``.
        voter_weights: Optional multiplicative weight of each row.

    Returns:
        An array of shape (Q, L, C, S).
    """
    n_groups, n_blocks = shape
    counts: FloatArray = np.zeros((n_groups, n_blocks, n_scales, states.shape[-1]))
    for scale, columns in enumerate(columns_by_scale(scale_index, n_scales)):
        if columns is None:
            counts[:, :, scale, :] = count_states(
                states, rows, cols, shape, voter_weights=voter_weights
            )
        else:
            counts[:, :, scale, :] = count_states(
                states[:, columns, :], rows, cols[columns], shape, voter_weights=voter_weights
            )
    return counts


def logsumexp(a: FloatArray, axis: int) -> FloatArray:
    """Return ``log(sum(exp(a)))`` along ``axis``, computed without overflow.

    A slice that holds only ``-inf`` gives ``-inf``.
    """
    highest = np.max(a, axis=axis, keepdims=True)
    shift = np.where(np.isfinite(highest), highest, 0.0)
    with np.errstate(divide="ignore"):
        total: FloatArray = np.log(np.sum(np.exp(a - shift), axis=axis))
    total += np.squeeze(shift, axis=axis)
    return total


def norm_cdf(x: ArrayLike) -> FloatArray:
    """Return the standard normal distribution function, accurate in the lower tail.

    ``0.5 * erfc(-x / sqrt(2))``.
    """
    values: FloatArray = 0.5 * _erfc(-np.asarray(x, dtype=np.float64) / math.sqrt(2.0))
    return values


def norm_pdf(x: ArrayLike) -> FloatArray:
    """Return the standard normal density. It is zero at infinity."""
    points = np.asarray(x, dtype=np.float64)
    density: FloatArray = np.exp(-0.5 * points * points) / math.sqrt(2.0 * math.pi)
    return density


def interval_mass(z: FloatArray) -> FloatArray:
    """Return the normal mass of each interval between successive standardised bounds.

    ``z`` has shape (..., K + 1); the result has shape (..., K) and holds
    ``cdf(z[..., k + 1]) - cdf(z[..., k])``, computed without numerical cancellation:
    when the lower bound is positive, the difference is taken on the survival
    functions, ``cdf(-lower) - cdf(-upper)``.
    """
    lower, upper = z[..., :-1], z[..., 1:]
    mass: FloatArray = np.where(
        lower > 0,
        norm_cdf(-lower) - norm_cdf(-upper),
        norm_cdf(upper) - norm_cdf(lower),
    )
    return mass


def _standardised_bounds(mean: ArrayLike, spread: ArrayLike, thresholds: ArrayLike) -> FloatArray:
    """Return the thresholds, with their two infinite ends, centred and reduced."""
    means = np.asarray(mean, dtype=np.float64)[..., np.newaxis]
    spreads = np.asarray(spread, dtype=np.float64)[..., np.newaxis]
    bounds = np.concatenate([[-np.inf], np.asarray(thresholds, dtype=np.float64), [np.inf]])
    z: FloatArray = (bounds - means) / spreads
    return z


def level_probabilities(mean: ArrayLike, spread: ArrayLike, thresholds: ArrayLike) -> FloatArray:
    """Return the probability of each level under a latent normal opinion cut by thresholds.

    With thresholds ``t_1 < ... < t_{K-1}``, ``t_0 = -inf`` and ``t_K = +inf``, the
    probability of the level ``k`` is
    ``cdf((t_k - mean) / spread) - cdf((t_{k-1} - mean) / spread)``.

    Args:
        mean: Mean of the latent opinion, of any shape (...).
        spread: Standard deviation of the latent opinion, of the same shape.
        thresholds: The K - 1 thresholds, strictly increasing.

    Returns:
        An array of shape (..., K), summing to 1 along the last axis.
    """
    return interval_mass(_standardised_bounds(mean, spread, thresholds))


def truncated_moments(
    mean: ArrayLike, spread: ArrayLike, thresholds: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Return the first two moments of the latent normal opinion truncated to each level.

    With ``z`` the standardised bounds, ``pdf`` the density,
    ``mass = max(interval_mass(z), 1e-300)``, ``r = diff(pdf(z)) / mass`` and
    ``s = diff(z * pdf(z)) / mass`` (the product is 0 at infinity)::

        m1 = mean - spread * r                      bounded by the two thresholds of the level
        m2 = max(mean**2 - 2 * spread * mean * r - spread**2 * (s - 1), m1**2)

    Returns:
        ``(m1, m2)``, each of shape (..., K).
    """
    means = np.asarray(mean, dtype=np.float64)[..., np.newaxis]
    spreads = np.asarray(spread, dtype=np.float64)[..., np.newaxis]
    bounds = np.concatenate([[-np.inf], np.asarray(thresholds, dtype=np.float64), [np.inf]])
    z = (bounds - means) / spreads
    density = norm_pdf(z)
    mass = np.maximum(interval_mass(z), _MIN_MASS)
    r = np.diff(density, axis=-1) / mass
    s = np.diff(np.where(np.isfinite(z), z, 0.0) * density, axis=-1) / mass
    first: FloatArray = np.clip(means - spreads * r, bounds[:-1], bounds[1:])
    second: FloatArray = np.maximum(
        means**2 - 2.0 * spreads * means * r - spreads**2 * (s - 1.0), first**2
    )
    return first, second


def scale_bounds(thresholds: Sequence[ArrayLike]) -> tuple[FloatArray, BoolArray]:
    """Return the bounds of the levels of each scale, padded to the longest one.

    For C grids of ``K_c - 1`` thresholds each, and K the levels of the longest scale:

    - ``bounds`` has the shape (C, K + 1): ``-inf``, the thresholds of the scale, then
      ``+inf`` up to the end, so that the levels past ``K_c`` are empty intervals;
    - ``possible`` has the shape (C, K): the level ``k`` exists on the scale ``c``.
    """
    grids = [np.asarray(grid, dtype=np.float64).reshape(-1) for grid in thresholds]
    n_levels = max(grid.shape[0] for grid in grids) + 1
    bounds: FloatArray = np.full((len(grids), n_levels + 1), np.inf)
    bounds[:, 0] = -np.inf
    possible: BoolArray = np.zeros((len(grids), n_levels), dtype=np.bool_)
    for scale, grid in enumerate(grids):
        bounds[scale, 1 : grid.shape[0] + 1] = grid
        possible[scale, : grid.shape[0] + 1] = True
    return bounds, possible


def _standardised_bounds_by_scale(
    mean: ArrayLike, spread: ArrayLike, bounds: FloatArray
) -> FloatArray:
    """Return the bounds of each scale, centred and reduced: shape (..., C, K + 1)."""
    means = np.asarray(mean, dtype=np.float64)[..., np.newaxis, np.newaxis]
    spreads = np.asarray(spread, dtype=np.float64)[..., np.newaxis, np.newaxis]
    z: FloatArray = (bounds - means) / spreads
    return z


def level_probabilities_by_scale(
    mean: ArrayLike, spread: ArrayLike, bounds: FloatArray
) -> FloatArray:
    """Return the probability of each level of each scale under a latent normal opinion.

    The same opinion, of mean ``mean`` and standard deviation ``spread``, is cut by the
    grid of each scale (see :func:`scale_bounds`). A level past the levels of a scale
    is an empty interval, of probability 0.

    Returns:
        An array of shape (..., C, K); on each scale, the probabilities of its levels
        sum to 1.
    """
    return interval_mass(_standardised_bounds_by_scale(mean, spread, bounds))


def truncated_moments_by_scale(
    mean: ArrayLike, spread: ArrayLike, bounds: FloatArray, possible: BoolArray
) -> tuple[FloatArray, FloatArray]:
    """Return the moments of the latent opinion truncated to each level of each scale.

    The formulas of :func:`truncated_moments`, on the grid of each scale. The moments
    of a level that does not exist on a scale are 0, so that a count of 0 times them
    adds nothing.

    Returns:
        ``(m1, m2)``, each of shape (..., C, K).
    """
    means = np.asarray(mean, dtype=np.float64)[..., np.newaxis, np.newaxis]
    spreads = np.asarray(spread, dtype=np.float64)[..., np.newaxis, np.newaxis]
    z = (bounds - means) / spreads
    density = norm_pdf(z)
    mass = np.maximum(interval_mass(z), _MIN_MASS)
    r = np.diff(density, axis=-1) / mass
    s = np.diff(np.where(np.isfinite(z), z, 0.0) * density, axis=-1) / mass
    with np.errstate(invalid="ignore"):
        first: FloatArray = np.clip(means - spreads * r, bounds[..., :-1], bounds[..., 1:])
        second: FloatArray = np.maximum(
            means**2 - 2.0 * spreads * means * r - spreads**2 * (s - 1.0), first**2
        )
    return np.where(possible, first, 0.0), np.where(possible, second, 0.0)


@dataclass(frozen=True)
class History:
    """What an iterative computation went through.

    Attributes:
        objectives: The value noted after each step. The starting state is not in it.
        converged: Whether the loop stopped because the gap went under the tolerance.
    """

    objectives: tuple[float, ...]
    converged: bool

    @property
    def n_iterations(self) -> int:
        """The number of steps taken."""
        return len(self.objectives)


def iterate(
    step: Callable[[StateT], StateT],
    state: StateT,
    *,
    gap: Callable[[StateT, StateT], float],
    value: Callable[[StateT], float],
    tolerance: float,
    max_iterations: int,
) -> tuple[StateT, History]:
    """Repeat ``step`` until it stops making progress. The only convergence loop of the package.

    After each step, ``value(new state)`` is noted. The loop stops, converged, as soon
    as ``gap(old state, new state) < tolerance``; otherwise after ``max_iterations``
    steps, not converged.

    Returns:
        The last state and the history of the loop.
    """
    objectives: list[float] = []
    converged = False
    for _ in range(max_iterations):
        new_state = step(state)
        objectives.append(float(value(new_state)))
        progress = gap(state, new_state)
        state = new_state
        if progress < tolerance:
            converged = True
            break
    return state, History(tuple(objectives), converged)


def fit_latent_normal(
    counts: FloatArray,
    mean: FloatArray,
    spread: FloatArray,
    thresholds: ArrayLike,
    *,
    min_spread: float,
    tolerance: float = 1e-8,
    max_iterations: int,
) -> tuple[FloatArray, FloatArray]:
    """Estimate the mean and the spread of the latent opinion of every co-cluster at once.

    Expectation-maximisation on the hidden opinion, for votes on one scale. ``counts[...,
    k]`` is the number, integer or fractional, of votes at the level ``k + 1`` of the
    co-cluster, and ``n`` their sum. At each step:

    1. the truncated moments ``m1`` and ``m2`` are computed at the current parameters;
    2. ``mean = sum_k(counts_k * m1_k) / n``;
    3. ``spread = sqrt(max(sum_k(counts_k * m2_k) / n - mean**2, min_spread**2))``;
    4. a co-cluster without a vote (``n <= 1e-12``) keeps its parameters.

    The loop stops when the largest absolute change of a mean or a spread, over all
    co-clusters, goes under ``tolerance``, or after ``max_iterations`` steps. It is
    :func:`fit_latent_normal_by_scale` on one scale.

    Args:
        counts: Shape (..., K): votes of each co-cluster at each level.
        mean: Shape (...): starting means.
        spread: Shape (...): starting spreads.
        thresholds: The K - 1 thresholds, fixed.
        min_spread: Lower bound of a spread. It keeps a unanimous co-cluster from
            sending its spread to zero.
        tolerance: Change of the parameters under which the loop stops.
        max_iterations: Largest number of steps.

    Returns:
        ``(mean, spread)``, each of the shape of the starting values.
    """
    bounds, possible = scale_bounds([thresholds])
    return fit_latent_normal_by_scale(
        np.asarray(counts)[..., np.newaxis, :],
        mean,
        spread,
        bounds,
        possible,
        min_spread=min_spread,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def fit_latent_normal_by_scale(
    counts: FloatArray,
    mean: FloatArray,
    spread: FloatArray,
    bounds: FloatArray,
    possible: BoolArray,
    *,
    min_spread: float,
    tolerance: float = 1e-8,
    max_iterations: int,
    estimable: BoolArray | None = None,
    fixed_spread: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Estimate the latent opinion of every co-cluster from votes on several scales.

    Expectation-maximisation on the hidden opinion. Each co-cluster has one opinion,
    normal, of mean ``mean`` and standard deviation ``spread``; a vote on the scale
    ``c`` at its level ``k`` says that the opinion lies between two thresholds of the
    grid of that scale. ``counts[..., c, k]`` is the number, integer or fractional, of
    votes at the level ``k + 1`` of the scale ``c``, and ``n`` their sum over the
    levels and the scales. At each step:

    1. the truncated moments ``m1`` and ``m2`` are computed on the grid of each scale;
    2. ``mean = sum_c sum_k(counts_ck * m1_ck) / n``;
    3. ``spread = sqrt(max(sum_c sum_k(counts_ck * m2_ck) / n - mean**2, min_spread**2))``,
       or ``fixed_spread`` where ``estimable`` is false;
    4. a co-cluster without a vote (``n <= 1e-12``) keeps its parameters.

    The sums run over the levels first, then over the scales, so that one scale gives
    exactly the result of :func:`fit_latent_normal`.

    Args:
        counts: Shape (..., C, K): votes of each co-cluster at each level of each scale.
        mean: Shape (...): starting means.
        spread: Shape (...): starting spreads.
        bounds: Shape (C, K + 1), from :func:`scale_bounds`.
        possible: Shape (C, K), from :func:`scale_bounds`.
        min_spread: Lower bound of a spread.
        tolerance: Change of the parameters under which the loop stops.
        max_iterations: Largest number of steps.
        estimable: Shape (...): where the spread can be estimated. Votes on scales of
            two levels only say on which side of one threshold the opinion lies, which
            fixes ``(threshold - mean) / spread`` and not the two apart: there, the
            spread is held at ``fixed_spread`` and only the mean is estimated. ``None``
            estimates every spread.
        fixed_spread: The spread where ``estimable`` is false.

    Returns:
        ``(mean, spread)``, each of the shape of the starting values.
    """
    n_votes = counts.sum(axis=-1).sum(axis=-1)
    has_votes = n_votes > _NO_VOTE
    safe_n = np.where(has_votes, n_votes, 1.0)

    def step(state: tuple[FloatArray, FloatArray]) -> tuple[FloatArray, FloatArray]:
        current_mean, current_spread = state
        first, second = truncated_moments_by_scale(current_mean, current_spread, bounds, possible)
        new_mean = (counts * first).sum(axis=-1).sum(axis=-1) / safe_n
        variance = (counts * second).sum(axis=-1).sum(axis=-1) / safe_n - new_mean**2
        new_spread = np.sqrt(np.maximum(variance, min_spread**2))
        if estimable is not None:
            new_spread = np.where(estimable, new_spread, fixed_spread)
        return (
            np.where(has_votes, new_mean, current_mean),
            np.where(has_votes, new_spread, current_spread),
        )

    def change(old: tuple[FloatArray, FloatArray], new: tuple[FloatArray, FloatArray]) -> float:
        if old[0].size == 0:
            return 0.0
        return float(max(np.abs(new[0] - old[0]).max(), np.abs(new[1] - old[1]).max()))

    start = (np.asarray(mean, dtype=np.float64), np.asarray(spread, dtype=np.float64))
    (fitted_mean, fitted_spread), _ = iterate(
        step,
        start,
        gap=change,
        # The inner loop records no objective: only the change of the parameters stops it.
        value=lambda _state: 0.0,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return fitted_mean, fitted_spread
