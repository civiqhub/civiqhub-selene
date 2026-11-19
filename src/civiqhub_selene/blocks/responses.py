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

A response model is a frozen dataclass: its settings enter the provenance of the
estimator that carries it as they are.
"""

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol, TypeVar

import numpy as np

from civiqhub_selene.blocks._arrays import FloatArray
from civiqhub_selene.blocks._numerics import fit_latent_normal, level_probabilities
from civiqhub_selene.blocks.coclusters import CoclusterTable, state_log_probabilities
from civiqhub_selene.blocks.data import OrdinalScale, VoteMatrix
from civiqhub_selene.errors import InvalidParameterError

__all__ = [
    "Categorical",
    "CategoricalParameters",
    "OrdinalProbit",
    "OrdinalProbitParameters",
    "ResponseModel",
]

ParametersT = TypeVar("ParametersT")

_EMPTY_COCLUSTER = 1e-12
_MIN_VOTE_SHARE = 1e-12


class ResponseModel(Protocol[ParametersT]):
    """What a block model asks of the response model of its co-clusters.

    ``ParametersT`` is the type of the parameters of the model, for Q x L co-clusters.
    """

    def initial_parameters(
        self, shape: tuple[int, int], votes: VoteMatrix, rng: np.random.Generator
    ) -> ParametersT:
        """Return starting parameters for ``shape = (Q, L)`` co-clusters."""
        ...

    def log_probabilities(self, params: ParametersT) -> FloatArray:
        """Return the log-probability of each state in each co-cluster, shape (Q, L, K + 1)."""
        ...

    def update(self, counts: FloatArray, previous: ParametersT) -> ParametersT:
        """Return the parameters that maximise the expected likelihood.

        ``counts[q, l, s]`` is the number, integer or fractional, of cells of the
        co-cluster ``(q, l)`` in the state ``s``.
        """
        ...

    def complexity_penalty(self, shape: tuple[int, int], votes: VoteMatrix) -> float:
        """Return the penalty of the parameters in the integrated likelihood.

        Half a logarithm of the sample size per free parameter.
        """
        ...

    def to_table(
        self, params: ParametersT, counts: FloatArray, scale: OrdinalScale
    ) -> CoclusterTable:
        """Return the readable table of the co-clusters."""
        ...


def _vote_rate(counts: FloatArray, previous: FloatArray, min_probability: float) -> FloatArray:
    """Return the vote rate of each co-cluster, bounded away from 0 and 1.

    ``votes / cells`` where the co-cluster has cells (a sum above 1e-12), the previous
    value elsewhere; the whole bounded between ``min_probability`` and
    ``1 - min_probability``.
    """
    n_cells = counts.sum(axis=-1)
    n_votes = counts[..., 1:].sum(axis=-1)
    rate = np.divide(n_votes, n_cells, out=previous.copy(), where=n_cells > _EMPTY_COCLUSTER)
    bounded: FloatArray = np.clip(rate, min_probability, 1.0 - min_probability)
    return bounded


def _overall_vote_rate(votes: VoteMatrix, min_probability: float) -> float:
    """Return the votes over the shown cells of the matrix, bounded away from 0 and 1."""
    rate = votes.n_votes / max(votes.n_shown, 1)
    return min(max(rate, min_probability), 1.0 - min_probability)


@dataclass(frozen=True, eq=False)
class CategoricalParameters:
    """Parameters of the :class:`Categorical` response model.

    Attributes:
        vote_rate: Shape (Q, L): probability that a shown cell carries a vote.
        vote_shares: Shape (Q, L, K): distribution of the votes over the scores.
    """

    vote_rate: FloatArray
    vote_shares: FloatArray


@dataclass(frozen=True)
class Categorical:
    """A vote rate per co-cluster, then a free distribution over the K scores.

    Definition. For a shown cell of the co-cluster ``(q, l)``::

        P(no vote) = 1 - vote_rate[q, l]
        P(vote at rank k) = vote_rate[q, l] * vote_shares[q, l, k]

    with K - 1 free parameters per co-cluster for the distribution and no constraint on
    its shape. The update is the maximum of the expected likelihood: the vote rate is
    the share of cells that carry a vote, and the distribution is the share of votes at
    each rank, after ``smoothing`` is added to each count.

    It is the simple reference against which the cost and the benefit of the ordinal
    probit model are judged.

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
        concentration 5 on each of the K scores.
        """
        rate = _overall_vote_rate(votes, self.min_probability)
        vote_shares: FloatArray = rng.dirichlet(np.full(votes.n_levels, 5.0), size=shape)
        return CategoricalParameters(np.full(shape, rate), vote_shares)

    def log_probabilities(self, params: CategoricalParameters) -> FloatArray:
        """Return the log-probability of each state in each co-cluster, shape (Q, L, K + 1)."""
        return state_log_probabilities(params.vote_rate, params.vote_shares)

    def update(self, counts: FloatArray, previous: CategoricalParameters) -> CategoricalParameters:
        """Return the vote rates and the distributions that the counts call for.

        A co-cluster without any cell keeps its previous vote rate. A co-cluster
        without any vote keeps its previous distribution when ``smoothing`` is 0.
        """
        smoothed = counts[..., 1:] + self.smoothing
        total = smoothed.sum(axis=-1, keepdims=True)
        vote_shares = np.divide(
            smoothed, total, out=previous.vote_shares.copy(), where=total > _EMPTY_COCLUSTER
        )
        vote_rate = _vote_rate(counts, previous.vote_rate, self.min_probability)
        return CategoricalParameters(vote_rate, vote_shares)

    def complexity_penalty(self, shape: tuple[int, int], votes: VoteMatrix) -> float:
        """Return half a logarithm of the sample size per free parameter.

        ``0.5 * Q * L * (K - 1) * log(max(n_votes, 1))`` for the distributions, plus
        ``0.5 * Q * L * log(max(n_shown, 1))`` for the vote rates.
        """
        n_groups, n_blocks = shape
        n_coclusters = n_groups * n_blocks
        distributions = 0.5 * n_coclusters * (votes.n_levels - 1) * math.log(max(votes.n_votes, 1))
        vote_rates = 0.5 * n_coclusters * math.log(max(votes.n_shown, 1))
        return distributions + vote_rates

    def to_table(
        self, params: CategoricalParameters, counts: FloatArray, scale: OrdinalScale
    ) -> CoclusterTable:
        """Return the table of the co-clusters, with the number of votes of the counts."""
        return CoclusterTable(
            scale, params.vote_rate, params.vote_shares, n_votes=counts[..., 1:].sum(axis=-1)
        )


@dataclass(frozen=True, eq=False)
class OrdinalProbitParameters:
    """Parameters of the :class:`OrdinalProbit` response model.

    Attributes:
        vote_rate: Shape (Q, L): probability that a shown cell carries a vote.
        latent_mean: Shape (Q, L): mean of the latent opinion, on the axis of levels.
        latent_spread: Shape (Q, L): standard deviation of the latent opinion.
        thresholds: The K - 1 thresholds between levels, fixed during the computation.
    """

    vote_rate: FloatArray
    latent_mean: FloatArray
    latent_spread: FloatArray
    thresholds: FloatArray


@dataclass(frozen=True)
class OrdinalProbit:
    """A vote rate per co-cluster, then a latent normal opinion cut into K levels.

    Definition. In a co-cluster, a shown cell carries a vote with the probability
    ``vote_rate``. The vote comes from a hidden continuous opinion, normal with mean
    ``latent_mean`` and standard deviation ``latent_spread``, which fixed thresholds
    ``t_1 < ... < t_{K-1}`` cut into K levels, with ``t_0 = -inf`` and ``t_K = +inf``::

        P(no vote) = 1 - vote_rate
        P(vote at rank k) = vote_rate * p_k
        p_k = cdf((t_k - latent_mean) / latent_spread)
              - cdf((t_{k-1} - latent_mean) / latent_spread)

    where ``cdf`` is the standard normal distribution function. Two numbers then sum
    up how a group answers a block: where the opinion stands and how spread it is.

    The thresholds default to ``k + 1/2`` for ``k`` from 1 to K - 1. They are fixed,
    not estimated: ``latent_mean`` then reads directly on the axis of levels, from 1
    to K. The mean and the spread are estimated by expectation-maximisation on the
    hidden opinion (see ``fit_latent_normal``).

    Reference: Corneli, Bouveyron, Latouche (2020), Co-clustering of ordinal data via
    latent continuous random variables and not missing at random entries.

    Attributes:
        thresholds: The K - 1 thresholds, strictly increasing, or ``None`` for the
            default ones.
        min_spread: Lower bound of ``latent_spread``, strictly positive. It keeps a
            unanimous co-cluster from sending its spread to zero. Defaults to 0.3.
        min_probability: Bound that keeps a vote rate away from 0 and from 1, strictly
            between 0 and 0.5. Defaults to 1e-6.
        max_inner_iterations: Largest number of expectation-maximisation steps on the
            hidden opinion at each update, at least 1. Defaults to 10.

    Raises:
        InvalidParameterError: If a setting is outside its range.
    """

    thresholds: tuple[float, ...] | None = None
    min_spread: float = 0.3
    min_probability: float = 1e-6
    max_inner_iterations: int = 10

    def __post_init__(self) -> None:
        if self.thresholds is not None:
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
        matrix, bounded. Each mean is the middle of the axis of levels, ``(K + 1) / 2``,
        plus a normal draw of standard deviation 0.5; each spread is 1.

        Raises:
            InvalidParameterError: If the thresholds given are not K - 1 values.
        """
        n_levels = votes.n_levels
        if self.thresholds is None:
            thresholds = tuple(level + 0.5 for level in range(1, n_levels))
        else:
            thresholds = self.thresholds
            _check_thresholds(thresholds, n_levels)
        rate = _overall_vote_rate(votes, self.min_probability)
        latent_mean: FloatArray = (n_levels + 1) / 2 + rng.normal(0.0, 0.5, size=shape)
        return OrdinalProbitParameters(
            np.full(shape, rate), latent_mean, np.ones(shape), np.array(thresholds)
        )

    def log_probabilities(self, params: OrdinalProbitParameters) -> FloatArray:
        """Return the log-probability of each state in each co-cluster, shape (Q, L, K + 1)."""
        return state_log_probabilities(params.vote_rate, self._vote_shares(params))

    def update(
        self, counts: FloatArray, previous: OrdinalProbitParameters
    ) -> OrdinalProbitParameters:
        """Return the vote rates and the latent opinions that the counts call for.

        The mean and the spread restart from the previous ones and take at most
        ``max_inner_iterations`` steps. A co-cluster without any cell keeps its
        previous vote rate; a co-cluster without any vote keeps its previous opinion.
        """
        latent_mean, latent_spread = fit_latent_normal(
            counts[..., 1:],
            previous.latent_mean,
            previous.latent_spread,
            previous.thresholds,
            min_spread=self.min_spread,
            max_iterations=self.max_inner_iterations,
        )
        vote_rate = _vote_rate(counts, previous.vote_rate, self.min_probability)
        return OrdinalProbitParameters(vote_rate, latent_mean, latent_spread, previous.thresholds)

    def complexity_penalty(self, shape: tuple[int, int], votes: VoteMatrix) -> float:
        """Return half a logarithm of the sample size per free parameter.

        Two parameters of opinion and one vote rate per co-cluster:
        ``Q * L * log(max(n_votes, 1)) + 0.5 * Q * L * log(max(n_shown, 1))``.
        """
        n_groups, n_blocks = shape
        n_coclusters = n_groups * n_blocks
        opinions = n_coclusters * math.log(max(votes.n_votes, 1))
        vote_rates = 0.5 * n_coclusters * math.log(max(votes.n_shown, 1))
        return opinions + vote_rates

    def to_table(
        self, params: OrdinalProbitParameters, counts: FloatArray, scale: OrdinalScale
    ) -> CoclusterTable:
        """Return the table of the co-clusters, with their latent mean and spread."""
        return CoclusterTable(
            scale,
            params.vote_rate,
            self._vote_shares(params),
            n_votes=counts[..., 1:].sum(axis=-1),
            latent_mean=params.latent_mean,
            latent_spread=params.latent_spread,
        )

    def _vote_shares(self, params: OrdinalProbitParameters) -> FloatArray:
        """Return the probability of each level, floored at 1e-12."""
        shares: FloatArray = np.maximum(
            level_probabilities(params.latent_mean, params.latent_spread, params.thresholds),
            _MIN_VOTE_SHARE,
        )
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
