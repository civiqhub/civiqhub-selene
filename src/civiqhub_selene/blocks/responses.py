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
from typing import Protocol, TypeVar

import numpy as np

from civiqhub_selene.blocks._arrays import FloatArray
from civiqhub_selene.blocks.coclusters import CoclusterTable, state_log_probabilities
from civiqhub_selene.blocks.data import OrdinalScale, VoteMatrix
from civiqhub_selene.errors import InvalidParameterError

__all__ = ["Categorical", "CategoricalParameters", "ResponseModel"]

ParametersT = TypeVar("ParametersT")

_EMPTY_COCLUSTER = 1e-12


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
        rate = votes.n_votes / max(votes.n_shown, 1)
        bounded = min(max(rate, self.min_probability), 1.0 - self.min_probability)
        vote_shares: FloatArray = rng.dirichlet(np.full(votes.n_levels, 5.0), size=shape)
        return CategoricalParameters(np.full(shape, bounded), vote_shares)

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
