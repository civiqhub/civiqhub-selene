# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""The latent block model: co-clustering where each voter belongs to one group.

It arranges each voter in one group and each item in one block, so that the cells of a
co-cluster look alike. See :class:`LatentBlockModel` for the definition.
"""

import math
from dataclasses import dataclass
from typing import Any, final

import numpy as np

from civiqhub_selene.blocks._arrays import FloatArray, IntArray
from civiqhub_selene.blocks._numerics import History, count_states, logsumexp
from civiqhub_selene.blocks.base import BlockModel
from civiqhub_selene.blocks.data import VoteMatrix
from civiqhub_selene.blocks.results import SingleCoclustering

__all__ = ["LatentBlockModel"]

_MIN_SHARE = 1e-300


@dataclass(frozen=True, eq=False)
class _HardState:
    """One state of the algorithm: an arrangement, its shares and its parameters.

    Attributes:
        group_index: Shape (M,): the group of each voter.
        block_index: Shape (P,): the block of each item.
        group_shares: Shape (Q,): the share of voters in each group.
        block_shares: Shape (L,): the share of items in each block.
        parameters: The parameters of the response model.
        objective: The classified log-likelihood of this state.
    """

    group_index: IntArray
    block_index: IntArray
    group_shares: FloatArray
    block_shares: FloatArray
    parameters: Any
    objective: float


def _voter_scores(
    states: FloatArray, logs: FloatArray, block_index: IntArray, group_shares: FloatArray
) -> FloatArray:
    """Return the score of each voter for each group, the blocks being fixed. Shape (M, Q)."""
    by_cell = np.einsum("ijs,qjs->iq", states, logs[:, block_index, :])
    scores: FloatArray = by_cell + np.log(np.maximum(group_shares, _MIN_SHARE))
    return scores


def _item_scores(
    states: FloatArray,
    logs: FloatArray,
    group_index: IntArray,
    block_shares: FloatArray,
    voter_weights: FloatArray | None = None,
) -> FloatArray:
    """Return the score of each item for each block, the groups being fixed. Shape (P, L)."""
    scored = logs[group_index]
    if voter_weights is None:
        by_cell = np.einsum("ijs,ils->jl", states, scored)
    else:
        by_cell = np.einsum("ijs,ils,i->jl", states, scored, voter_weights)
    scores: FloatArray = by_cell + np.log(np.maximum(block_shares, _MIN_SHARE))
    return scores


def _sweep(index: IntArray, scores: FloatArray) -> IntArray:
    """Move each element to its best cluster, in order, without emptying a cluster.

    The elements whose best cluster differs from their current one are visited in
    increasing order. An element moves if its current cluster has more than one member
    and if the score of its best cluster is strictly larger. Sizes are updated as
    elements move: the last member of a cluster never leaves it.
    """
    moved: IntArray = index.copy()
    sizes = np.bincount(moved, minlength=scores.shape[1])
    best = scores.argmax(axis=1)
    for element in np.flatnonzero(best != moved):
        current, target = moved[element], best[element]
        if sizes[current] > 1 and scores[element, target] > scores[element, current]:
            sizes[current] -= 1
            sizes[target] += 1
            moved[element] = target
    return moved


def _softmax(scores: FloatArray) -> FloatArray:
    memberships: FloatArray = np.exp(scores - logsumexp(scores, axis=1)[:, np.newaxis])
    return memberships


@final
@dataclass(frozen=True, kw_only=True)
class LatentBlockModel(BlockModel[_HardState, SingleCoclustering]):
    """Co-clustering where each voter belongs to one group and each item to one block.

    Definition. With M voters, P items, Q groups and L blocks, the voter ``i`` has a
    group ``R[i]`` and the item ``j`` a block ``C[j]``. Given the arrangement, the
    shown cells are independent, and the state ``s`` of the cell ``(i, j)`` (shown
    without a vote, or a vote at one rank) follows the response model of its
    co-cluster, with probability ``p[R[i], C[j], s]``. Groups have shares
    ``group_shares`` and blocks have shares ``block_shares``. A cell that is not shown
    carries no information and enters nowhere.

    The model is estimated by classification expectation-maximisation: it maximises
    the classified log-likelihood, where ``S`` is the indicator of the state of each
    shown cell::

        objective = sum_ij sum_s S[i, j, s] * log p[R[i], C[j], s]
                    + sum_i log group_shares[R[i]] + sum_j log block_shares[C[j]]

    With optional mean-one survey weights ``w[i]``, this becomes the weighted
    pseudo-likelihood of Skinner (1989) and Pfeffermann (1993)::

        objective = sum_i w[i] * (sum_j sum_s S[i, j, s] * log p[R[i], C[j], s]
                                  + log group_shares[R[i]])
                    + sum_j log block_shares[C[j]]

    When every weight is 1 this is the unweighted objective. The block-share term is
    not multiplied by ``w[i]``: with a single block it is zero, and the response
    parameters then match those of a row duplication. The spectral start and the
    reported ``group_memberships`` do not use the survey weights.

    Each start begins from the arrangement its ``initialization`` gives: random by
    default, or the one of regularised spectral co-clustering with ``SpectralStart()``.
    Each iteration updates the parameters of the response model and the shares from
    the current arrangement, then moves the voters to their best group, the blocks
    being fixed, then the items to their best block. No group and no block ever
    empties. Several random starts are run and the one with the largest objective is
    kept.

    The value carries the integrated completed likelihood, which compares shapes::

        integrated_likelihood = objective - complexity_penalty of the response model
                                - 0.5 * (Q - 1) * log(M) - 0.5 * (L - 1) * log(P)

    Using that same penalty on the weighted objective is a Hypothesis: a weighted ICL
    is not a settled model-selection criterion. A shown cell of item ``j`` has
    ``K_j + 1`` modalities; this estimator currently reads one shared scale of ``K``
    scores. Distinct observation scales per item are SEL-8.2 (ItemScales).

    The memberships of the value are soft readings of the final arrangement: the
    scores of each voter for each group, turned into shares that sum to 1.

    References: Govaert and Nadif (2003), Clustering with block mixture models,
    Pattern Recognition 36(2); Biernacki, Celeux and Govaert (2000), Assessing a
    mixture model for clustering with the integrated completed likelihood, IEEE
    Transactions on Pattern Analysis and Machine Intelligence 22(7); Corneli,
    Bouveyron and Latouche (2020) for the ordinal response model; Skinner (1989)
    and Pfeffermann (1993) for the weighted pseudo-likelihood.

    The settings are those of :class:`~civiqhub_selene.blocks.base.BlockModel`.
    """

    name = "latent_block_model"
    version = "0.1.0"
    reference = "Govaert and Nadif 2003; Corneli, Bouveyron, Latouche 2020"
    membership = "single"
    meaning = "each voter belongs to one group"
    default_criterion = "integrated_likelihood"

    def _start(self, votes: VoteMatrix, rng: np.random.Generator) -> _HardState:
        shape = (self.n_groups, self.n_blocks)
        group_index, block_index = self.initialization.arrangement(votes, shape, rng)
        return _HardState(
            group_index,
            block_index,
            np.full(self.n_groups, 1.0 / self.n_groups),
            np.full(self.n_blocks, 1.0 / self.n_blocks),
            self.response_model.initial_parameters(shape, votes, rng),
            -math.inf,
        )

    def _step(self, votes: VoteMatrix, state: _HardState) -> _HardState:
        states = votes._states()
        n_voters, n_items = votes.shape
        shape = (self.n_groups, self.n_blocks)
        weights = votes._likelihood_weights
        counts = count_states(
            states, state.group_index, state.block_index, shape, voter_weights=weights
        )
        parameters = self.response_model.update(counts, state.parameters)
        group_mass = np.bincount(state.group_index, weights=weights, minlength=self.n_groups)
        group_shares = group_mass / n_voters
        block_shares = np.bincount(state.block_index, minlength=self.n_blocks) / n_items
        logs = self.response_model.log_probabilities(parameters)
        group_index = _sweep(
            state.group_index, _voter_scores(states, logs, state.block_index, group_shares)
        )
        block_index = _sweep(
            state.block_index, _item_scores(states, logs, group_index, block_shares, weights)
        )
        cell_logs = logs[group_index][:, block_index, :]
        group_logs = np.log(np.maximum(group_shares[group_index], _MIN_SHARE))
        if weights is None:
            response_score = float(np.einsum("ijs,ijs->", states, cell_logs))
            group_score = float(group_logs.sum())
        else:
            response_score = float(np.einsum("ijs,ijs,i->", states, cell_logs, weights))
            group_score = float(group_logs @ weights)
        objective = (
            response_score
            + group_score
            + float(np.log(np.maximum(block_shares[block_index], _MIN_SHARE)).sum())
        )
        return _HardState(
            group_index, block_index, group_shares, block_shares, parameters, objective
        )

    def _finish(
        self, votes: VoteMatrix, state: _HardState, history: History, computation: str
    ) -> SingleCoclustering:
        states = votes._states()
        n_voters, n_items = votes.shape
        shape = (self.n_groups, self.n_blocks)
        logs = self.response_model.log_probabilities(state.parameters)
        voter_scores = _voter_scores(states, logs, state.block_index, state.group_shares)
        item_scores = _item_scores(
            states, logs, state.group_index, state.block_shares, votes._likelihood_weights
        )
        counts = count_states(states, state.group_index, state.block_index, shape)
        integrated_likelihood = (
            state.objective
            - self.response_model.complexity_penalty(shape, votes)
            - 0.5 * (self.n_groups - 1) * math.log(max(n_voters, 1))
            - 0.5 * (self.n_blocks - 1) * math.log(max(n_items, 1))
        )
        return SingleCoclustering(
            computation=computation,
            voters=votes.voters,
            items=votes.items,
            scale=votes.scale,
            group_index=state.group_index,
            block_index=state.block_index,
            group_memberships=_softmax(voter_scores),
            block_memberships=_softmax(item_scores),
            coclusters=self.response_model.to_table(state.parameters, counts, votes.scale),
            objective=state.objective,
            history=history,
            integrated_likelihood=integrated_likelihood,
        )
