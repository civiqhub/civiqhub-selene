# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""The mixed-membership block model: a voter can follow different groups on different blocks.

It describes each voter by a degree of membership to each group and each item by a
degree of membership to each block. A voter takes one role per block (the group it
follows on that block), and each cell takes one block. See
:class:`MixedMembershipBlockModel` for the definition.
"""

import itertools
import math
from dataclasses import dataclass
from typing import Any, final

import numpy as np

from civiqhub_selene.blocks._arrays import BoolArray, FloatArray, IntArray
from civiqhub_selene.blocks._numerics import History, count_states, logsumexp
from civiqhub_selene.blocks.base import BlockModel
from civiqhub_selene.blocks.data import VoteMatrix
from civiqhub_selene.blocks.results import MixedCoclustering
from civiqhub_selene.errors import InvalidParameterError

__all__ = ["MAX_PROFILES", "MixedMembershipBlockModel"]

MAX_PROFILES = 4096
"""Largest number of profiles ``n_groups ** n_blocks`` the exact expectation enumerates."""

_MIN_PROBABILITY = 1e-300
# Cells read at once by the expectation step: voters x items x profiles.
_CHUNK_CELLS = 2_000_000


@dataclass(frozen=True, eq=False)
class _Expectation:
    """What the expectation step returns at given parameters.

    Attributes:
        profile_probabilities: Shape (M, C): posterior probability of each profile.
        role_by_block: Shape (M, L, Q): posterior probability of each role on each block.
        counts: Shape (Q, L, K + 1): expected number of cells of each co-cluster in
            each state.
        item_mass: Shape (P, L): expected number of the shown cells of each item that
            take each block.
        log_likelihood: The observed log-likelihood.
    """

    profile_probabilities: FloatArray
    role_by_block: FloatArray
    counts: FloatArray
    item_mass: FloatArray
    log_likelihood: float


@dataclass(frozen=True, eq=False)
class _MixedState:
    """One state of the algorithm: memberships, parameters, and their expectation step.

    Attributes:
        voter_memberships: Shape (M, Q): the degree of membership of each voter.
        item_memberships: Shape (P, L): the degree of membership of each item.
        parameters: The parameters of the response model.
        expectation: The expectation step at these memberships and parameters.
        objective: The penalised log-likelihood at these memberships and parameters.
    """

    voter_memberships: FloatArray
    item_memberships: FloatArray
    parameters: Any
    expectation: _Expectation
    objective: float


def _profiles(n_groups: int, n_blocks: int) -> IntArray:
    """Every profile, the group followed on each block, in the order of itertools.product."""
    profiles: IntArray = np.array(
        list(itertools.product(range(n_groups), repeat=n_blocks)), dtype=np.int64
    ).reshape(n_groups**n_blocks, n_blocks)
    return profiles


def _smoothed(labels: IntArray, n_clusters: int, weight: float) -> FloatArray:
    """Return memberships with ``weight`` on the label and the rest shared by the others."""
    if n_clusters == 1:
        return np.ones((labels.shape[0], 1))
    memberships: FloatArray = np.full(
        (labels.shape[0], n_clusters), (1.0 - weight) / (n_clusters - 1)
    )
    memberships[np.arange(labels.shape[0]), labels] = weight
    return memberships


def _normalised(weights: FloatArray) -> FloatArray:
    """Divide each row by its sum; a row that sums to 0 becomes uniform."""
    totals = weights.sum(axis=1, keepdims=True)
    uniform = np.full_like(weights, 1.0 / weights.shape[1])
    rows: FloatArray = np.divide(weights, totals, out=uniform, where=totals > 0)
    return rows


def _expect(
    states: FloatArray,
    shown: BoolArray,
    voter_memberships: FloatArray,
    item_memberships: FloatArray,
    logs: FloatArray,
    profiles: IntArray,
    voter_weights: FloatArray | None = None,
) -> _Expectation:
    """The exact expectation step over the profiles.

    With ``F[q, l, s]`` the probability of the state ``s`` in the co-cluster ``(q, l)``
    and ``den[i, j, c] = sum_l item_memberships[j, l] * F[c_l, l, s_ij]`` the
    probability of the cell ``(i, j)`` under the profile ``c``, the posterior of a
    profile is proportional to ``prod_l voter_memberships[i, c_l] * prod_j den[i, j, c]``
    over the shown cells, and the cell takes the block ``l`` with the probability
    ``item_memberships[j, l] * F[c_l, l, s_ij] / den[i, j, c]`` under that profile.
    Voter weights multiply contributions to the shared counts and likelihood.
    """
    n_voters, n_items, n_states = states.shape
    n_profiles, n_blocks = profiles.shape
    n_groups = voter_memberships.shape[1]
    probabilities = np.exp(logs)
    # follows[c, l, q] = 1 when the profile c follows the group q on the block l.
    follows = np.zeros((n_profiles, n_blocks, n_groups))
    follows[np.arange(n_profiles)[:, np.newaxis], np.arange(n_blocks)[np.newaxis, :], profiles] = 1
    by_profile = probabilities[profiles, np.arange(n_blocks)[np.newaxis, :], :]  # (C, L, S)
    cell_probability = np.einsum("jl,cls->jsc", item_memberships, by_profile)  # (P, S, C)
    prior = np.einsum(
        "iq,clq->ic", np.log(np.maximum(voter_memberships, _MIN_PROBABILITY)), follows
    )
    profile_probabilities = np.zeros((n_voters, n_profiles))
    gathered = np.zeros((n_items, n_states, n_profiles))
    log_likelihood = 0.0
    chunk = max(1, _CHUNK_CELLS // max(1, n_items * n_profiles))
    for start in range(0, n_voters, chunk):
        rows = slice(start, start + chunk)
        den = np.einsum("ijs,jsc->ijc", states[rows], cell_probability)
        visible = shown[rows][..., np.newaxis]
        safe = np.where(visible, np.maximum(den, _MIN_PROBABILITY), 1.0)
        scores = prior[rows] + np.log(safe).sum(axis=1)
        totals = logsumexp(scores, axis=1)
        posterior = np.exp(scores - totals[:, np.newaxis])
        profile_probabilities[rows] = posterior
        ratio = np.where(visible, posterior[:, np.newaxis, :] / safe, 0.0)
        if voter_weights is None:
            log_likelihood += float(totals.sum())
        else:
            log_likelihood += float(totals @ voter_weights[rows])
            ratio *= voter_weights[rows, np.newaxis, np.newaxis]
        gathered += np.einsum("ijc,ijs->jsc", ratio, states[rows])
    counts: FloatArray = probabilities * np.einsum(
        "jsc,clq,jl->qls", gathered, follows, item_memberships
    )
    item_mass: FloatArray = item_memberships * np.einsum(
        "jsc,clq,qls->jl", gathered, follows, probabilities
    )
    role_by_block: FloatArray = np.einsum("ic,clq->ilq", profile_probabilities, follows)
    return _Expectation(profile_probabilities, role_by_block, counts, item_mass, log_likelihood)


@final
@dataclass(frozen=True, kw_only=True)
class MixedMembershipBlockModel(BlockModel[_MixedState, MixedCoclustering]):
    """Co-clustering where a voter can follow different groups on different blocks.

    Definition. With M voters, P items, Q groups and L blocks, the voter ``i`` has a
    degree of membership ``theta[i, q]`` to each group and the item ``j`` a degree of
    membership ``phi[j, l]`` to each block, each row summing to 1. The votes are drawn
    as :func:`~civiqhub_selene.blocks.synthetic.make_votes` draws them:

    1. for each block ``l``, the voter ``i`` takes a role ``c_l``, the group it follows
       on that block, drawn from ``theta[i]``; its profile is ``c = (c_1, ..., c_L)``,
       one of the ``Q**L`` profiles;
    2. each shown cell ``(i, j)`` takes a block ``l`` drawn from ``phi[j]``;
    3. its state ``s`` (shown without a vote, or a vote at one rank) follows the
       response model of the co-cluster ``(c_l, l)``, with probability
       ``F[c_l, l, s]``.

    A cell that is not shown carries no information and enters nowhere. With
    memberships of 0 and 1 only, this is the latent block model.

    Given the profile of a voter, its cells are independent, so the observed
    log-likelihood is computed exactly by summing over the profiles::

        log_likelihood = sum_i log sum_c prod_l theta[i, c_l]
                                     * prod_j sum_l phi[j, l] * F[c_l, l, s_ij]

    With optional mean-one survey weights ``w[i]``, ``log_likelihood`` is the
    weighted pseudo-likelihood of Skinner (1989) and Pfeffermann (1993)::

        log_likelihood = sum_i w[i] * log sum_c prod_l theta[i, c_l]
                                              * prod_j sum_l phi[j, l] * F[c_l, l, s_ij]

    The value reports this weighted ``log_likelihood``. The co-cluster counts on the
    value stay the raw, unweighted number of survey rows. A shown cell of item ``j``
    has ``K_j + 1`` modalities; this estimator reads one scale of ``K`` scores, shared
    by every item. A matrix whose items use several scales gets an absence of value,
    ``UNSUPPORTED_INPUT``, until the response models give each scale its grid (SEL-8.4).

    The model is estimated by expectation-maximisation, regularised by symmetric
    Dirichlet priors on the memberships, of concentrations ``voter_concentration``
    (alpha) and ``item_concentration`` (beta). It maximises::

        objective = log_likelihood + (alpha - 1) * sum_i w[i] * sum_q log theta[i, q]
                                   + (beta - 1) * sum_jl log phi[j, l]

    ``w[i]`` is 1 when the matrix has no survey weights. The prior on ``theta`` is
    weighted by voter; the prior on ``phi`` is not.

    The expectation step is exact: it enumerates the profiles and gives the posterior
    probability of each profile of each voter, the probability of each role on each
    block, and the expected number of cells of each co-cluster in each state. The
    maximisation step then sets::

        theta[i, q] proportional to  sum_l P(role on block l = q) + alpha - 1
        phi[j, l]   proportional to  expected cells of item j in block l + beta - 1

    and updates the response model from the expected counts, which are fractional.
    The objective never decreases when the response model updates by maximum
    likelihood, as ``OrdinalProbit`` and ``Categorical(smoothing=0)`` do; the
    pseudo-counts of a smoothed ``Categorical`` can lower it by a tiny amount near
    convergence, where the loop then stops. Several starts are run and the one with the largest
    objective is kept. Each start reads the arrangement of its ``initialization``
    (random by default, spectral with ``SpectralStart()``, or the result of a
    single-membership model with ``FixedStart``) and turns it into memberships that
    put ``start_weight`` on the group or block given and share the rest; the
    parameters of the response model start from the counts of that arrangement.

    The memberships of the value are ``theta`` and ``phi``; the main group of a voter
    and the main block of an item are their largest membership. Its table of
    co-clusters reads the expected counts. The value carries the observed
    log-likelihood; it has no integrated likelihood, and its shape is compared by the
    log-likelihood of held-out cells. A weighted ICL, were it used, would be a
    Hypothesis. The spectral start does not use the survey weights.

    The cost of an iteration grows as ``M * P * Q**L``: the number of profiles is
    bounded by :data:`MAX_PROFILES`.

    References: Airoldi, Blei, Fienberg and Xing (2008), Mixed membership stochastic
    blockmodels, Journal of Machine Learning Research 9, for the mixed membership;
    Dempster, Laird and Rubin (1977) for expectation-maximisation; Corneli, Bouveyron
    and Latouche (2020) for the ordinal response model; Skinner (1989) and
    Pfeffermann (1993) for the weighted pseudo-likelihood. The role per block and the
    exact expectation over the profiles follow the reference engine of the CiviqHub
    explainer "LBM and MM-OLBM step by step".

    Attributes:
        voter_concentration: The concentration alpha of the Dirichlet prior on the
            memberships of a voter, a finite number of at least 1. 1 is no prior.
            Defaults to 1.2, which keeps every membership away from 0.
        item_concentration: The concentration beta of the prior on the memberships of
            an item, likewise. Defaults to 1.
        start_weight: The membership a start gives to the group or block of its
            arrangement, from 0.5 to 1; the rest is shared evenly by the other
            clusters. With a single cluster, the membership is 1. Defaults to 0.9.

    The other settings are those of :class:`~civiqhub_selene.blocks.base.BlockModel`.

    Raises:
        InvalidParameterError: If a setting is out of range, or if ``n_groups **
            n_blocks`` exceeds :data:`MAX_PROFILES`.
    """

    name = "mixed_membership_block_model"
    version = "0.1.0"
    reference = (
        "Airoldi, Blei, Fienberg, Xing 2008; Dempster, Laird, Rubin 1977; "
        "Corneli, Bouveyron, Latouche 2020"
    )
    membership = "mixed"
    meaning = "a voter can follow different groups on different blocks"
    default_criterion = "held_out_likelihood"

    voter_concentration: float = 1.2
    item_concentration: float = 1.0
    start_weight: float = 0.9

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("voter_concentration", "item_concentration"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 1 <= value < math.inf
            ):
                raise InvalidParameterError(
                    f"{name} is a finite number of at least 1, got {value!r}"
                )
        weight = self.start_weight
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not 0.5 <= weight <= 1
        ):
            raise InvalidParameterError(f"start_weight is a number from 0.5 to 1, got {weight!r}")
        n_profiles = self.n_groups**self.n_blocks
        if n_profiles > MAX_PROFILES:
            raise InvalidParameterError(
                f"{self.n_groups} groups and {self.n_blocks} blocks make {n_profiles} profiles, "
                f"more than the {MAX_PROFILES} the exact expectation enumerates",
                hint="Lower n_groups or n_blocks: the cost grows as n_groups ** n_blocks.",
            )

    # Hooks ------------------------------------------------------------------------------

    def _start(self, votes: VoteMatrix, rng: np.random.Generator) -> _MixedState:
        shape = (self.n_groups, self.n_blocks)
        group_index, block_index = self.initialization.arrangement(votes, shape, rng)
        initial = self.response_model.initial_parameters(shape, votes, rng)
        counts = count_states(
            votes._states(),
            group_index,
            block_index,
            shape,
            voter_weights=votes._likelihood_weights,
        )
        parameters = self.response_model.update(counts, initial)
        return self._state(
            votes,
            _smoothed(group_index, self.n_groups, float(self.start_weight)),
            _smoothed(block_index, self.n_blocks, float(self.start_weight)),
            parameters,
        )

    def _step(self, votes: VoteMatrix, state: _MixedState) -> _MixedState:
        expectation = state.expectation
        alpha, beta = float(self.voter_concentration), float(self.item_concentration)
        voter_memberships = _normalised(
            np.maximum(expectation.role_by_block.sum(axis=1) + alpha - 1.0, 0.0)
        )
        item_memberships = _normalised(np.maximum(expectation.item_mass + beta - 1.0, 0.0))
        parameters = self.response_model.update(expectation.counts, state.parameters)
        return self._state(votes, voter_memberships, item_memberships, parameters)

    def _finish(
        self, votes: VoteMatrix, state: _MixedState, history: History, computation: str
    ) -> MixedCoclustering:
        expectation = state.expectation
        profiles = _profiles(self.n_groups, self.n_blocks)
        counts = expectation.counts
        if votes.voter_weights is not None:
            # The result still reports the number of observed votes, in survey rows.
            counts = _expect(
                votes._states(),
                votes.shown,
                state.voter_memberships,
                state.item_memberships,
                self.response_model.log_probabilities(state.parameters),
                profiles,
            ).counts
        return MixedCoclustering(
            computation=computation,
            voters=votes.voters,
            items=votes.items,
            item_scales=votes.item_scales,
            group_index=state.voter_memberships.argmax(axis=1),
            block_index=state.item_memberships.argmax(axis=1),
            group_memberships=state.voter_memberships,
            block_memberships=state.item_memberships,
            coclusters=self.response_model.to_table(state.parameters, counts, votes.scale),
            objective=state.objective,
            history=history,
            role_by_block=expectation.role_by_block,
            profiles=profiles,
            profile_probabilities=expectation.profile_probabilities,
            log_likelihood=expectation.log_likelihood,
        )

    # Pieces -----------------------------------------------------------------------------

    def _state(
        self,
        votes: VoteMatrix,
        voter_memberships: FloatArray,
        item_memberships: FloatArray,
        parameters: Any,
    ) -> _MixedState:
        """Run the expectation step at these memberships and parameters, and score them."""
        weights = votes._likelihood_weights
        expectation = _expect(
            votes._states(),
            votes.shown,
            voter_memberships,
            item_memberships,
            self.response_model.log_probabilities(parameters),
            _profiles(self.n_groups, self.n_blocks),
            weights,
        )
        alpha, beta = float(self.voter_concentration), float(self.item_concentration)
        objective = expectation.log_likelihood
        if alpha != 1.0:
            log_prior = np.log(np.maximum(voter_memberships, _MIN_PROBABILITY))
            prior_sum = log_prior.sum() if weights is None else log_prior.sum(axis=1) @ weights
            objective += (alpha - 1.0) * float(prior_sum)
        if beta != 1.0:
            objective += (beta - 1.0) * float(
                np.log(np.maximum(item_memberships, _MIN_PROBABILITY)).sum()
            )
        return _MixedState(voter_memberships, item_memberships, parameters, expectation, objective)
