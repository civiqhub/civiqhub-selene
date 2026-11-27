# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
from typing import Any

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene.blocks import (
    FixedStart,
    LatentBlockModel,
    MixedMembershipBlockModel,
    OrdinalScale,
    Partition,
    SpectralStart,
    VoteMatrix,
)
from civiqhub_selene.blocks.base import BlockModel, estimator_for, memberships
from civiqhub_selene.blocks.mixed import MAX_PROFILES, _expect, _profiles, _smoothed
from civiqhub_selene.blocks.responses import Categorical, OrdinalProbit
from civiqhub_selene.blocks.results import HasIntegratedLikelihood, MixedCoclustering, PredictsVotes
from civiqhub_selene.blocks.synthetic import make_votes, single_memberships
from civiqhub_selene.errors import InvalidParameterError
from civiqhub_selene.registry import default_registry
from civiqhub_selene.testing import EstimatorContract
from civiqhub_selene.types import Method, UnavailableReason

pytestmark = pytest.mark.blocks

# 60 voters of a single group, 20 shared between two groups; 12 items of one block, 2 shared.
VOTER_MEMBERSHIPS = np.vstack(
    [single_memberships((20, 20, 20)), [[0.5, 0.5, 0.0]] * 10, [[0.0, 0.5, 0.5]] * 10]
)
ITEM_MEMBERSHIPS = np.vstack([single_memberships((6, 6)), [[0.5, 0.5]] * 2])
VOTES, TRUTH = make_votes(VOTER_MEMBERSHIPS, ITEM_MEMBERSHIPS, seed=3)
MODEL = MixedMembershipBlockModel(n_groups=3, n_blocks=2, initialization=SpectralStart())


def small_parameters(votes: VoteMatrix, shape: tuple[int, int], seed: int) -> tuple[Any, Any, Any]:
    """Random memberships and state probabilities for a test of the expectation step."""
    rng = np.random.default_rng(seed)
    n_groups, n_blocks = shape
    theta = rng.dirichlet(np.ones(n_groups), size=votes.shape[0])
    phi = rng.dirichlet(np.ones(n_blocks), size=votes.shape[1])
    probabilities = rng.dirichlet(np.ones(votes.n_levels + 1), size=shape)
    return theta, phi, np.log(probabilities)


# Identity and settings ----------------------------------------------------------------


def test_identity() -> None:
    assert MixedMembershipBlockModel.method() == Method(
        "mixed_membership_block_model",
        "0.1.0",
        "Airoldi, Blei, Fienberg, Xing 2008; Dempster, Laird, Rubin 1977; "
        "Corneli, Bouveyron, Latouche 2020",
    )
    assert MixedMembershipBlockModel.membership == "mixed"
    assert MixedMembershipBlockModel.default_criterion == "held_out_likelihood"
    assert issubclass(MixedMembershipBlockModel, BlockModel)
    assert MixedMembershipBlockModel.input_type is VoteMatrix


def test_the_plain_word_mixed_gives_this_model() -> None:
    assert default_registry()["coclustering.mixed"].load() is MixedMembershipBlockModel
    assert estimator_for("mixed") == MixedMembershipBlockModel()
    assert memberships()["mixed"] == "a voter can follow different groups on different blocks"


def test_its_settings_and_its_shape() -> None:
    model = MixedMembershipBlockModel(voter_concentration=1.5, start_weight=0.6)
    assert model.with_shape(4, 3) == MixedMembershipBlockModel(
        n_groups=4, n_blocks=3, voter_concentration=1.5, start_weight=0.6
    )
    # A single group or block accepts the default start weight.
    assert MixedMembershipBlockModel().with_shape(1, 3).n_groups == 1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"voter_concentration": 0.9}, "voter_concentration"),
        ({"item_concentration": float("inf")}, "item_concentration"),
        ({"voter_concentration": True}, "voter_concentration"),
        ({"start_weight": 0.4}, "start_weight"),
        ({"start_weight": 1.1}, "start_weight"),
        ({"n_groups": 0}, "n_groups"),
        ({"n_groups": 5, "n_blocks": 6}, "profiles"),
    ],
)
def test_settings_out_of_range_are_refused(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(InvalidParameterError, match=message):
        MixedMembershipBlockModel(**changes)


def test_the_number_of_profiles_is_bounded() -> None:
    assert MAX_PROFILES == 4096
    assert MixedMembershipBlockModel(n_groups=4, n_blocks=6).n_blocks == 6
    with pytest.raises(InvalidParameterError, match="16384 profiles"):
        MixedMembershipBlockModel(n_groups=4, n_blocks=7)


# The expectation step -----------------------------------------------------------------


def brute_force(
    votes: VoteMatrix, theta: Any, phi: Any, logs: Any, shape: tuple[int, int]
) -> tuple[float, Any, Any, Any]:
    """The expectation step by explicit enumeration of profiles and blocks, cell by cell."""
    n_groups, n_blocks = shape
    probabilities = np.exp(logs)
    states = votes.levels
    log_likelihood = 0.0
    roles = np.zeros((votes.shape[0], n_blocks, n_groups))
    counts = np.zeros((n_groups, n_blocks, votes.n_levels + 1))
    item_mass = np.zeros((votes.shape[1], n_blocks))
    profiles = list(itertools.product(range(n_groups), repeat=n_blocks))
    for voter in range(votes.shape[0]):
        weights = []
        for profile in profiles:
            weight = math.prod(theta[voter, profile[block]] for block in range(n_blocks))
            for item in range(votes.shape[1]):
                if votes.shown[voter, item]:
                    weight *= sum(
                        phi[item, block] * probabilities[profile[block], block, states[voter, item]]
                        for block in range(n_blocks)
                    )
            weights.append(weight)
        total = sum(weights)
        log_likelihood += math.log(total)
        for profile, weight in zip(profiles, weights, strict=True):
            posterior = weight / total
            for block in range(n_blocks):
                roles[voter, block, profile[block]] += posterior
            for item in range(votes.shape[1]):
                if not votes.shown[voter, item]:
                    continue
                state = states[voter, item]
                cell = [
                    phi[item, block] * probabilities[profile[block], block, state]
                    for block in range(n_blocks)
                ]
                for block in range(n_blocks):
                    share = posterior * cell[block] / sum(cell)
                    counts[profile[block], block, state] += share
                    item_mass[item, block] += share
    return log_likelihood, roles, counts, item_mass


@pytest.mark.parametrize("shape", [(2, 2), (3, 2), (2, 3), (1, 2), (3, 1)])
def test_the_expectation_step_matches_a_brute_force_enumeration(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(sum(shape))
    votes = VoteMatrix.from_arrays(rng.integers(0, 6, size=(5, 4)), shown=rng.random((5, 4)) < 0.8)
    theta, phi, logs = small_parameters(votes, shape, seed=1)
    expectation = _expect(votes._states(), votes.shown, theta, phi, logs, _profiles(*shape))
    log_likelihood, roles, counts, item_mass = brute_force(votes, theta, phi, logs, shape)
    assert expectation.log_likelihood == pytest.approx(log_likelihood, rel=1e-12)
    assert expectation.role_by_block == pytest.approx(roles, abs=1e-12)
    assert expectation.counts == pytest.approx(counts, abs=1e-12)
    assert expectation.item_mass == pytest.approx(item_mass, abs=1e-12)
    assert expectation.profile_probabilities.sum(axis=1) == pytest.approx(np.ones(5))


def test_the_expectation_counts_every_shown_cell_once() -> None:
    theta, phi, logs = small_parameters(VOTES, (3, 2), seed=2)
    expectation = _expect(VOTES._states(), VOTES.shown, theta, phi, logs, _profiles(3, 2))
    assert expectation.counts.sum() == pytest.approx(VOTES.n_shown)
    assert expectation.counts[..., 1:].sum() == pytest.approx(VOTES.n_votes)
    assert expectation.item_mass.sum(axis=1) == pytest.approx(VOTES.shown.sum(axis=0))
    assert expectation.role_by_block.sum(axis=2) == pytest.approx(np.ones((80, 2)))


def test_the_expectation_is_the_same_by_chunks() -> None:
    import civiqhub_selene.blocks.mixed as mixed

    theta, phi, logs = small_parameters(VOTES, (3, 2), seed=4)
    whole = _expect(VOTES._states(), VOTES.shown, theta, phi, logs, _profiles(3, 2))
    original = mixed._CHUNK_CELLS
    try:
        mixed._CHUNK_CELLS = 50
        chunked = _expect(VOTES._states(), VOTES.shown, theta, phi, logs, _profiles(3, 2))
    finally:
        mixed._CHUNK_CELLS = original
    assert chunked.log_likelihood == pytest.approx(whole.log_likelihood, rel=1e-12)
    assert chunked.counts == pytest.approx(whole.counts)


def test_profiles_follow_the_order_of_itertools_product() -> None:
    assert _profiles(2, 2).tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert _profiles(1, 3).tolist() == [[0, 0, 0]]
    assert _smoothed(np.array([1, 0]), 3, 0.9) == pytest.approx(
        np.array([[0.05, 0.9, 0.05], [0.9, 0.05, 0.05]])
    )
    assert _smoothed(np.array([0, 0]), 1, 0.9).tolist() == [[1.0], [1.0]]


# Estimation ---------------------------------------------------------------------------


def test_the_value_is_a_mixed_coclustering_of_the_votes() -> None:
    result = MODEL.estimate(VOTES, seed=0)
    value = result.value
    assert isinstance(value, MixedCoclustering)
    assert isinstance(value, PredictsVotes)
    assert not isinstance(value, HasIntegratedLikelihood)
    assert (value.n_groups, value.n_blocks) == (3, 2)
    assert value.voters == VOTES.voters
    assert value.items == VOTES.items
    assert value.computation == result.fingerprint()
    assert value.profiles.tolist() == _profiles(3, 2).tolist()
    assert value.profile_probabilities.sum(axis=1) == pytest.approx(np.ones(80))
    assert value.group_memberships.sum(axis=1) == pytest.approx(np.ones(80))
    assert value.block_memberships.sum(axis=1) == pytest.approx(np.ones(14))
    assert value.coclusters.n_votes.sum() == pytest.approx(VOTES.n_votes)
    assert value.objective <= value.log_likelihood
    assert np.isfinite(value.log_probability(VOTES)).all()


def test_the_planted_roles_are_found_again() -> None:
    value = MODEL.estimate(VOTES, seed=0).value
    roles = value.role_by_block.argmax(axis=2)
    # Match the blocks found with the planted ones through the pure items.
    planted_block_of = {value.block_of(f"i{item}").index: block for item, block in ((1, 0), (7, 1))}
    for found, planted in planted_block_of.items():
        groups = Partition.from_voter_labels(VOTES.voters, roles[:, found])
        truth = Partition.from_voter_labels(VOTES.voters, TRUTH.roles[:, planted])
        assert groups.agreement_with(truth) >= 0.9


def test_shared_voters_are_more_mixed_than_the_others() -> None:
    value = MODEL.estimate(VOTES, seed=0).value
    assert value.mixedness[60:].mean() > value.mixedness[:60].mean() + 0.1
    assert value.role_changes().sum() == 80


@pytest.mark.parametrize("response_model", [OrdinalProbit(), Categorical(smoothing=0.0)])
def test_the_objective_never_goes_down(response_model: Any) -> None:
    rng = np.random.default_rng(11)
    matrices = [
        VOTES,
        VoteMatrix.from_arrays(rng.integers(0, 6, (25, 8)), shown=rng.random((25, 8)) < 0.7),
    ]
    for votes in matrices:
        for seed in range(2):
            model = MixedMembershipBlockModel(
                n_groups=3, n_blocks=2, response_model=response_model, n_starts=1, max_iterations=60
            )
            value = model.estimate(votes, seed=seed).value
            objectives = np.array(value.history.objectives)
            assert np.isfinite(objectives).all()
            assert (np.diff(objectives) >= -1e-7).all()
            assert value.objective == objectives[-1]


def test_a_smoothed_categorical_stops_where_its_pseudo_counts_lower_the_objective() -> None:
    model = MixedMembershipBlockModel(
        n_groups=3, n_blocks=2, response_model=Categorical(), n_starts=1
    )
    value = model.estimate(VOTES, seed=0).value
    steps = np.diff(value.history.objectives)
    assert value.history.converged
    # Every step but the last one raises the objective; the last one is a tiny dip at most.
    assert (steps[:-1] >= -1e-7).all()
    assert steps[-1] > -1e-3


def test_a_spectral_start_reaches_a_better_optimum_than_few_random_starts() -> None:
    spectral = MixedMembershipBlockModel(
        n_groups=3, n_blocks=2, n_starts=2, initialization=SpectralStart()
    ).estimate(VOTES, seed=0)
    random = MixedMembershipBlockModel(n_groups=3, n_blocks=2, n_starts=2).estimate(VOTES, seed=0)
    assert spectral.value.objective >= random.value.objective - 1e-3


def test_it_can_start_from_a_single_membership_result() -> None:
    single = LatentBlockModel(n_groups=3, n_blocks=2).estimate(VOTES, seed=0).value
    start = FixedStart.from_coclustering(single)
    model = MixedMembershipBlockModel(n_groups=3, n_blocks=2, n_starts=1, initialization=start)
    result = model.estimate(VOTES, seed=0)
    assert result.value.groups.agreement_with(single.groups) >= 0.9
    parameters = dict(result.provenance.parameters)
    assert ("source", single.computation) in parameters["initialization"]


def test_what_lies_under_a_cell_that_is_not_shown_changes_nothing() -> None:
    hidden = np.random.default_rng(7).random(VOTES.shape) < 0.3
    held = VOTES.hold_out(hidden)
    erased = VoteMatrix.from_arrays(np.where(hidden, 0, VOTES.levels), shown=~hidden)
    model = MixedMembershipBlockModel(n_groups=3, n_blocks=2, n_starts=2, max_iterations=50)
    first = model.estimate(held, seed=0).value
    second = model.estimate(erased, seed=0).value
    assert first.fingerprint() == second.fingerprint()
    assert first.objective == second.objective


def test_same_votes_settings_and_seed_give_the_same_value() -> None:
    model = MixedMembershipBlockModel(n_groups=3, n_blocks=2, n_starts=2, max_iterations=50)
    first = model.estimate(VOTES, seed=0)
    second = model.estimate(VOTES, seed=0)
    assert first == second
    assert first.value.fingerprint() == second.value.fingerprint()
    assert first.fingerprint() != model.estimate(VOTES, seed=1).fingerprint()


def test_a_matrix_without_votes_is_an_explained_absence() -> None:
    absent = MODEL.estimate(VoteMatrix.from_records([], voters=["a", "b", "c"], items=["x", "y"]))
    assert not absent.is_available
    assert absent.unavailable is not None
    assert absent.unavailable.reason == UnavailableReason.NO_VOTES


class TestMixedMembershipBlockModel(EstimatorContract):
    def make_estimator(self) -> MixedMembershipBlockModel:
        return MixedMembershipBlockModel(n_groups=3, n_blocks=2, n_starts=2, max_iterations=40)

    def make_data(self) -> VoteMatrix:
        return VOTES

    def make_empty_data(self) -> VoteMatrix:
        return VoteMatrix.from_records([], voters=["a", "b"], items=["x", "y"])


# Items on several scales --------------------------------------------------------------
# TEST ONLY. Synthetic collectives whose items mix scales of two to eleven answers.


def several_scales(seed: int) -> tuple[VoteMatrix, Any]:
    two, four, five, six = (OrdinalScale(tuple(range(1, size + 1))) for size in (2, 4, 5, 6))
    return make_votes(
        single_memberships((40, 35, 30)),
        single_memberships((12, 12)),
        latent_mean=[[0.8, 0.25], [0.2, 0.75], [0.5, 0.5]],
        latent_spread=[[0.12, 0.14], [0.14, 0.12], [0.2, 0.2]],
        item_scales=[(two, four, five, six)[item % 4] for item in range(24)],
        seed=seed,
    )


@pytest.mark.parametrize("seed", [1, 2])
def test_planted_groups_are_found_on_items_of_four_scales(seed: int) -> None:
    votes, truth = several_scales(seed)
    value = MixedMembershipBlockModel(n_groups=3, n_blocks=2).estimate(votes, seed=0).value
    groups = Partition.from_voter_labels(votes.voters, truth.roles[:, 0])
    blocks = Partition.from_item_labels(votes.items, np.repeat([0, 1], 12))
    assert value.groups.agreement_with(groups) >= 0.95
    assert value.blocks.agreement_with(blocks) >= 0.95
    assert (np.diff(value.history.objectives) >= -1e-9).all()
    assert value.coclusters.scales == votes.scales
    assert value.coclusters.n_votes.sum() == pytest.approx(votes.n_votes)
    # The value predicts each cell on the table of the scale of its item.
    logs = value.log_probability(votes)
    assert logs.shape == votes.shape
    assert np.isfinite(logs).all()
    assert (logs <= 0).all()


def test_the_expectation_reads_each_item_on_its_scale() -> None:
    votes, _ = several_scales(3)
    model = MixedMembershipBlockModel(n_groups=2, n_blocks=2)
    rng = np.random.default_rng(0)
    params = model.response_model.initial_parameters((2, 2), votes, rng)
    theta = rng.dirichlet(np.ones(2), size=votes.shape[0])
    phi = rng.dirichlet(np.ones(2), size=votes.shape[1])
    logs = model.response_model.log_probabilities_by_scale(params)
    expectation = _expect(
        votes._states(),
        votes.shown,
        theta,
        phi,
        logs,
        _profiles(2, 2),
        scale_index=votes.item_scales.scale_index,
    )
    assert expectation.counts.shape == (2, 2, 4, 7)
    # Every shown cell is counted once, on its scale, never past the levels of its scale.
    assert expectation.counts.sum() == pytest.approx(votes.n_shown)
    possible = votes.item_scales._possible_states
    assert np.abs(expectation.counts[:, :, ~possible]).max() < 1e-9
    by_item = np.bincount(votes.item_scales.scale_index, weights=votes.shown.sum(axis=0))
    assert expectation.counts.sum(axis=(0, 1, 3)) == pytest.approx(by_item)
    # On one scale, the expectation without an axis of scales is the same computation.
    one, _ = make_votes(VOTER_MEMBERSHIPS, ITEM_MEMBERSHIPS, seed=0)
    params_one = model.response_model.initial_parameters((2, 2), one, np.random.default_rng(1))
    flat = _expect(
        one._states(),
        one.shown,
        np.full((one.shape[0], 2), 0.5),
        np.full((one.shape[1], 2), 0.5),
        model.response_model.log_probabilities(params_one),
        _profiles(2, 2),
    )
    by_scale = _expect(
        one._states(),
        one.shown,
        np.full((one.shape[0], 2), 0.5),
        np.full((one.shape[1], 2), 0.5),
        model.response_model.log_probabilities_by_scale(params_one),
        _profiles(2, 2),
        scale_index=one.item_scales.scale_index,
    )
    assert np.array_equal(flat.counts, by_scale.counts[:, :, 0, :])
    assert flat.log_likelihood == by_scale.log_likelihood


def test_the_memory_stays_bounded_at_the_size_of_a_national_survey() -> None:
    import tracemalloc

    # TEST ONLY. 4,000 voters and 270 items on seven scales of 2 to 11 answers: the shape
    # of a national opinion barometer, with synthetic votes.
    sizes = (2, 3, 4, 5, 6, 7, 11)
    scales = [OrdinalScale(tuple(range(1, sizes[item % 7] + 1))) for item in range(270)]
    votes, _ = make_votes(
        single_memberships((1500, 1500, 1000)),
        single_memberships((135, 135)),
        latent_mean=[[0.8, 0.25], [0.2, 0.75], [0.5, 0.5]],
        latent_spread=[[0.12, 0.14], [0.14, 0.12], [0.2, 0.2]],
        item_scales=scales,
        seed=3,
    )
    model = MixedMembershipBlockModel(n_groups=3, n_blocks=2, n_starts=1, max_iterations=2)
    tracemalloc.start()
    try:
        result = model.estimate(votes, seed=0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.is_available
    # The states of the cells take 99 MiB; the whole computation stays under 512 MiB.
    assert peak < 512 * 2**20


def test_estimated_thresholds_keep_the_objective_from_decreasing() -> None:
    from civiqhub_selene.blocks.responses import OrdinalProbit, Thresholds

    six = OrdinalScale(tuple(range(1, 7)))
    planted = (1 / 6, 0.25, 0.45, 0.7, 5 / 6)
    votes, _ = make_votes(
        single_memberships((80, 80)),
        single_memberships((10, 10)),
        vote_rate=[[0.9, 0.9], [0.9, 0.9]],
        latent_mean=[[0.75, 0.35], [0.3, 0.65]],
        latent_spread=[[0.2, 0.2], [0.2, 0.2]],
        item_scales=[six, OrdinalScale((1, 2))] * 10,
        thresholds={six: planted},
        seed=0,
    )
    model = MixedMembershipBlockModel(
        n_groups=2,
        n_blocks=2,
        max_iterations=60,
        response_model=OrdinalProbit(thresholds=Thresholds(mode="estimated")),
    )
    value = model.estimate(votes, seed=0).value
    assert (np.diff(value.history.objectives) >= -1e-7).all()
    assert value.coclusters.thresholds is not None
    grid = value.coclusters.thresholds[value.coclusters.scales.index(six)]
    assert grid == pytest.approx(planted, abs=0.03)


# Shared vote rate ---------------------------------------------------------------------
# TEST ONLY. Two planted groups with the same opinion on every item, which differ only in
# how often they vote: only the abstentions tell them apart.

RATE_TRUTH = np.repeat([0, 1], 60)
RATE_ONLY, _ = make_votes(
    single_memberships((60, 60)),
    single_memberships((12,)),
    vote_rate=[[0.95], [0.55]],
    latent_mean=[[3.0], [3.0]],
    latent_spread=[[1.0], [1.0]],
    seed=0,
)
RATE_MASK = np.random.default_rng(0).random(RATE_ONLY.shape) < 0.1


def _recovered(groups: Any) -> float:
    found = np.asarray(groups)
    return float(max((found == RATE_TRUTH).mean(), (found != RATE_TRUTH).mean()))


def test_groups_told_apart_by_their_abstentions_only_need_a_vote_rate_per_cocluster() -> None:
    by_cocluster = MixedMembershipBlockModel(n_groups=2, n_blocks=1)
    shared = MixedMembershipBlockModel(
        n_groups=2, n_blocks=1, response_model=OrdinalProbit(shared_vote_rate=True)
    )
    found = by_cocluster.estimate(RATE_ONLY, seed=0).value
    blind = shared.estimate(RATE_ONLY, seed=0).value
    assert _recovered(found.group_index) > 0.9
    assert _recovered(blind.group_index) < 0.7
    assert np.unique(blind.coclusters.vote_rate).size == 1
    assert blind.coclusters.vote_rate[0, 0] == pytest.approx(RATE_ONLY.n_votes / RATE_ONLY.n_shown)
    held_out = [
        model.estimate(RATE_ONLY.hold_out(RATE_MASK), seed=0)
        .value.log_probability(RATE_ONLY)[RATE_MASK]
        .mean()
        for model in (by_cocluster, shared)
    ]
    assert held_out[0] > held_out[1]
