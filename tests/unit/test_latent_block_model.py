# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import math
import subprocess
import sys
import time
from typing import Any

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene.blocks import LatentBlockModel, OrdinalScale, Partition, VoteMatrix
from civiqhub_selene.blocks.base import BlockModel, estimator_for, memberships
from civiqhub_selene.blocks.responses import Categorical, OrdinalProbit, Thresholds
from civiqhub_selene.blocks.results import (
    HasIntegratedLikelihood,
    PredictsVotes,
    SingleCoclustering,
)
from civiqhub_selene.blocks.single import _sweep
from civiqhub_selene.blocks.synthetic import make_votes, single_memberships
from civiqhub_selene.errors import InvalidInputError
from civiqhub_selene.registry import default_registry
from civiqhub_selene.testing import EstimatorContract
from civiqhub_selene.types import Method, UnavailableReason, settings_of

pytestmark = pytest.mark.blocks

VOTES, TRUTH = make_votes(single_memberships((20, 16, 12)), single_memberships((7, 7)), seed=0)
PLANTED_GROUPS = Partition.from_voter_labels(VOTES.voters, TRUTH.roles[:, 0])
PLANTED_BLOCKS = Partition.from_item_labels(VOTES.items, np.repeat([0, 1], 7))
MODEL = LatentBlockModel(n_groups=3, n_blocks=2)
RESPONSE_MODELS = [OrdinalProbit(), Categorical()]


def ten_matrices() -> list[VoteMatrix]:
    """Planted collectives, pure noise, mixed memberships and unanimous co-clusters."""
    matrices = [
        make_votes(single_memberships((20, 16, 12)), single_memberships((7, 7)), seed=seed)[0]
        for seed in range(4)
    ]
    rng = np.random.default_rng(100)
    for _ in range(4):
        shape = (int(rng.integers(8, 40)), int(rng.integers(4, 16)))
        matrices.append(
            VoteMatrix.from_arrays(rng.integers(0, 6, size=shape), shown=rng.random(shape) < 0.8)
        )
    spread = np.vstack([single_memberships((15, 15)), np.full((20, 2), 0.5)])
    matrices.append(
        make_votes(
            spread,
            single_memberships((6, 6)),
            vote_rate=np.full((2, 2), 0.8),
            latent_mean=[[4.7, 1.4], [1.3, 4.6]],
            latent_spread=np.full((2, 2), 0.6),
            seed=3,
        )[0]
    )
    unanimous = np.full((12, 6), 5)
    unanimous[:6, 3:] = 3
    unanimous[6:, 3:] = 1
    matrices.append(VoteMatrix.from_arrays(unanimous))
    return matrices


# Identity and settings ----------------------------------------------------------------


def test_identity() -> None:
    assert LatentBlockModel.method() == Method(
        "latent_block_model", "0.1.0", "Govaert and Nadif 2003; Corneli, Bouveyron, Latouche 2020"
    )
    assert LatentBlockModel.membership == "single"
    assert LatentBlockModel.meaning == "each voter belongs to one group"
    assert LatentBlockModel.default_criterion == "integrated_likelihood"
    assert issubclass(LatentBlockModel, BlockModel)
    assert LatentBlockModel.input_type is VoteMatrix


def test_it_has_no_setting_of_its_own() -> None:
    assert [name for name, _ in settings_of(LatentBlockModel())] == [
        "initialization",
        "max_iterations",
        "n_blocks",
        "n_groups",
        "n_starts",
        "response_model",
        "tolerance",
    ]
    assert LatentBlockModel() == LatentBlockModel(n_groups=2, n_blocks=2)
    assert LatentBlockModel().with_shape(4, 3) == LatentBlockModel(n_groups=4, n_blocks=3)


def test_the_plain_word_single_gives_this_model() -> None:
    assert default_registry()["coclustering.single"].load() is LatentBlockModel
    assert estimator_for("single") == LatentBlockModel()
    assert memberships()["single"] == "each voter belongs to one group"


# Recovery -----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_planted_groups_and_blocks_are_found_again(seed: int) -> None:
    result = MODEL.estimate(VOTES, seed=seed)
    value = result.value
    assert isinstance(value, SingleCoclustering)
    assert value.groups.agreement_with(PLANTED_GROUPS) >= 0.9
    assert value.blocks.agreement_with(PLANTED_BLOCKS) == 1.0
    assert (value.n_groups, value.n_blocks) == (3, 2)
    assert value.history.converged is True
    assert value.computation == result.fingerprint()
    assert value.groups.computation == result.fingerprint()
    assert value.voters == VOTES.voters
    assert value.items == VOTES.items
    assert value.scale == VOTES.scale


def test_the_table_reads_how_each_group_answers_each_block() -> None:
    value = MODEL.estimate(VOTES, seed=0).value
    approving = value.group_of("v1").index
    first_block = value.block_of("i1").index
    second_block = value.block_of("i14").index
    assert value[approving, first_block].typical_score == 2
    assert value[approving, second_block].typical_score == -2
    assert value[approving, first_block].latent_mean == pytest.approx(4.7, abs=0.3)
    assert value[approving, first_block].vote_rate == pytest.approx(0.85, abs=0.1)
    assert value.coclusters.n_votes.sum() == VOTES.n_votes


def test_hidden_cells_do_not_weigh_on_the_arrangement() -> None:
    hidden = np.random.default_rng(7).random(VOTES.shape) < 0.3
    held = VOTES.hold_out(hidden)
    value = MODEL.estimate(held, seed=0).value
    assert value.groups.agreement_with(PLANTED_GROUPS) > 0.8
    assert value.blocks.agreement_with(PLANTED_BLOCKS) == 1.0
    assert value.coclusters.n_votes.sum() == held.n_votes


def test_what_lies_under_a_cell_that_is_not_shown_changes_nothing() -> None:
    hidden = np.random.default_rng(7).random(VOTES.shape) < 0.3
    held = VOTES.hold_out(hidden)
    erased = VoteMatrix.from_arrays(np.where(hidden, 0, VOTES.levels), shown=~hidden)
    assert held != erased
    first = MODEL.estimate(held, seed=0).value
    second = MODEL.estimate(erased, seed=0).value
    assert first.fingerprint() == second.fingerprint()
    assert first.objective == second.objective


# Guarantees ---------------------------------------------------------------------------


@pytest.mark.parametrize("response_model", RESPONSE_MODELS)
def test_the_objective_never_goes_down(response_model: Any) -> None:
    for votes in ten_matrices():
        for seed in range(3):
            model = LatentBlockModel(
                n_groups=3, n_blocks=2, response_model=response_model, n_starts=1
            )
            value = model.estimate(votes, seed=seed).value
            objectives = np.array(value.history.objectives)
            assert np.isfinite(objectives).all()
            assert (np.diff(objectives) >= -1e-9).all()
            assert value.objective == objectives[-1]


@pytest.mark.parametrize("response_model", RESPONSE_MODELS)
@pytest.mark.parametrize("shape", [(3, 2), (5, 4), (1, 3)])
def test_no_group_and_no_block_is_empty(response_model: Any, shape: tuple[int, int]) -> None:
    for votes in ten_matrices()[:6]:
        n_groups, n_blocks = shape
        model = LatentBlockModel(
            n_groups=n_groups, n_blocks=n_blocks, response_model=response_model, n_starts=2
        )
        value = model.estimate(votes, seed=0).value
        assert min(value.groups.sizes) >= 1
        assert min(value.blocks.sizes) >= 1
        assert sum(value.groups.sizes) == votes.shape[0]
        assert value.group_memberships.sum(axis=1) == pytest.approx(np.ones(votes.shape[0]))
        assert value.block_memberships.sum(axis=1) == pytest.approx(np.ones(votes.shape[1]))
        assert np.isfinite(value.integrated_likelihood)


@pytest.mark.parametrize("response_model", RESPONSE_MODELS)
def test_same_votes_settings_and_seed_give_the_same_value(response_model: Any) -> None:
    model = LatentBlockModel(n_groups=3, n_blocks=2, response_model=response_model)
    first = model.estimate(VOTES, seed=0)
    second = model.estimate(VOTES, seed=0)
    assert first == second
    assert first.value.fingerprint() == second.value.fingerprint()
    assert first.value.objective == second.value.objective
    assert first.fingerprint() != model.estimate(VOTES, seed=1).fingerprint()


def test_the_value_has_the_same_fingerprint_in_another_process() -> None:
    script = (
        "from civiqhub_selene.blocks import LatentBlockModel\n"
        "from civiqhub_selene.blocks.synthetic import make_votes, single_memberships\n"
        "votes, _ = make_votes(single_memberships((20, 16, 12)), single_memberships((7, 7)))\n"
        "result = LatentBlockModel(n_groups=3, n_blocks=2).estimate(votes, seed=0)\n"
        "print(result.value.fingerprint(), result.fingerprint())\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    result = MODEL.estimate(VOTES, seed=0)
    assert completed.stdout.split() == [result.value.fingerprint(), result.fingerprint()]


def test_a_single_cocluster_has_the_vote_rate_of_the_matrix() -> None:
    value = LatentBlockModel(n_groups=1, n_blocks=1).estimate(VOTES, seed=0).value
    assert (value.n_groups, value.n_blocks) == (1, 1)
    assert value.groups.sizes == (48,)
    assert value[0, 0].vote_rate == pytest.approx(VOTES.n_votes / VOTES.n_shown)
    assert value[0, 0].n_votes == VOTES.n_votes
    assert value.group_memberships.tolist() == [[1.0]] * 48
    # No shape term: the integrated likelihood only pays for the response model.
    assert value.integrated_likelihood == pytest.approx(
        value.objective - OrdinalProbit().complexity_penalty((1, 1), VOTES)
    )


def test_no_answer_is_explicit_and_nothing_is_computed(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*arguments: object) -> None:
        raise AssertionError("a computation was started")

    monkeypatch.setattr(LatentBlockModel, "_start", refuse)
    empty = MODEL.estimate(VoteMatrix.from_records([]))
    assert empty.unavailable is not None
    assert empty.unavailable.reason is UnavailableReason.NO_VOTES
    two_voters = VoteMatrix.from_table([[2, 1, 0], [-2, -1, 0], [None, None, None]])
    too_many = MODEL.estimate(two_voters)
    assert too_many.unavailable is not None
    assert too_many.unavailable.reason is UnavailableReason.NOT_ENOUGH_DATA
    assert too_many.provenance.method.name == "latent_block_model"


@pytest.mark.parametrize("response_model", RESPONSE_MODELS)
def test_integrated_likelihood_follows_its_formula(response_model: Any) -> None:
    model = LatentBlockModel(n_groups=3, n_blocks=2, response_model=response_model)
    value = model.estimate(VOTES, seed=0).value
    expected = (
        value.objective
        - response_model.complexity_penalty((3, 2), VOTES)
        - 0.5 * (3 - 1) * math.log(48)
        - 0.5 * (2 - 1) * math.log(14)
    )
    assert value.integrated_likelihood == pytest.approx(expected, abs=1e-9)
    assert value.integrated_likelihood < value.objective
    assert isinstance(value, HasIntegratedLikelihood)
    assert isinstance(value, PredictsVotes)


@pytest.mark.parametrize("response_model", RESPONSE_MODELS)
def test_the_objective_is_the_classified_log_likelihood_of_the_value(response_model: Any) -> None:
    model = LatentBlockModel(n_groups=3, n_blocks=2, response_model=response_model)
    value = model.estimate(VOTES, seed=0).value
    cells = float(value.log_probability(VOTES)[VOTES.shown].sum())
    group_shares = np.array(value.groups.sizes) / 48
    block_shares = np.array(value.blocks.sizes) / 14
    labels = float(np.log(group_shares[value.group_index]).sum()) + float(
        np.log(block_shares[value.block_index]).sum()
    )
    assert value.objective == pytest.approx(cells + labels, abs=1e-5)


def test_categorical_response_model_finds_the_same_arrangement() -> None:
    probit = MODEL.estimate(VOTES, seed=0)
    categorical = LatentBlockModel(n_groups=3, n_blocks=2, response_model=Categorical()).estimate(
        VOTES, seed=0
    )
    assert categorical.value.groups.agreement_with(probit.value.groups) == 1.0
    assert categorical.value.blocks.agreement_with(PLANTED_BLOCKS) == 1.0
    assert categorical.value[0, 0].latent_mean is None
    assert probit.value[0, 0].latent_mean is not None
    assert categorical.fingerprint() != probit.fingerprint()
    # The free distribution pays for more parameters than the two of the latent opinion.
    penalty = categorical.value.objective - categorical.value.integrated_likelihood
    assert penalty > probit.value.objective - probit.value.integrated_likelihood


def test_one_computation_on_the_reference_votes_is_fast() -> None:
    started = time.perf_counter()
    MODEL.estimate(VOTES, seed=0)
    assert time.perf_counter() - started < 2.0


def test_the_soft_memberships_lean_to_the_group_of_each_voter() -> None:
    value = MODEL.estimate(VOTES, seed=0).value
    assert (value.group_memberships.argmax(axis=1) == value.group_index).mean() >= 0.95
    assert (value.block_memberships.argmax(axis=1) == value.block_index).all()
    assert ((value.group_memberships >= 0) & (value.group_memberships <= 1)).all()


# Sweep --------------------------------------------------------------------------------


def test_the_sweep_moves_to_a_strictly_better_cluster_in_order() -> None:
    index = np.array([0, 0, 1, 1])
    scores = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [2.0, 1.0]])
    assert _sweep(index, scores).tolist() == [1, 0, 1, 0]
    assert index.tolist() == [0, 0, 1, 1]


def test_the_last_member_of_a_cluster_never_leaves_it() -> None:
    # Every element prefers the cluster 1: the cluster 0 keeps its last member.
    index = np.array([0, 0, 0, 1])
    scores = np.array([[0.0, 1.0]] * 4)
    assert _sweep(index, scores).tolist() == [1, 1, 0, 1]
    alone = np.array([0, 1])
    assert _sweep(alone, np.array([[0.0, 1.0], [1.0, 0.0]])).tolist() == [0, 1]


def test_sizes_are_updated_as_elements_move() -> None:
    # The element 0 joins the cluster 1 first, which then lets the element 2 leave it.
    index = np.array([0, 0, 1])
    scores = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    assert _sweep(index, scores).tolist() == [1, 0, 0]


class TestLatentBlockModel(EstimatorContract):
    def make_estimator(self) -> LatentBlockModel:
        return LatentBlockModel(n_groups=3, n_blocks=2, n_starts=2)

    def make_data(self) -> VoteMatrix:
        return VOTES

    def make_empty_data(self) -> VoteMatrix:
        return VoteMatrix.from_records([])


# Items on several scales --------------------------------------------------------------
# TEST ONLY. Synthetic collectives whose items mix scales of two, four, five and six
# answers in each block, like the questions of a survey; the truth is planted on the unit
# axis, where the ordinal probit reads several scales by default.

TWO, FOUR, FIVE, SIX = (OrdinalScale(tuple(range(1, size + 1))) for size in (2, 4, 5, 6))
MIXED_SCALES = [(TWO, FOUR, FIVE, SIX)[item % 4] for item in range(24)]
PLANTED_MEANS = [[0.8, 0.25], [0.2, 0.75], [0.5, 0.5]]
PLANTED_SPREADS = [[0.12, 0.14], [0.14, 0.12], [0.2, 0.2]]


def most_common(labels: Any) -> int:
    counts: Any = np.bincount(labels)
    return int(counts.argmax())


def mixed(seed: int) -> tuple[VoteMatrix, Any]:
    return make_votes(
        single_memberships((40, 35, 30)),
        single_memberships((12, 12)),
        latent_mean=PLANTED_MEANS,
        latent_spread=PLANTED_SPREADS,
        item_scales=MIXED_SCALES,
        seed=seed,
    )


@pytest.mark.parametrize("seed", range(5))
def test_planted_groups_and_opinions_are_found_on_items_of_four_scales(seed: int) -> None:
    votes, truth = mixed(seed)
    assert votes.item_scales.n_scales == 4
    value = LatentBlockModel(n_groups=3, n_blocks=2).estimate(votes, seed=seed).value
    groups = Partition.from_voter_labels(votes.voters, truth.roles[:, 0])
    blocks = Partition.from_item_labels(votes.items, np.repeat([0, 1], 12))
    assert value.groups.agreement_with(groups) >= 0.95
    assert value.blocks.agreement_with(blocks) >= 0.95
    # Match the groups and blocks found to the planted ones, then read their opinions.
    planted_blocks = np.repeat([0, 1], 12)
    found_group = [most_common(value.group_index[truth.roles[:, 0] == q]) for q in range(3)]
    found_block = [most_common(value.block_index[planted_blocks == block]) for block in range(2)]
    table = value.coclusters
    assert table.latent_mean is not None
    assert table.latent_spread is not None
    for q in range(3):
        for block in range(2):
            cell = (found_group[q], found_block[block])
            assert table.latent_mean[cell] == pytest.approx(PLANTED_MEANS[q][block], abs=0.06)
            assert table.latent_spread[cell] == pytest.approx(PLANTED_SPREADS[q][block], abs=0.06)
    assert (np.diff(value.history.objectives) >= -1e-9).all()


def test_a_value_on_several_scales_keeps_them_apart() -> None:
    votes, _ = mixed(0)
    result = LatentBlockModel(n_groups=3, n_blocks=2).estimate(votes, seed=0)
    value = result.value
    assert value.item_scales == votes.item_scales
    assert value.scales == (TWO, FOUR, FIVE, SIX)
    with pytest.raises(InvalidInputError, match="4 scales, not one"):
        _ = value.scale
    table = value.coclusters
    assert [array.shape[-1] for array in table.vote_shares_by_scale] == [2, 4, 5, 6]
    assert table.thresholds is not None
    assert next(grid.tolist() for grid in table.thresholds) == [0.5]
    data = value.to_dict()
    assert "scores" not in data
    assert data["scales"] == [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]
    assert data["scale_index"] == votes.item_scales.scale_index.tolist()
    assert "vote_shares_by_scale" in data["coclusters"]
    # The cells read the table of their scale: their log-probabilities sum to the objective.
    cells = float(value.log_probability(votes)[votes.shown].sum())
    shares = np.bincount(value.group_index) / votes.shape[0]
    block_shares = np.bincount(value.block_index) / votes.shape[1]
    labels = float(np.log(shares[value.group_index]).sum()) + float(
        np.log(block_shares[value.block_index]).sum()
    )
    assert value.objective == pytest.approx(cells + labels, abs=1e-6)
    with pytest.raises(InvalidInputError, match="same voters, items and scales"):
        value.log_probability(VoteMatrix.from_arrays(votes.levels.clip(0, 2), scale=TWO))


def test_the_levels_axis_and_the_unit_axis_give_the_same_arrangement_on_one_scale() -> None:
    on_levels = LatentBlockModel(
        n_groups=3, n_blocks=2, response_model=OrdinalProbit(thresholds=Thresholds(axis="levels"))
    ).estimate(VOTES, seed=0)
    on_unit = LatentBlockModel(
        n_groups=3, n_blocks=2, response_model=OrdinalProbit(thresholds=Thresholds(axis="unit"))
    ).estimate(VOTES, seed=0)
    default = MODEL.estimate(VOTES, seed=0).value
    levels, unit = on_levels.value, on_unit.value
    assert np.array_equal(levels.group_index, default.group_index)
    assert np.array_equal(levels.group_index, unit.group_index)
    assert np.array_equal(levels.block_index, unit.block_index)
    assert unit.objective == pytest.approx(levels.objective, rel=1e-9)
    assert unit.integrated_likelihood == pytest.approx(levels.integrated_likelihood, rel=1e-9)
    assert levels.coclusters.latent_mean is not None
    assert unit.coclusters.latent_mean is not None
    assert levels.coclusters.latent_spread is not None
    assert unit.coclusters.latent_spread is not None
    # The unit axis is the axis of levels carried by x -> (x - 1/2) / K.
    assert unit.coclusters.latent_mean == pytest.approx((levels.coclusters.latent_mean - 0.5) / 5)
    assert unit.coclusters.latent_spread == pytest.approx(levels.coclusters.latent_spread / 5)
    assert on_unit.fingerprint() != on_levels.fingerprint()


def test_a_block_of_yes_no_questions_holds_its_spread_and_says_so() -> None:
    scales = [TWO] * 8 + [FIVE] * 8
    votes, _ = make_votes(
        single_memberships((30, 30)),
        single_memberships((8, 8)),
        vote_rate=[[0.9, 0.9], [0.9, 0.9]],
        latent_mean=[[0.8, 0.3], [0.2, 0.7]],
        latent_spread=[[0.15, 0.15], [0.15, 0.15]],
        item_scales=scales,
        seed=2,
    )
    value = LatentBlockModel(n_groups=2, n_blocks=2).estimate(votes, seed=0).value
    table = value.coclusters
    assert table.has_fixed_spread is not None
    assert table.latent_spread is not None
    binary_block = int(value.block_index[0])
    assert table.has_fixed_spread[:, binary_block].all()
    assert not table.has_fixed_spread[:, 1 - binary_block].any()
    assert table.latent_spread[:, binary_block] == pytest.approx([0.2, 0.2])
    assert np.isfinite(table.latent_mean).all()


def test_survey_weights_enter_a_computation_on_several_scales() -> None:
    votes, _ = mixed(1)
    weighted = VoteMatrix.from_arrays(
        votes.levels,
        item_scales=votes.item_scales,
        voter_weights=np.linspace(0.5, 2.0, votes.shape[0]),
    )
    plain = LatentBlockModel(n_groups=3, n_blocks=2).estimate(votes, seed=0).value
    value = LatentBlockModel(n_groups=3, n_blocks=2).estimate(weighted, seed=0).value
    assert value.objective != plain.objective
    assert value.coclusters.n_votes.sum() == pytest.approx(votes.n_votes)


SIX_ANSWERS = OrdinalScale(tuple(range(1, 7)))
PLANTED_GRID = (1 / 6, 0.25, 0.45, 0.7, 5 / 6)


def uneven(seed: int) -> VoteMatrix:
    """Items of six answers whose inner thresholds are not regular, and items of two."""
    votes, _ = make_votes(
        single_memberships((80, 80)),
        single_memberships((10, 10)),
        vote_rate=[[0.9, 0.9], [0.9, 0.9]],
        latent_mean=[[0.75, 0.35], [0.3, 0.65]],
        latent_spread=[[0.2, 0.2], [0.2, 0.2]],
        item_scales=[SIX_ANSWERS, TWO] * 10,
        thresholds={SIX_ANSWERS: PLANTED_GRID},
        seed=seed,
    )
    return votes


@pytest.mark.parametrize("seed", range(3))
def test_estimated_thresholds_find_a_planted_grid(seed: int) -> None:
    votes = uneven(seed)
    estimated = OrdinalProbit(thresholds=Thresholds(mode="estimated"))
    value = (
        LatentBlockModel(n_groups=2, n_blocks=2, response_model=estimated)
        .estimate(votes, seed=0)
        .value
    )
    assert value.coclusters.thresholds is not None
    grid = value.coclusters.thresholds[value.coclusters.scales.index(SIX_ANSWERS)]
    assert grid == pytest.approx(PLANTED_GRID, abs=0.03)
    assert (np.diff(grid) > 0).all()
    assert (np.diff(value.history.objectives) >= -1e-9).all()
    fixed = LatentBlockModel(n_groups=2, n_blocks=2).estimate(votes, seed=0).value
    # The regular grid misreads these answers: the estimate explains them better, even
    # after the penalty of its three inner thresholds.
    assert value.objective > fixed.objective
    assert value.integrated_likelihood > fixed.integrated_likelihood


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
    by_cocluster = LatentBlockModel(n_groups=2, n_blocks=1)
    shared = LatentBlockModel(
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
    assert found.integrated_likelihood > blind.integrated_likelihood
