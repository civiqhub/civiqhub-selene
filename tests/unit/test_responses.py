# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import math
from typing import Any

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene.blocks import CoclusterTable, OrdinalScale, VoteMatrix
from civiqhub_selene.blocks._numerics import level_probabilities
from civiqhub_selene.blocks.responses import (
    Categorical,
    CategoricalParameters,
    OrdinalProbit,
    OrdinalProbitParameters,
    ResponseModel,
    Thresholds,
    _vote_rate,
)
from civiqhub_selene.errors import InvalidInputError, InvalidParameterError
from civiqhub_selene.types import settings_of

pytestmark = pytest.mark.blocks

# One co-cluster: 10 cells shown without a vote and 30 votes.
COUNTS = np.array([[[10.0, 0.0, 5.0, 5.0, 0.0, 20.0]]])
VOTES = VoteMatrix.from_table([[2, 2, None, -1], [1, None, None, 0], [-2, -2, 2, None]])


def uniform(shape: tuple[int, int], n_levels: int = 5) -> CategoricalParameters:
    return CategoricalParameters(np.full(shape, 0.5), np.full((*shape, n_levels), 1 / n_levels))


def accepts(model: ResponseModel[Any]) -> ResponseModel[Any]:
    return model


def test_categorical_is_a_response_model_without_isinstance() -> None:
    # mypy --strict checks this call: Categorical satisfies the protocol structurally.
    model = accepts(Categorical())
    assert isinstance(model, Categorical)


def test_update_on_hand_written_counts() -> None:
    updated = Categorical(smoothing=0.0).update(COUNTS, uniform((1, 1)))
    assert updated.vote_rate.tolist() == [[0.75]]
    assert updated.vote_shares[0, 0].tolist() == pytest.approx([0, 1 / 6, 1 / 6, 0, 2 / 3])
    smoothed = Categorical().update(COUNTS, uniform((1, 1)))
    assert smoothed.vote_rate.tolist() == [[0.75]]
    assert smoothed.vote_shares[0, 0].tolist() == pytest.approx(
        [0.5 / 32.5, 5.5 / 32.5, 5.5 / 32.5, 0.5 / 32.5, 20.5 / 32.5]
    )


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (4, 1)])
@pytest.mark.parametrize("n_levels", [2, 5, 7])
def test_log_probabilities_sum_to_one_over_the_states(
    shape: tuple[int, int], n_levels: int
) -> None:
    rng = np.random.default_rng(1)
    model = Categorical()
    scale = OrdinalScale(tuple(range(n_levels)))
    votes = VoteMatrix.from_arrays(rng.integers(0, n_levels + 1, size=(6, 5)), scale=scale)
    params = model.initial_parameters(shape, votes, rng)
    assert params.vote_rate.shape == shape
    assert params.vote_shares.shape == (*shape, n_levels)
    logs = model.log_probabilities(params)
    assert logs.shape == (*shape, n_levels + 1)
    assert np.exp(logs).sum(axis=-1) == pytest.approx(np.ones(shape), abs=1e-9)
    counts = rng.random((*shape, n_levels + 1)) * 10
    updated = model.log_probabilities(model.update(counts, params))
    assert np.exp(updated).sum(axis=-1) == pytest.approx(np.ones(shape), abs=1e-9)


def test_initial_parameters_start_from_the_overall_vote_rate() -> None:
    model = Categorical()
    params = model.initial_parameters((2, 2), VOTES, np.random.default_rng(0))
    assert params.vote_rate.tolist() == [[8 / 12] * 2] * 2
    assert params.vote_shares.sum(axis=-1) == pytest.approx(np.ones((2, 2)))
    again = model.initial_parameters((2, 2), VOTES, np.random.default_rng(0))
    other = model.initial_parameters((2, 2), VOTES, np.random.default_rng(1))
    assert np.array_equal(params.vote_shares, again.vote_shares)
    assert not np.array_equal(params.vote_shares, other.vote_shares)
    # No vote and no shown cell: the rate is bounded, never 0.
    empty = model.initial_parameters((1, 1), VoteMatrix.from_records([]), np.random.default_rng(0))
    assert empty.vote_rate.tolist() == [[1e-6]]


def test_a_cocluster_without_any_cell_keeps_its_previous_parameters() -> None:
    counts = np.zeros((1, 2, 6))
    counts[0, 0] = COUNTS[0, 0]
    previous = CategoricalParameters(
        np.array([[0.2, 0.3]]),
        np.array([[[0.2] * 5, [0.1, 0.2, 0.3, 0.2, 0.2]]]),
    )
    updated = Categorical(smoothing=0.0).update(counts, previous)
    assert updated.vote_rate.tolist() == [[0.75, 0.3]]
    assert updated.vote_shares[0, 1].tolist() == [0.1, 0.2, 0.3, 0.2, 0.2]
    # With smoothing, a co-cluster without a vote goes to the uniform distribution.
    smoothed = Categorical(smoothing=0.5).update(counts, previous)
    assert smoothed.vote_rate.tolist() == [[0.75, 0.3]]
    assert smoothed.vote_shares[0, 1].tolist() == pytest.approx([0.2] * 5)
    # The previous parameters are not modified.
    assert previous.vote_rate.tolist() == [[0.2, 0.3]]


def test_the_vote_rate_is_bounded_away_from_zero_and_one() -> None:
    counts = np.array([[[0.0, 4.0, 0.0], [4.0, 0.0, 0.0]]])
    rate = _vote_rate(counts, np.full((1, 2), 0.5), 1e-3)
    assert rate.tolist() == [[1 - 1e-3, 1e-3]]


def test_update_accepts_fractional_counts_and_only_reads_proportions() -> None:
    model = Categorical(smoothing=0.0)
    counts = np.array([[[0.25, 0.5, 0.0, 0.125, 0.0, 0.125]]])
    small = model.update(counts, uniform((1, 1)))
    large = model.update(counts * 10, uniform((1, 1)))
    assert small.vote_rate.tolist() == large.vote_rate.tolist() == [[0.75]]
    assert small.vote_shares == pytest.approx(large.vote_shares)
    assert small.vote_shares[0, 0].tolist() == pytest.approx([2 / 3, 0, 1 / 6, 0, 1 / 6])


def test_complexity_penalty_follows_its_formula() -> None:
    model = Categorical()
    assert VOTES.n_votes == 8
    assert VOTES.n_shown == 12
    expected = 0.5 * 2 * 3 * 4 * math.log(8) + 0.5 * 2 * 3 * math.log(12)
    assert model.complexity_penalty((2, 3), VOTES) == pytest.approx(expected)
    assert model.complexity_penalty((1, 1), VoteMatrix.from_records([])) == 0.0


def test_to_table_carries_the_parameters_and_the_number_of_votes() -> None:
    model = Categorical(smoothing=0.0)
    params = model.update(COUNTS, uniform((1, 1)))
    table = model.to_table(params, COUNTS, OrdinalScale())
    assert isinstance(table, CoclusterTable)
    assert table.n_votes.tolist() == [[30.0]]
    assert table[0, 0].vote_rate == 0.75
    assert table[0, 0].typical_score == 2
    assert table[0, 0].latent_mean is None
    assert np.array_equal(table.log_probabilities(), model.log_probabilities(params))


def test_settings_enter_the_provenance_as_they_are() -> None:
    assert settings_of(Categorical(smoothing=1.0)) == (
        ("min_probability", 1e-6),
        ("smoothing", 1.0),
    )
    assert Categorical() == Categorical(smoothing=0.5, min_probability=1e-6)
    assert hash(Categorical()) == hash(Categorical())
    with pytest.raises(dataclasses.FrozenInstanceError):
        Categorical().smoothing = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("settings", "named"),
    [
        ({"smoothing": -0.1}, "smoothing"),
        ({"smoothing": float("nan")}, "smoothing"),
        ({"smoothing": float("inf")}, "smoothing"),
        ({"min_probability": 0.0}, "min_probability"),
        ({"min_probability": 0.5}, "min_probability"),
        ({"min_probability": -1e-6}, "min_probability"),
    ],
)
def test_settings_out_of_range_are_refused(settings: dict[str, float], named: str) -> None:
    with pytest.raises(InvalidParameterError, match=named):
        Categorical(**settings)


# OrdinalProbit ------------------------------------------------------------------------

THRESHOLDS = np.array([1.5, 2.5, 3.5, 4.5])


def probit_parameters(shape: tuple[int, int]) -> OrdinalProbitParameters:
    return OrdinalProbitParameters(
        np.full(shape, 0.5), np.full(shape, 3.0), np.ones(shape), THRESHOLDS
    )


def test_ordinal_probit_is_a_response_model_without_isinstance() -> None:
    model = accepts(OrdinalProbit())
    assert isinstance(model, OrdinalProbit)


def test_probit_initial_parameters() -> None:
    model = OrdinalProbit()
    params = model.initial_parameters((2, 3), VOTES, np.random.default_rng(0))
    assert params.thresholds.tolist() == [1.5, 2.5, 3.5, 4.5]
    assert params.vote_rate.tolist() == [[8 / 12] * 3] * 2
    assert params.latent_spread.tolist() == [[1.0] * 3] * 2
    assert params.latent_mean.shape == (2, 3)
    expected = 3.0 + np.random.default_rng(0).normal(0.0, 0.5, size=(2, 3))
    assert np.array_equal(params.latent_mean, expected)
    other = model.initial_parameters((2, 3), VOTES, np.random.default_rng(1))
    assert not np.array_equal(params.latent_mean, other.latent_mean)


def test_probit_thresholds_follow_the_scale_or_the_ones_given() -> None:
    three = VoteMatrix.from_table([[1, 3], [2, None]], scale=OrdinalScale((1, 2, 3)))
    rng = np.random.default_rng(0)
    default = OrdinalProbit().initial_parameters((1, 1), three, rng)
    assert default.thresholds.tolist() == [1.5, 2.5]
    assert default.latent_mean[0, 0] == pytest.approx(2.0, abs=2.5)
    given = OrdinalProbit(thresholds=(1.2, 2.9)).initial_parameters((1, 1), three, rng)
    assert given.thresholds.tolist() == [1.2, 2.9]
    with pytest.raises(InvalidParameterError, match="5 scores has 4 thresholds, got 2"):
        OrdinalProbit(thresholds=(1.2, 2.9)).initial_parameters((1, 1), VOTES, rng)


def test_probit_update_on_hand_written_counts() -> None:
    model = OrdinalProbit(max_inner_iterations=200)
    counts = np.array([[[10.0, 3.0, 9.0, 14.0, 6.0, 2.0]]])
    updated = model.update(counts, probit_parameters((1, 1)))
    assert updated.vote_rate.tolist() == [[34 / 44]]
    assert updated.latent_mean[0, 0] == pytest.approx(2.848198, abs=1e-5)
    assert updated.latent_spread[0, 0] == pytest.approx(0.998967, abs=1e-5)
    assert updated.thresholds is THRESHOLDS
    # The default takes ten steps at most from the previous parameters.
    ten = OrdinalProbit().update(counts, probit_parameters((1, 1)))
    one = OrdinalProbit(max_inner_iterations=1).update(counts, probit_parameters((1, 1)))
    assert abs(ten.latent_mean[0, 0] - 2.848198) < abs(one.latent_mean[0, 0] - 2.848198)


def test_probit_update_accepts_fractional_counts_and_only_reads_proportions() -> None:
    model = OrdinalProbit(max_inner_iterations=100)
    counts = np.array([[[0.25, 0.05, 0.125, 0.4, 0.25, 0.075]]])
    small = model.update(counts, probit_parameters((1, 1)))
    large = model.update(counts * 100, probit_parameters((1, 1)))
    assert small.vote_rate == pytest.approx(large.vote_rate)
    assert small.latent_mean == pytest.approx(large.latent_mean, abs=1e-7)
    assert small.latent_spread == pytest.approx(large.latent_spread, abs=1e-7)


def test_a_unanimous_cocluster_stops_at_the_minimal_spread() -> None:
    counts = np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, 25.0]]])
    updated = OrdinalProbit(max_inner_iterations=500).update(counts, probit_parameters((1, 1)))
    assert updated.latent_spread.tolist() == [[0.3]]
    assert np.isfinite(updated.latent_mean).all()
    assert updated.vote_rate.tolist() == [[1 - 1e-6]]
    wider = OrdinalProbit(min_spread=0.5, max_inner_iterations=500).update(
        counts, probit_parameters((1, 1))
    )
    assert wider.latent_spread.tolist() == [[0.5]]


def test_a_probit_cocluster_without_any_cell_keeps_its_previous_parameters() -> None:
    previous = OrdinalProbitParameters(
        np.array([[0.2, 0.3]]), np.array([[2.5, 4.0]]), np.array([[0.75, 1.5]]), THRESHOLDS
    )
    counts = np.zeros((1, 2, 6))
    counts[0, 0] = [10.0, 3.0, 9.0, 14.0, 6.0, 2.0]
    updated = OrdinalProbit().update(counts, previous)
    assert updated.vote_rate[0, 1] == 0.3
    assert updated.latent_mean[0, 1] == 4.0
    assert updated.latent_spread[0, 1] == 1.5
    assert updated.latent_mean[0, 0] != 2.5
    assert previous.latent_mean.tolist() == [[2.5, 4.0]]


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (4, 1)])
def test_probit_log_probabilities_sum_to_one_over_the_states(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(2)
    model = OrdinalProbit()
    params = model.initial_parameters(shape, VOTES, rng)
    logs = model.log_probabilities(params)
    assert logs.shape == (*shape, 6)
    assert np.exp(logs).sum(axis=-1) == pytest.approx(np.ones(shape), abs=1e-9)
    updated = model.update(rng.random((*shape, 6)) * 20, params)
    assert np.exp(model.log_probabilities(updated)).sum(axis=-1) == pytest.approx(
        np.ones(shape), abs=1e-9
    )
    # Far levels are floored at 1e-12, so that one vote never costs an infinite amount.
    tight = OrdinalProbitParameters(
        np.array([[0.5]]), np.array([[9.0]]), np.array([[0.3]]), THRESHOLDS
    )
    assert model.log_probabilities(tight)[0, 0, 1] == pytest.approx(np.log(0.5 * 1e-12))


def test_probit_complexity_penalty_follows_its_formula() -> None:
    expected = 2 * 3 * math.log(8) + 0.5 * 2 * 3 * math.log(12)
    assert OrdinalProbit().complexity_penalty((2, 3), VOTES) == pytest.approx(expected)
    assert OrdinalProbit().complexity_penalty((2, 3), VOTES) < Categorical().complexity_penalty(
        (2, 3), VOTES
    )


def test_probit_to_table_carries_the_latent_parameters() -> None:
    model = OrdinalProbit()
    params = OrdinalProbitParameters(
        np.array([[0.8]]), np.array([[4.7]]), np.array([[0.6]]), THRESHOLDS
    )
    counts = np.array([[[5.0, 0.0, 0.0, 1.0, 4.0, 15.0]]])
    table = model.to_table(params, counts, OrdinalScale())
    cocluster = table[0, 0]
    assert cocluster.latent_mean == 4.7
    assert cocluster.latent_spread == 0.6
    assert cocluster.n_votes == 20.0
    assert cocluster.vote_rate == 0.8
    assert cocluster.typical_score == 2
    assert cocluster.vote_shares == pytest.approx(
        np.maximum(level_probabilities(4.7, 0.6, THRESHOLDS), 1e-12)
    )
    assert np.array_equal(table.log_probabilities(), model.log_probabilities(params))


def test_probit_settings_enter_the_provenance_as_they_are() -> None:
    assert settings_of(OrdinalProbit()) == (
        ("max_inner_iterations", 10),
        ("min_probability", 1e-6),
        ("min_spread", 0.3),
        ("thresholds", None),
    )
    assert settings_of(OrdinalProbit(thresholds=[1, 2.5]))[3] == ("thresholds", (1.0, 2.5))  # type: ignore[arg-type]
    assert OrdinalProbit() == OrdinalProbit(min_spread=0.3)
    assert hash(OrdinalProbit()) == hash(OrdinalProbit())


@pytest.mark.parametrize(
    ("settings", "named"),
    [
        ({"min_spread": 0.0}, "min_spread"),
        ({"min_spread": -1.0}, "min_spread"),
        ({"min_spread": float("nan")}, "min_spread"),
        ({"min_probability": 0.5}, "min_probability"),
        ({"max_inner_iterations": 0}, "max_inner_iterations"),
        ({"max_inner_iterations": 2.5}, "max_inner_iterations"),
        ({"max_inner_iterations": True}, "max_inner_iterations"),
        ({"thresholds": (2.5, 1.5)}, "strictly increasing"),
        ({"thresholds": (1.5, 1.5)}, "strictly increasing"),
        ({"thresholds": (1.5, float("inf"))}, "finite"),
        ({"thresholds": ()}, "thresholds"),
        ({"thresholds": ("a", "b")}, "thresholds are numbers"),
    ],
)
def test_probit_settings_out_of_range_are_refused(settings: dict[str, object], named: str) -> None:
    with pytest.raises(InvalidParameterError, match=named):
        OrdinalProbit(**settings)  # type: ignore[arg-type]


# Several scales -----------------------------------------------------------------------
# TEST ONLY. Synthetic counts on a scale of two answers and a scale of four.

YES_NO = OrdinalScale((0, 1))
FOUR = OrdinalScale((1, 2, 3, 4))
SURVEY = VoteMatrix.from_responses(
    [[1, 4, 0], [0, 2, None], [1, 1, 1], [None, 3, 1]],
    item_scales={"q1": YES_NO, "q2": FOUR, "q3": YES_NO},
)


def test_the_axis_of_the_thresholds_follows_the_number_of_scales() -> None:
    rule = Thresholds()
    assert rule.axis_for(VOTES) == "levels"
    assert rule.axis_for(SURVEY) == "unit"
    assert rule.grid_of(FOUR, "levels") == (1.5, 2.5, 3.5)
    assert rule.grid_of(FOUR, "unit") == (0.25, 0.5, 0.75)
    assert rule.grid_of(YES_NO, "unit") == (0.5,)
    assert Thresholds(axis="levels").axis_for(SURVEY) == "levels"


def test_grids_given_by_hand_replace_the_regular_ones() -> None:
    rule = Thresholds(axis="unit", grids={FOUR: (0.2, 0.5, 0.8)})
    assert rule.grid_of(FOUR, "unit") == (0.2, 0.5, 0.8)
    assert rule.grid_of(YES_NO, "unit") == (0.5,)
    assert rule.grids == ((FOUR, (0.2, 0.5, 0.8)),)
    assert OrdinalProbit(thresholds={FOUR: (0.2, 0.5, 0.8)}).thresholds == Thresholds(
        grids={FOUR: (0.2, 0.5, 0.8)}
    )
    settings = dict(settings_of(OrdinalProbit(thresholds=rule)))
    assert settings["thresholds"][0] == ("type", "Thresholds")


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"axis": "log"}, "'auto', 'levels' or 'unit'"),
        ({"grids": {FOUR: (0.5, 0.2, 0.8)}}, "strictly increasing"),
        ({"grids": {FOUR: (0.5,)}}, "has 3 thresholds"),
        ({"grids": {(1, 2): (0.5,)}}, "keyed by OrdinalScale"),
        ({"grids": ((FOUR, (0.2, 0.5, 0.8)), (FOUR, (0.3, 0.5, 0.7)))}, "twice"),
    ],
)
def test_inconsistent_thresholds_are_refused(changes: dict[str, Any], named: str) -> None:
    with pytest.raises(InvalidParameterError, match=named):
        Thresholds(**changes)


def test_the_probit_starts_on_the_unit_axis_when_items_use_several_scales() -> None:
    params = OrdinalProbit().initial_parameters((2, 2), SURVEY, np.random.default_rng(0))
    assert [grid.tolist() for grid in params.thresholds_by_scale] == [[0.5], [0.25, 0.5, 0.75]]
    assert params.spread_unit == 0.25
    assert params.latent_spread.tolist() == [[0.25, 0.25], [0.25, 0.25]]
    expected = 0.5 + np.random.default_rng(0).normal(0.0, 0.5, size=(2, 2)) / 4
    assert np.array_equal(params.latent_mean, expected)
    with pytest.raises(InvalidInputError, match="2 scales, not one"):
        _ = params.thresholds
    with pytest.raises(InvalidParameterError, match="grid of one scale"):
        OrdinalProbit(thresholds=(1.5,)).initial_parameters(
            (1, 1), SURVEY, np.random.default_rng(0)
        )


def test_each_vote_is_cut_by_the_grid_of_its_scale() -> None:
    model = OrdinalProbit()
    params = OrdinalProbitParameters(
        np.array([[0.8]]),
        np.array([[0.6]]),
        np.array([[0.2]]),
        (np.array([0.5]), np.array([0.25, 0.5, 0.75])),
        spread_unit=0.25,
    )
    logs = model.log_probabilities_by_scale(params)
    assert logs.shape == (1, 1, 2, 5)
    probabilities = np.exp(logs[0, 0])
    yes = level_probabilities(0.6, 0.2, [0.5])[1]
    assert probabilities[0, :3].tolist() == pytest.approx([0.2, 0.8 * (1 - yes), 0.8 * yes])
    assert probabilities[0, 3:].max() < 1e-299
    assert probabilities[1].sum() == pytest.approx(1.0)
    assert probabilities[1, 1:].tolist() == pytest.approx(
        (0.8 * level_probabilities(0.6, 0.2, [0.25, 0.5, 0.75])).tolist()
    )
    with pytest.raises(InvalidInputError, match="2 scales"):
        model.log_probabilities(params)


def test_the_opinion_is_one_for_the_votes_of_every_scale() -> None:
    # A co-cluster votes yes 30 times out of 40 on the scale of two answers, and mostly
    # at the top of the scale of four: one opinion near 0.6 explains both.
    counts = np.zeros((1, 1, 2, 5))
    counts[0, 0, 0, :3] = [5, 10, 30]
    counts[0, 0, 1, :] = [5, 2, 6, 14, 18]
    start = OrdinalProbitParameters(
        np.array([[0.5]]),
        np.array([[0.5]]),
        np.array([[0.25]]),
        (np.array([0.5]), np.array([0.25, 0.5, 0.75])),
        spread_unit=0.25,
    )
    fitted = OrdinalProbit(max_inner_iterations=500).update(counts, start)
    assert float(fitted.vote_rate[0, 0]) == pytest.approx(80 / 90)
    mean, spread = float(fitted.latent_mean[0, 0]), float(fitted.latent_spread[0, 0])
    assert 0.55 < mean < 0.75
    assert 0.1 < spread < 0.4
    # Alone, the votes of the scale of four give another opinion: the two scales pool.
    alone = counts.copy()
    alone[0, 0, 0] = 0
    on_four = OrdinalProbit(max_inner_iterations=500).update(alone, start)
    assert float(on_four.latent_mean[0, 0]) != pytest.approx(mean)


def test_votes_on_two_answers_only_hold_the_spread_at_one_level() -> None:
    counts = np.zeros((1, 2, 2, 5))
    counts[0, 0, 0, :3] = [2, 8, 32]  # yes and no only
    counts[0, 1, 0, :3] = [2, 8, 32]
    counts[0, 1, 1, :] = [0, 1, 2, 3, 4]  # a few votes on four answers
    start = OrdinalProbitParameters(
        np.full((1, 2), 0.5),
        np.full((1, 2), 0.5),
        np.full((1, 2), 0.1),
        (np.array([0.5]), np.array([0.25, 0.5, 0.75])),
        spread_unit=0.25,
    )
    fitted = OrdinalProbit(max_inner_iterations=500).update(counts, start)
    assert fitted.latent_spread[0, 0] == 0.25
    assert fitted.latent_spread[0, 1] != 0.25
    # With the spread held, the mean puts 32 of 40 votes above 1/2.
    assert level_probabilities(fitted.latent_mean[0, 0], 0.25, [0.5])[1] == pytest.approx(0.8)
    table = OrdinalProbit().to_table(fitted, counts, (YES_NO, FOUR))
    assert table.has_fixed_spread is not None
    assert table.has_fixed_spread.tolist() == [[True, False]]
    assert table[0, 0].has_fixed_spread
    assert table.to_dict()["has_fixed_spread"] == [[True, False]]


def test_the_categorical_model_has_one_free_distribution_per_scale() -> None:
    model = Categorical(smoothing=0.0)
    params = model.initial_parameters((1, 2), SURVEY, np.random.default_rng(3))
    assert [array.shape for array in params.vote_shares_by_scale] == [(1, 2, 2), (1, 2, 4)]
    first = np.random.default_rng(3).dirichlet(np.full(2, 5.0), size=(1, 2))
    assert np.array_equal(params.vote_shares_by_scale[0], first)
    counts = np.zeros((1, 2, 2, 5))
    counts[0, 0, 0, :3] = [1, 3, 6]
    counts[0, 0, 1, :] = [0, 0, 1, 1, 2]
    updated = model.update(counts, params)
    assert updated.vote_shares_by_scale[0][0, 0].tolist() == pytest.approx([1 / 3, 2 / 3])
    assert updated.vote_shares_by_scale[1][0, 0].tolist() == pytest.approx([0, 0.25, 0.25, 0.5])
    # No vote in the second co-cluster: it keeps its distributions.
    assert np.array_equal(
        updated.vote_shares_by_scale[1][0, 1], params.vote_shares_by_scale[1][0, 1]
    )
    logs = model.log_probabilities_by_scale(updated)
    assert logs.shape == (1, 2, 2, 5)
    assert np.exp(logs[0, 0, 0, :3]).sum() == pytest.approx(1.0)
    expected = 0.5 * 2 * (1 + 3) * math.log(SURVEY.n_votes) + 0.5 * 2 * math.log(SURVEY.n_shown)
    assert model.complexity_penalty((1, 2), SURVEY) == pytest.approx(expected)
    table = model.to_table(updated, counts, (YES_NO, FOUR))
    assert table.vote_shares_by_scale[1][0, 0].tolist() == pytest.approx([0, 0.25, 0.25, 0.5])


def test_one_scale_counts_are_read_with_or_without_the_axis_of_scales() -> None:
    model = OrdinalProbit()
    params = probit_parameters((1, 1))
    assert np.array_equal(
        model.update(COUNTS, params).latent_mean,
        model.update(COUNTS[:, :, np.newaxis, :], params).latent_mean,
    )
    with pytest.raises(InvalidInputError, match="groups, blocks, scales, states"):
        model.update(COUNTS[0], params)


def test_estimated_thresholds_move_the_inner_thresholds_only() -> None:
    six = OrdinalScale(tuple(range(1, 7)))
    planted = np.array([1 / 6, 0.25, 0.45, 0.7, 5 / 6])
    mean, spread = np.array([[0.3, 0.7]]), np.array([[0.2, 0.2]])
    counts = np.zeros((1, 2, 1, 7))
    counts[..., 0, 1:] = 500 * level_probabilities(mean, spread, planted)
    start = OrdinalProbitParameters(
        np.full((1, 2), 0.9),
        mean,
        spread,
        (np.array([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]),),
        spread_unit=1 / 6,
    )
    model = OrdinalProbit(thresholds=Thresholds(mode="estimated"), max_inner_iterations=50)
    # Each update moves the opinions, then the grid at those opinions: a few updates
    # reach the joint maximum, which the planted grid gives on these exact counts.
    updated = start
    for _ in range(20):
        updated = model.update(counts, updated)
    grid = updated.thresholds_by_scale[0]
    assert grid[0] == 1 / 6
    assert grid[-1] == 5 / 6
    assert grid == pytest.approx(planted, abs=0.02)
    fixed = OrdinalProbit(max_inner_iterations=50).update(counts, start)
    assert np.array_equal(fixed.thresholds_by_scale[0], start.thresholds_by_scale[0])
    votes = VoteMatrix.from_arrays([[1, 6, 3], [2, 4, 5]], scale=six)
    plain = OrdinalProbit().complexity_penalty((2, 2), votes)
    assert model.complexity_penalty((2, 2), votes) == pytest.approx(
        plain + 0.5 * 3 * math.log(votes.n_votes)
    )
    with pytest.raises(InvalidParameterError, match="'fixed' or 'estimated'"):
        Thresholds(mode="free")  # type: ignore[arg-type]
    assert Thresholds(mode="estimated").is_estimated
    assert not Thresholds().is_estimated
