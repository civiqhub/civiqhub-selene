# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene.blocks import (
    LatentBlockModel,
    MixedMembershipBlockModel,
    ShapeSelection,
    SpectralStart,
    VoteMatrix,
    select_shape,
)
from civiqhub_selene.blocks.responses import Categorical
from civiqhub_selene.blocks.results import MixedCoclustering, SingleCoclustering
from civiqhub_selene.blocks.selection import CRITERIA
from civiqhub_selene.blocks.synthetic import make_votes, single_memberships
from civiqhub_selene.errors import InvalidParameterError
from civiqhub_selene.testing import EstimatorContract
from civiqhub_selene.types import UnavailableReason, settings_of

pytestmark = pytest.mark.blocks

VOTES, _ = make_votes(single_memberships((20, 16, 12)), single_memberships((7, 7)), seed=0)
SPECTRAL = LatentBlockModel(n_starts=2, initialization=SpectralStart())
MIXED_VOTES, _ = make_votes(
    np.vstack([single_memberships((15, 15, 15)), [[0.5, 0.5, 0.0]] * 8]),
    single_memberships((6, 6)),
    seed=1,
)


# The integrated likelihood --------------------------------------------------------------


def test_the_integrated_likelihood_finds_the_planted_shape() -> None:
    result = select_shape(VOTES, SPECTRAL, seed=0)
    value = result.value
    selection = value.selection
    assert isinstance(value, SingleCoclustering)
    assert selection is not None
    assert selection.criterion == "integrated_likelihood"
    assert selection.best == (3, 2)
    assert (value.n_groups, value.n_blocks) == (3, 2)
    assert selection.n_groups == (1, 2, 3, 4)
    assert selection.n_blocks == (1, 2, 3)
    assert selection.scores.shape == (4, 3)
    assert np.isfinite(selection.scores).all()
    assert selection.standard_errors is None
    assert selection.alternatives == ()
    best = selection.scores[2, 1]
    assert best == float(np.max(selection.scores))
    assert value.integrated_likelihood == pytest.approx(best)


def test_each_score_is_the_criterion_of_the_model_at_that_shape() -> None:
    selection = select_shape(VOTES, SPECTRAL, n_groups=(2, 3), n_blocks=(2,)).value.selection
    assert selection is not None
    for row, n_groups in enumerate((2, 3)):
        alone = SPECTRAL.with_shape(n_groups, 2).estimate(VOTES, seed=0).value
        assert selection.scores[row, 0] == pytest.approx(alone.integrated_likelihood)


def test_the_value_cites_the_selection_not_the_inner_estimation() -> None:
    result = select_shape(VOTES, SPECTRAL, n_groups=(2, 3), n_blocks=(2,))
    value = result.value
    assert value.computation == result.fingerprint()
    assert value.groups.computation == result.fingerprint()
    inner = SPECTRAL.with_shape(3, 2).estimate(VOTES, seed=0)
    assert inner.fingerprint() != result.fingerprint()
    assert value.objective == inner.value.objective
    data = value.to_dict()
    assert data["selection"]["best"] == [3, 2]
    assert data["selection"]["criterion"] == "integrated_likelihood"


# The held-out likelihood --------------------------------------------------------------


def test_the_held_out_likelihood_finds_the_planted_shape_of_the_mixed_model() -> None:
    model = MixedMembershipBlockModel(n_starts=1, initialization=SpectralStart())
    result = select_shape(MIXED_VOTES, model, n_groups=(2, 3), n_blocks=(1, 2), n_folds=3)
    value = result.value
    selection = value.selection
    assert isinstance(value, MixedCoclustering)
    assert selection is not None
    assert selection.criterion == "held_out_likelihood"
    assert selection.best == (3, 2)
    assert selection.standard_errors is not None
    assert (selection.standard_errors > 0).all()
    # Every alternative lies within one standard error of the best score.
    floor = selection.scores[1, 1] - selection.standard_errors[1, 1]
    for n_groups, n_blocks in selection.alternatives:
        row, column = (2, 3).index(n_groups), (1, 2).index(n_blocks)
        assert selection.scores[row, column] >= floor
    # A score is a mean log-probability per cell: negative, above log(1 / (K + 1)) times a few.
    assert (selection.scores < 0).all()
    assert (selection.scores > -5).all()


def test_the_held_out_score_follows_its_definition() -> None:
    selection = select_shape(
        VOTES, SPECTRAL, n_groups=(3,), n_blocks=(2,), criterion="held_out_likelihood", n_folds=3
    ).value.selection
    assert selection is not None
    folds = np.random.default_rng(0).integers(0, 3, size=VOTES.shape)
    per_fold = []
    for fold in range(3):
        held = VOTES.shown & (folds == fold)
        value = SPECTRAL.with_shape(3, 2).estimate(VOTES.hold_out(held), seed=0).value
        per_fold.append(value.log_probability(VOTES)[held].mean())
    assert selection.scores[0, 0] == pytest.approx(np.mean(per_fold))
    assert selection.standard_errors is not None
    assert selection.standard_errors[0, 0] == pytest.approx(np.std(per_fold, ddof=1) / math.sqrt(3))


def test_the_default_criterion_is_the_one_of_the_model() -> None:
    assert ShapeSelection().effective_criterion == "integrated_likelihood"
    assert ShapeSelection(model=MixedMembershipBlockModel()).effective_criterion == (
        "held_out_likelihood"
    )
    assert set(CRITERIA) == {"integrated_likelihood", "held_out_likelihood"}


def test_the_integrated_likelihood_is_refused_for_the_mixed_model() -> None:
    selection = ShapeSelection(
        model=MixedMembershipBlockModel(n_starts=1, max_iterations=20),
        n_groups=(2,),
        n_blocks=(2,),
        criterion="integrated_likelihood",
    )
    with pytest.raises(InvalidParameterError, match="held_out_likelihood"):
        selection.estimate(MIXED_VOTES)


# The greedy search --------------------------------------------------------------------


def test_the_greedy_search_visits_a_path_and_finds_the_same_best_shape() -> None:
    grid = select_shape(VOTES, SPECTRAL).value.selection
    greedy = select_shape(VOTES, SPECTRAL, search="greedy").value.selection
    assert grid is not None
    assert greedy is not None
    assert greedy.best == grid.best == (3, 2)
    visited = np.isfinite(greedy.scores)
    assert visited.sum() < visited.size
    assert visited[0, 0]
    # The shapes it visits get the same score as in the full grid.
    assert greedy.scores[visited] == pytest.approx(grid.scores[visited])


# Absences and settings ----------------------------------------------------------------


def test_a_shape_without_enough_data_scores_minus_infinity() -> None:
    small = VoteMatrix.from_arrays([[5, 1, 4], [5, 2, 4], [1, 5, 2], [2, 5, 1]])
    selection = select_shape(small, SPECTRAL, n_groups=(2, 5), n_blocks=(1, 2)).value.selection
    assert selection is not None
    assert selection.scores[1].tolist() == [-math.inf, -math.inf]
    assert np.isfinite(selection.scores[0]).all()
    assert selection.to_dict()["scores"][1] == [None, None]


def test_without_any_possible_shape_the_result_is_an_explained_absence() -> None:
    small = VoteMatrix.from_arrays([[5, 1], [5, 2], [1, 5]])
    absent = select_shape(small, SPECTRAL, n_groups=(4, 5), n_blocks=(1,))
    assert not absent.is_available
    assert absent.unavailable is not None
    assert absent.unavailable.reason == UnavailableReason.NOT_ENOUGH_DATA
    empty = select_shape(VoteMatrix.from_records([], voters=["a", "b"], items=["x", "y"]))
    assert empty.unavailable is not None
    assert empty.unavailable.reason == UnavailableReason.NO_VOTES


def test_the_settings_enter_the_provenance() -> None:
    selection = ShapeSelection(
        model=SPECTRAL,
        n_groups=(3, 2, 2),
        n_blocks=[2],  # type: ignore[arg-type]
        n_folds=3,
    )
    assert selection.n_groups == (2, 3)
    assert selection.n_blocks == (2,)
    settings = dict(settings_of(selection))
    assert settings["n_folds"] == 3
    assert dict(settings["model"])["type"] == "LatentBlockModel"
    result = selection.estimate(VOTES, seed=0)
    assert result.provenance.parameters == settings_of(selection)


def test_a_plain_word_gives_the_default_model() -> None:
    single = select_shape(VOTES, "single", n_groups=(3,), n_blocks=(2,))
    assert isinstance(single.value, SingleCoclustering)
    assert dict(single.provenance.parameters)["model"][0] == ("type", "LatentBlockModel")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"model": "single"}, "model"),
        ({"n_groups": ()}, "n_groups"),
        ({"n_groups": (0, 2)}, "n_groups"),
        ({"n_blocks": "12"}, "n_blocks"),
        ({"n_blocks": (1.0,)}, "n_blocks"),
        ({"criterion": "bic"}, "criterion"),
        ({"n_folds": 1}, "n_folds"),
        ({"search": "random"}, "search"),
        ({"model": MixedMembershipBlockModel(), "n_groups": (5,), "n_blocks": (6,)}, "profiles"),
    ],
)
def test_settings_out_of_range_are_refused(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(InvalidParameterError, match=message):
        ShapeSelection(**changes)


def test_a_model_with_another_response_model_is_selected_the_same_way() -> None:
    model = LatentBlockModel(response_model=Categorical(), initialization=SpectralStart())
    selection = select_shape(VOTES, model, n_groups=(2, 3), n_blocks=(2,)).value.selection
    assert selection is not None
    assert selection.best[1] == 2


class TestShapeSelection(EstimatorContract):
    def make_estimator(self) -> ShapeSelection:
        return ShapeSelection(model=SPECTRAL, n_groups=(2, 3), n_blocks=(1, 2))

    def make_data(self) -> VoteMatrix:
        return VOTES

    def make_empty_data(self) -> VoteMatrix:
        return VoteMatrix.from_records([], voters=["a", "b"], items=["x", "y"])
