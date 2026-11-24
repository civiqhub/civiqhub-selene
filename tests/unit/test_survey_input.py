# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

from io import StringIO

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene.blocks import (
    LatentBlockModel,
    MixedMembershipBlockModel,
    OrdinalScale,
    VoteMatrix,
    VoterCovariates,
    read_survey_csv,
)
from civiqhub_selene.blocks.responses import Categorical
from civiqhub_selene.errors import InvalidInputError

pytestmark = pytest.mark.blocks


def test_survey_responses_use_declared_scales_and_preserve_the_shown_mask() -> None:
    three = OrdinalScale((-10, 0, 10))
    five = OrdinalScale((10, 20, 30, 40, 50))
    votes = VoteMatrix.from_responses(
        [[-10, 10], [0, 20], [10, None], [None, float("nan")]],
        voters=["a", "b", "c", "d"],
        items=["short", "long"],
        item_scales={"long": five, "short": three},
        shown=[[True, True], [True, False], [True, True], [False, True]],
        voter_weights=[1, 2, 3, 4],
    )
    np.testing.assert_array_equal(votes.levels, [[1, 1], [3, 2], [5, 0], [0, 0]])
    assert votes.scale.scores == (1, 2, 3, 4, 5)
    assert (votes.n_shown, votes.n_votes) == (6, 4)
    np.testing.assert_array_equal(votes[["b", "d"], :].voter_weights, [2, 4])
    np.testing.assert_array_equal(votes.hold_out(votes.shown).voter_weights, [1, 2, 3, 4])
    assert votes != VoteMatrix.from_arrays(
        votes.levels, shown=votes.shown, voters=votes.voters, items=votes.items, scale=votes.scale
    )


def test_survey_responses_validate_source_and_target_scales() -> None:
    three = OrdinalScale((1, 2, 3))
    four = OrdinalScale((10, 20, 30, 40))
    with pytest.raises(InvalidInputError, match="exactly the response table items"):
        VoteMatrix.from_responses([[1]], items=["a"], item_scales={"other": three})
    with pytest.raises(InvalidInputError, match="score 4 is not on the scale"):
        VoteMatrix.from_responses([[4]], item_scales={"short": three}, scale=four)
    with pytest.raises(InvalidInputError, match="at least 4 levels"):
        VoteMatrix.from_responses([[20]], item_scales={"long": four}, scale=three)
    votes = VoteMatrix.from_responses([[2]], item_scales={"short": three}, scale=four)
    assert votes.levels.tolist() == [[3]]


@pytest.mark.parametrize("weights", [[], [0], [-1], [float("nan")], [float("inf")], [True]])
def test_survey_weights_are_positive_finite_and_aligned(weights: list[float]) -> None:
    with pytest.raises(InvalidInputError, match="voter_weights"):
        VoteMatrix.from_responses([[1]], voter_weights=weights)


def test_voter_covariates_align_by_identifier() -> None:
    votes = VoteMatrix.from_table([[1], [1]], voters=["b", "a"])
    covariates = VoterCovariates.from_records(
        votes,
        {"a": {"mode": "web", "age": None}, "b": {"mode": "phone", "age": 41}},
    )
    assert covariates.voters == votes.voters
    assert covariates.columns == {"mode": ("phone", "web"), "age": (41, None)}
    with pytest.raises(InvalidInputError, match="voter IDs"):
        VoterCovariates.from_records(votes, {"a": {"mode": "web"}})
    with pytest.raises(InvalidInputError, match="same fields"):
        VoterCovariates.from_records(votes, {"a": {"mode": "web"}, "b": {"age": 41}})


@pytest.mark.parametrize("model_type", [LatentBlockModel, MixedMembershipBlockModel])
def test_survey_weights_change_the_fit_but_not_the_number_of_observations(
    model_type: type[LatentBlockModel] | type[MixedMembershipBlockModel],
) -> None:
    scale = OrdinalScale((1, 2))
    scores = [[1, 1], [2, 2]]
    model = model_type(n_groups=1, n_blocks=1, response_model=Categorical(smoothing=0), n_starts=1)
    plain = model.estimate(VoteMatrix.from_responses(scores, scale=scale), seed=7).value
    weighted = model.estimate(
        VoteMatrix.from_responses(scores, scale=scale, voter_weights=[1, 3]), seed=7
    ).value
    scaled = model.estimate(
        VoteMatrix.from_responses(scores, scale=scale, voter_weights=[10, 30]), seed=7
    ).value
    assert plain.coclusters.vote_shares[0, 0] == pytest.approx([0.5, 0.5])
    assert weighted.coclusters.vote_shares[0, 0] == pytest.approx([0.25, 0.75])
    assert weighted.coclusters.n_votes[0, 0] == pytest.approx(4)
    assert scaled.objective == pytest.approx(weighted.objective)
    rate = weighted.coclusters.vote_rate[0, 0]
    expected = np.log(0.25) + 3 * np.log(0.75) + 4 * np.log(rate)
    assert weighted.objective == pytest.approx(expected)


def test_csv_loads_mixed_scales_weights_and_aligned_covariates() -> None:
    csv = StringIO(
        '\ufeffresponseid;Q2;Q1;POIDS;mode\n001;5;2;1,5;"phone; landline"\n002;;NA;0,5;\n'
    )
    matrix, covariates = read_survey_csv(
        csv,
        item_scales={
            "Q1": OrdinalScale((1, 2, 3)),
            "Q2": OrdinalScale((1, 2, 3, 4, 5)),
        },
        delimiter=";",
        decimal=",",
        weight_column="POIDS",
        covariate_columns=("mode",),
        unshown_label="NA",
    )
    assert matrix.voters == ("001", "002")
    assert matrix.items == ("Q1", "Q2")
    assert matrix.levels.tolist() == [[3, 5], [0, 0]]
    assert matrix.shown.tolist() == [[True, True], [False, True]]
    np.testing.assert_array_equal(matrix.voter_weights, [1.5, 0.5])
    assert covariates.voters == matrix.voters
    assert covariates.columns["mode"] == ("phone; landline", None)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("", "must have a header"),
        ("responseid,responseid,Q\na,b,1\n", "column names must be unique"),
        ("responseid,other\na,1\n", "missing columns"),
        ("responseid,Q\na,1,extra\n", "has 3 fields"),
        ("responseid,Q\n,2\n", "empty voter identifier"),
        ("responseid,Q\na,one\n", "integer score"),
        ("responseid,Q\na,3\n", "score 3 is not on the scale"),
        ("responseid,Q\na,1\na,2\n", "identifier 'a' is repeated"),
    ],
)
def test_csv_rejects_invalid_survey_rows_before_fitting(rows: str, message: str) -> None:
    with pytest.raises(InvalidInputError, match=message):
        read_survey_csv(StringIO(rows), item_scales={"Q": OrdinalScale((1, 2))})
