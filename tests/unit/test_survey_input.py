# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import ast
import re
from io import StringIO
from pathlib import Path

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene.blocks import (
    FixedStart,
    LatentBlockModel,
    MixedMembershipBlockModel,
    OrdinalScale,
    VoteMatrix,
    VoterCovariates,
    read_survey_csv,
)
from civiqhub_selene.blocks.covariates import VoterCovariates as CovariatesFromModule
from civiqhub_selene.blocks.responses import Categorical
from civiqhub_selene.errors import InvalidInputError

pytestmark = pytest.mark.blocks

# TEST ONLY. Synthetic scores and seed 7; not a real survey.


def test_homogeneous_from_responses_keeps_original_ranks_and_matches_from_records() -> None:
    scale = OrdinalScale((-10, 0, 10))
    votes = VoteMatrix.from_responses(
        [[-10, 10], [0, None], [None, float("nan")]],
        voters=["a", "b", "c"],
        items=["left", "right"],
        item_scales={"left": scale, "right": scale},
        shown=[[True, True], [True, True], [False, True]],
    )
    np.testing.assert_array_equal(votes.levels, [[1, 3], [2, 0], [0, 0]])
    assert votes.scale.scores == (-10, 0, 10)
    records = VoteMatrix.from_records(
        [("a", "left", -10), ("a", "right", 10), ("b", "left", 0)],
        shown=[("a", "left"), ("a", "right"), ("b", "left"), ("b", "right"), ("c", "right")],
        voters=["a", "b", "c"],
        items=["left", "right"],
        scale=scale,
    )
    assert votes == records
    weighted = VoteMatrix.from_responses(
        [[-10, 10], [0, None], [None, float("nan")]],
        voters=["a", "b", "c"],
        items=["left", "right"],
        item_scales={"left": scale, "right": scale},
        shown=[[True, True], [True, True], [False, True]],
        voter_weights=[1, 2, 3],
    )
    np.testing.assert_array_equal(weighted[["b", "c"], :].voter_weights, [2, 3])
    np.testing.assert_array_equal(weighted.hold_out(weighted.shown).voter_weights, [1, 2, 3])
    assert weighted != records


def test_four_distinct_item_scales_give_one_scale_per_item() -> None:
    scales = {
        "q1": OrdinalScale((1, 2)),
        "q2": OrdinalScale((1, 2, 3)),
        "q3": OrdinalScale((1, 2, 3, 4)),
        "q4": OrdinalScale((1, 2, 3, 4, 5)),
    }
    votes = VoteMatrix.from_responses([[2, 3, 4, 5], [1, None, 1, 3]], item_scales=scales)
    np.testing.assert_array_equal(votes.levels, [[2, 3, 4, 5], [1, 0, 1, 3]])
    assert votes.scales == tuple(scales.values())
    assert votes.item_scales.scale_index.tolist() == [0, 1, 2, 3]
    assert [votes.scale_of(item) for item in votes.items] == list(scales.values())
    assert votes.n_levels == 5
    with pytest.raises(InvalidInputError, match="4 scales, not one"):
        _ = votes.scale


def test_a_score_in_an_unshown_cell_is_refused() -> None:
    with pytest.raises(InvalidInputError, match=r"voter 'b'.*item 'long'.*not shown"):
        VoteMatrix.from_responses(
            [[1, 2], [2, 1]],
            voters=["a", "b"],
            items=["short", "long"],
            item_scales={"short": OrdinalScale((1, 2)), "long": OrdinalScale((1, 2))},
            shown=[[True, True], [True, False]],
        )


def test_survey_responses_name_the_item_when_a_score_is_off_the_scale() -> None:
    three = OrdinalScale((1, 2, 3))
    with pytest.raises(
        InvalidInputError, match="exactly the items of the matrix; no scale for 'a'"
    ):
        VoteMatrix.from_responses([[1]], items=["a"], item_scales={"other": three})
    with pytest.raises(InvalidInputError, match="item 'short'"):
        VoteMatrix.from_responses([[4]], item_scales={"short": three})
    with pytest.raises(
        InvalidInputError, match=r"item 'short'.*score 4 is not on the scale \+1, \+2, \+3"
    ):
        VoteMatrix.from_responses(
            [[4, 4]],
            item_scales={"long": OrdinalScale((1, 2, 3, 4)), "short": three},
        )


@pytest.mark.parametrize("weights", [[], [0], [-1], [float("nan")], [float("inf")], [True]])
def test_survey_weights_are_positive_finite_and_aligned(weights: list[float]) -> None:
    with pytest.raises(InvalidInputError, match="voter_weights"):
        VoteMatrix.from_responses([[1]], voter_weights=weights)


def test_voter_covariates_is_exported_from_the_covariates_module() -> None:
    assert CovariatesFromModule is VoterCovariates
    assert VoterCovariates.__module__ == "civiqhub_selene.blocks.covariates"


def test_blocks_modules_do_not_import_covariates() -> None:
    """Engine modules do not import covariates; only the package re-export and survey do."""
    root = Path(__file__).resolve().parents[2] / "src" / "civiqhub_selene" / "blocks"
    allowed = {"__init__.py", "covariates.py", "survey.py"}
    offenders: list[str] = []
    for path in sorted(root.glob("*.py")):
        if path.name in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules.append(node.module)
            if any(
                name == "civiqhub_selene.blocks.covariates" or name.endswith(".covariates")
                for name in modules
            ):
                offenders.append(path.name)
                break
    assert offenders == []


def test_voter_covariates_align_by_identifier_and_keep_a_stable_fingerprint() -> None:
    votes = VoteMatrix.from_table([[1], [1]], voters=["b", "a"])
    covariates = VoterCovariates.from_records(
        votes,
        {"a": {"mode": "web", "age": None}, "b": {"mode": "phone", "age": 41}},
    )
    same = VoterCovariates.from_records(
        votes,
        {"b": {"mode": "phone", "age": 41}, "a": {"mode": "web", "age": None}},
    )
    other = VoterCovariates.from_records(votes, {"a": {"mode": "web"}, "b": {"mode": "phone"}})
    assert covariates.voters == votes.voters
    assert covariates.columns == {"mode": ("phone", "web"), "age": (41, None)}
    assert covariates.fingerprint() == same.fingerprint()
    assert covariates.fingerprint() != other.fingerprint()
    assert re.fullmatch(r"[0-9a-f]{16}", covariates.fingerprint())
    by_mode_first = VoterCovariates(votes.voters, {"mode": ("phone", "web"), "age": (41, None)})
    by_age_first = VoterCovariates(votes.voters, {"age": (41, None), "mode": ("phone", "web")})
    assert by_mode_first == by_age_first
    assert hash(by_mode_first) == hash(by_age_first)
    assert by_mode_first.fingerprint() == by_age_first.fingerprint()
    assert {by_mode_first, by_age_first} == {by_mode_first}
    with pytest.raises(InvalidInputError, match="voter IDs"):
        VoterCovariates.from_records(votes, {"a": {"mode": "web"}})
    with pytest.raises(InvalidInputError, match="same fields"):
        VoterCovariates.from_records(votes, {"a": {"mode": "web"}, "b": {"age": 41}})


def _one_block_model(
    model_type: type[LatentBlockModel] | type[MixedMembershipBlockModel],
) -> LatentBlockModel | MixedMembershipBlockModel:
    return model_type(n_groups=1, n_blocks=1, response_model=Categorical(smoothing=0), n_starts=1)


@pytest.mark.parametrize("model_type", [LatentBlockModel, MixedMembershipBlockModel])
def test_unit_weights_match_the_unweighted_fit(
    model_type: type[LatentBlockModel] | type[MixedMembershipBlockModel],
) -> None:
    # TEST ONLY. Seed 7.
    scale = OrdinalScale((1, 2))
    scores = [[1, 1], [2, 2]]
    model = _one_block_model(model_type)
    plain = model.estimate(VoteMatrix.from_responses(scores, scale=scale), seed=7).value
    ones = model.estimate(
        VoteMatrix.from_responses(scores, scale=scale, voter_weights=[1, 1]), seed=7
    ).value
    assert ones.objective == pytest.approx(plain.objective)
    assert ones.coclusters.vote_shares[0, 0] == pytest.approx(plain.coclusters.vote_shares[0, 0])
    assert ones.coclusters.n_votes[0, 0] == pytest.approx(plain.coclusters.n_votes[0, 0])


@pytest.mark.parametrize("model_type", [LatentBlockModel, MixedMembershipBlockModel])
def test_survey_weights_change_the_fit_but_not_the_number_of_observations(
    model_type: type[LatentBlockModel] | type[MixedMembershipBlockModel],
) -> None:
    # TEST ONLY. Seed 7. Mean-one weights: [1, 3] and [10, 30] are the same parameters.
    scale = OrdinalScale((1, 2))
    scores = [[1, 1], [2, 2]]
    model = _one_block_model(model_type)
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


@pytest.mark.parametrize("model_type", [LatentBlockModel, MixedMembershipBlockModel])
def test_weighted_parameters_match_duplication_when_block_shares_are_neutral(
    model_type: type[LatentBlockModel] | type[MixedMembershipBlockModel],
) -> None:
    # TEST ONLY. Seed 7. One block: the block-share term is log(1) = 0.
    scale = OrdinalScale((1, 2))
    model = _one_block_model(model_type)
    weighted = model.estimate(
        VoteMatrix.from_responses(
            [[1, 1], [2, 2]],
            voters=["a", "b"],
            items=["x", "y"],
            scale=scale,
            voter_weights=[1, 3],
        ),
        seed=7,
    ).value
    duplicated = model.estimate(
        VoteMatrix.from_responses(
            [[1, 1], [2, 2], [2, 2], [2, 2]],
            voters=["a", "b1", "b2", "b3"],
            items=["x", "y"],
            scale=scale,
        ),
        seed=7,
    ).value
    assert weighted.coclusters.vote_shares[0, 0] == pytest.approx(
        duplicated.coclusters.vote_shares[0, 0]
    )
    assert weighted.coclusters.vote_rate[0, 0] == pytest.approx(
        duplicated.coclusters.vote_rate[0, 0]
    )


def _shares_by_group(shares: np.ndarray) -> np.ndarray:
    rows = np.asarray(shares[:, 0, :])
    return rows[np.lexsort(rows.T)]


@pytest.mark.parametrize("model_type", [LatentBlockModel, MixedMembershipBlockModel])
def test_weighted_parameters_match_duplication_with_two_groups_and_the_default_model(
    model_type: type[LatentBlockModel] | type[MixedMembershipBlockModel],
) -> None:
    # TEST ONLY. Seed 7. Two groups, one block, default OrdinalProbit: the block-share
    # term is log(1) = 0. FixedStart keeps the three copies of b in the same group;
    # RandomStart on four rows can split them. A tight tolerance is needed because the
    # outer loop stops on the absolute gain of an extensive objective.
    scale = OrdinalScale((1, 2))
    weighted_model = model_type(
        n_groups=2,
        n_blocks=1,
        n_starts=1,
        tolerance=1e-8,
        initialization=FixedStart((0, 1), (0, 0)),
    )
    duplicated_model = model_type(
        n_groups=2,
        n_blocks=1,
        n_starts=1,
        tolerance=1e-8,
        initialization=FixedStart((0, 1, 1, 1), (0, 0)),
    )
    weighted = weighted_model.estimate(
        VoteMatrix.from_responses(
            [[1, 1], [2, 2]],
            voters=["a", "b"],
            items=["x", "y"],
            scale=scale,
            voter_weights=[1, 3],
        ),
        seed=7,
    ).value
    duplicated = duplicated_model.estimate(
        VoteMatrix.from_responses(
            [[1, 1], [2, 2], [2, 2], [2, 2]],
            voters=["a", "b1", "b2", "b3"],
            items=["x", "y"],
            scale=scale,
        ),
        seed=7,
    ).value
    np.testing.assert_allclose(
        _shares_by_group(weighted.coclusters.vote_shares),
        _shares_by_group(duplicated.coclusters.vote_shares),
        atol=1e-8,
    )
    rates = np.sort(np.asarray(weighted.coclusters.vote_rate[:, 0]))
    duplicated_rates = np.sort(np.asarray(duplicated.coclusters.vote_rate[:, 0]))
    np.testing.assert_allclose(rates, duplicated_rates, atol=1e-8)


def test_csv_loads_homogeneous_scales_weights_and_aligned_covariates() -> None:
    csv = StringIO(
        '\ufeffresponseid;Q2;Q1;POIDS;mode\n001;5;2;1,5;"phone; landline"\n002;;NA;0,5;\n'
    )
    five = OrdinalScale((1, 2, 3, 4, 5))
    matrix, covariates = read_survey_csv(
        csv,
        item_scales={"Q1": five, "Q2": five},
        delimiter=";",
        decimal=",",
        weight_column="POIDS",
        covariate_columns=("mode",),
        unshown_label="NA",
    )
    assert matrix.voters == ("001", "002")
    assert matrix.items == ("Q1", "Q2")
    assert matrix.levels.tolist() == [[2, 5], [0, 0]]
    assert matrix.shown.tolist() == [[True, True], [False, True]]
    np.testing.assert_array_equal(matrix.voter_weights, [1.5, 0.5])
    assert covariates.voters == matrix.voters
    assert covariates.columns["mode"] == ("phone; landline", None)


def test_csv_reads_items_on_different_scales_and_names_the_column_of_a_bad_score() -> None:
    scales = {"Q1": OrdinalScale((1, 2)), "Q2": OrdinalScale((1, 2, 3))}
    votes, _ = read_survey_csv(StringIO("responseid,Q1,Q2\na,1,3\nb,2,\n"), item_scales=scales)
    np.testing.assert_array_equal(votes.levels, [[1, 3], [2, 0]])
    assert votes.scales == (scales["Q1"], scales["Q2"])
    with pytest.raises(InvalidInputError, match="item 'Q1'"):
        read_survey_csv(StringIO("responseid,Q1,Q2\na,3,1\n"), item_scales=scales)


@pytest.mark.parametrize(
    ("rows", "message", "weight_column"),
    [
        ("", "must have a header", None),
        ("responseid,responseid,Q\na,b,1\n", "column names must be unique", None),
        ("responseid,other\na,1\n", "missing columns", None),
        ("responseid,Q\na,1,extra\n", "has 3 fields", None),
        ("responseid,Q\n,2\n", "column 'responseid'", None),
        ("responseid,Q\na,one\n", "integer score", None),
        ("responseid,Q\na,3\n", "item 'Q'", None),
        ("responseid,Q\na,1\na,2\n", "identifier 'a' is repeated", None),
        ("responseid,Q,w\na,1,x\n", "column 'w'", "w"),
        ("responseid,Q,w\na,1,0\n", "column 'w'", "w"),
    ],
)
def test_csv_rejects_invalid_survey_rows_before_fitting(
    rows: str, message: str, weight_column: str | None
) -> None:
    with pytest.raises(InvalidInputError, match=message):
        read_survey_csv(
            StringIO(rows),
            item_scales={"Q": OrdinalScale((1, 2))},
            weight_column=weight_column,
        )
