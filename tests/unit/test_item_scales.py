# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Each item of a vote matrix has its scale: ItemScales, and what the matrix does with it.

TEST ONLY. Synthetic votes; the scales mimic the shapes of survey questions (two, three,
four and five answers), not any real survey.
"""

import re

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene.blocks import (
    CoclusterTable,
    ItemScales,
    LatentBlockModel,
    MixedMembershipBlockModel,
    OrdinalScale,
    Partition,
    VoteMatrix,
)
from civiqhub_selene.errors import InvalidInputError
from civiqhub_selene.types import UnavailableReason

pytestmark = pytest.mark.blocks

YES_NO = OrdinalScale((0, 1))
THREE = OrdinalScale((1, 2, 3))
FOUR = OrdinalScale((1, 2, 3, 4))
AGREE = OrdinalScale()


def survey() -> VoteMatrix:
    """Four voters, five items on three scales."""
    return VoteMatrix.from_responses(
        [
            [1, 4, -2, 0, 3],
            [0, 1, 2, 1, None],
            [None, 3, 0, 1, 1],
            [1, 2, 1, None, 2],
        ],
        voters=["v1", "v2", "v3", "v4"],
        item_scales={"q1": YES_NO, "q2": FOUR, "q3": AGREE, "q4": YES_NO, "q5": FOUR},
        shown=[
            [True, True, True, True, True],
            [True, True, True, True, False],
            [True, True, True, True, True],
            [True, True, True, False, True],
        ],
    )


# ItemScales ----------------------------------------------------------------------------


def test_from_scales_merges_equal_scales_in_the_order_of_their_first_item() -> None:
    scales = ItemScales.from_scales([FOUR, YES_NO, OrdinalScale((1, 2, 3, 4)), YES_NO, THREE])
    assert scales.scales == (FOUR, YES_NO, THREE)
    assert scales.scales[0] is FOUR
    assert scales.scale_index.tolist() == [0, 1, 0, 1, 2]
    assert (scales.n_items, scales.n_scales, len(scales)) == (5, 3, 5)
    assert not scales.is_uniform
    assert scales.n_levels == 4
    assert scales.levels_by_item.tolist() == [4, 2, 4, 2, 3]
    assert list(scales) == [FOUR, YES_NO, FOUR, YES_NO, THREE]
    assert scales[3] == YES_NO
    assert scales[-1] == THREE
    assert scales.to_dict() == {
        "scales": [[1, 2, 3, 4], [0, 1], [1, 2, 3]],
        "scale_index": [0, 1, 0, 1, 2],
    }


def test_uniform_scales_read_as_one_scale() -> None:
    scales = ItemScales.uniform(THREE, 4)
    assert scales.is_uniform
    assert scales.scale is THREE
    assert scales.scale_index.tolist() == [0, 0, 0, 0]
    assert ItemScales.from_scales([THREE] * 4) == scales
    assert hash(ItemScales.from_scales([THREE] * 4)) == hash(scales)
    assert ItemScales.from_scales([]) == ItemScales.uniform(AGREE, 0)
    with pytest.raises(InvalidInputError, match="3 scales, not one") as error:
        _ = ItemScales.from_scales([YES_NO, THREE, FOUR]).scale
    assert "_by_scale" in error.value.hint


def test_the_impossible_states_of_each_scale_are_known() -> None:
    scales = ItemScales.from_scales([YES_NO, FOUR])
    assert scales._possible_states.tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]


@pytest.mark.parametrize(
    ("scales", "index", "message"),
    [
        ((), [], "at least one scale"),
        (THREE, [0], "collection of OrdinalScale"),
        ((THREE, (1, 2)), [0, 1], "a scale is an OrdinalScale"),
        ((THREE, OrdinalScale((1, 2, 3))), [0, 1], "same scores"),
        ((THREE, FOUR), [0, 2], "positions from 0 to 1"),
        ((THREE, FOUR), [0, 0], "only 1 of the 2 scales"),
        ((THREE, FOUR), [1, 0], "order of their first item"),
        ((THREE, FOUR), [], "exactly one scale"),
        ((THREE,), [[0]], "one integer per item"),
        ((THREE,), [0.0], "one integer per item"),
    ],
)
def test_an_inconsistent_item_scales_is_refused(
    scales: object, index: object, message: str
) -> None:
    with pytest.raises(InvalidInputError, match=message):
        ItemScales(scales, np.asarray(index))  # type: ignore[arg-type]


def test_the_constructors_refuse_what_is_not_a_scale() -> None:
    with pytest.raises(InvalidInputError, match="single scale"):
        ItemScales.from_scales(THREE)  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError, match="OrdinalScale"):
        ItemScales.from_scales([THREE, (1, 2)])  # type: ignore[list-item]
    with pytest.raises(InvalidInputError, match="non-negative integer"):
        ItemScales.uniform(THREE, -1)
    with pytest.raises(TypeError, match="position"):
        ItemScales.uniform(THREE, 2)["q1"]  # type: ignore[index]
    with pytest.raises(IndexError, match="2 items"):
        ItemScales.uniform(THREE, 2)[2]


def test_item_scales_describe_themselves() -> None:
    lines = str(ItemScales.from_scales([YES_NO, FOUR, YES_NO])).splitlines()
    assert lines == [
        "ItemScales(2 scales of 2 to 4 levels over 3 items)",
        "scale 0: 0, +1 (2 items)",
        "scale 1: +1, +2, +3, +4 (1 item)",
    ]
    assert repr(ItemScales.uniform(AGREE, 1)) == "ItemScales(1 scale of 5 levels over 1 item)"


# The matrix ----------------------------------------------------------------------------


def test_each_score_is_read_on_the_scale_of_its_item() -> None:
    votes = survey()
    assert votes.scales == (YES_NO, FOUR, AGREE)
    assert votes.item_scales.scale_index.tolist() == [0, 1, 2, 0, 1]
    assert votes.scale_of("q3") is AGREE
    assert votes.n_levels == 5
    np.testing.assert_array_equal(
        votes.levels,
        [[2, 4, 1, 1, 3], [1, 1, 5, 2, 0], [0, 3, 3, 2, 1], [2, 2, 4, 0, 2]],
    )
    assert sorted(votes.to_records())[:3] == [("v1", "q1", 1), ("v1", "q2", 4), ("v1", "q3", -2)]
    assert repr(votes) == "VoteMatrix(4 voters x 5 items, 17 votes, 3 scales of 2 to 5 levels)"
    with pytest.raises(InvalidInputError, match="this vote matrix has 3 scales, not one"):
        _ = votes.scale
    with pytest.raises(InvalidInputError, match="'q9' is not in the matrix"):
        votes.scale_of("q9")


def test_the_states_have_the_width_of_the_longest_scale() -> None:
    states = survey()._states()
    assert states.shape == (4, 5, 6)
    # A yes/no item never reaches the states past its two levels.
    assert not states[:, [0, 3], 3:].any()
    assert states[0, 2].tolist() == [0, 1, 0, 0, 0, 0]


def test_a_level_beyond_the_scale_of_its_column_is_refused() -> None:
    scales = ItemScales.from_scales([YES_NO, FOUR])
    VoteMatrix.from_arrays([[2, 4]], item_scales=scales)
    with pytest.raises(InvalidInputError, match=r"item 'i1' is on a scale of 2 levels.*got 3"):
        VoteMatrix.from_arrays([[3, 4]], item_scales=scales)
    with pytest.raises(InvalidInputError, match="2 items to 3 columns"):
        VoteMatrix.from_arrays([[1, 1, 1]], item_scales=scales)


def test_records_read_each_score_on_the_scale_of_its_item() -> None:
    by_item = {"long": FOUR, "short": YES_NO}
    votes = VoteMatrix.from_records(
        [("a", "short", 1), ("a", "long", 4), ("b", "long", 1)], item_scales=by_item
    )
    assert votes.items == ("long", "short")
    np.testing.assert_array_equal(votes.levels, [[4, 2], [1, 0]])
    in_order = VoteMatrix.from_records(
        [("a", "short", 1), ("a", "long", 4), ("b", "long", 1)],
        items=["long", "short"],
        item_scales=ItemScales.from_scales([FOUR, YES_NO]),
    )
    assert in_order == votes
    with pytest.raises(
        InvalidInputError, match=r"item 'short'.*score 4 is not on the scale 0, \+1"
    ):
        VoteMatrix.from_records([("a", "short", 4)], item_scales=by_item)
    with pytest.raises(InvalidInputError, match="give items with it"):
        VoteMatrix.from_records([("a", "short", 1)], item_scales=ItemScales.uniform(YES_NO, 1))
    with pytest.raises(InvalidInputError, match="no scale for 'other'"):
        VoteMatrix.from_records([], items=["long", "other"], item_scales=by_item)


def test_scale_and_item_scales_must_agree() -> None:
    three = {"a": THREE, "b": THREE}
    one = VoteMatrix.from_responses([[1, 3]], item_scales=three, scale=THREE)
    assert one.scale is THREE
    with pytest.raises(InvalidInputError, match="differs from the item scale"):
        VoteMatrix.from_responses([[1, 3]], item_scales=three, scale=FOUR)
    with pytest.raises(InvalidInputError, match="gives 2"):
        VoteMatrix.from_responses([[1, 1]], item_scales={"a": THREE, "b": FOUR}, scale=THREE)
    with pytest.raises(InvalidInputError, match="disagree"):
        VoteMatrix(
            np.array([[1, 1]]),
            np.ones((1, 2), dtype=bool),
            ("v1",),
            ("a", "b"),
            THREE,
            item_scales=ItemScales.from_scales([THREE, FOUR]),
        )
    with pytest.raises(InvalidInputError, match="item_scales is an ItemScales"):
        VoteMatrix(
            np.array([[1]]),
            np.ones((1, 1), dtype=bool),
            ("v1",),
            ("a",),
            item_scales={"a": THREE},  # type: ignore[arg-type]
        )
    with pytest.raises(InvalidInputError, match="maps each item identifier"):
        VoteMatrix.from_arrays([[1]], item_scales=[THREE])  # type: ignore[arg-type]


def test_selection_and_hold_out_keep_the_scale_of_each_item() -> None:
    votes = survey()
    both = votes[:, ["q1", "q2", "q4"]]
    assert both.scales == (YES_NO, FOUR)
    assert both.item_scales.scale_index.tolist() == [0, 1, 0]
    one = votes[:, ["q2", "q5"]]
    assert one.scale is FOUR
    assert one.item_scales.is_uniform
    none = votes[:, []]
    assert none.shape == (4, 0)
    assert none.item_scales.n_scales == 1
    held = votes.hold_out(np.eye(4, 5, dtype=bool))
    assert held.item_scales == votes.item_scales
    assert held.n_shown == votes.n_shown - 3  # v4 was never shown q4


def test_the_fingerprint_of_a_matrix_on_one_scale_is_unchanged() -> None:
    # Stored on develop at d80da06, before items could have their own scales.
    votes = VoteMatrix.from_records(
        [("a", "p", 2), ("a", "q", -1), ("b", "p", 0)], shown=[("b", "q")]
    )
    assert votes.fingerprint() == "9e249439a90d3537"
    by_item = VoteMatrix.from_records(
        [("a", "p", 2), ("a", "q", -1), ("b", "p", 0)],
        shown=[("b", "q")],
        item_scales={"p": AGREE, "q": AGREE},
    )
    assert by_item == votes
    assert by_item.fingerprint() == "9e249439a90d3537"


def test_the_fingerprint_follows_the_scale_of_each_item() -> None:
    levels = [[1, 2], [2, 1]]
    fingerprints = {
        VoteMatrix.from_arrays(levels, scale=FOUR).fingerprint(),
        VoteMatrix.from_arrays(
            levels, item_scales=ItemScales.from_scales([FOUR, YES_NO])
        ).fingerprint(),
        VoteMatrix.from_arrays(
            levels, item_scales=ItemScales.from_scales([YES_NO, FOUR])
        ).fingerprint(),
        VoteMatrix.from_arrays(
            levels, item_scales=ItemScales.from_scales([THREE, YES_NO])
        ).fingerprint(),
    }
    assert len(fingerprints) == 4
    for value in fingerprints:
        assert re.fullmatch(r"[0-9a-f]{16}", value)


# What reads the matrix -----------------------------------------------------------------


def test_counts_by_scale_keep_the_levels_of_each_scale_apart() -> None:
    votes = survey()
    groups = Partition.from_voter_labels(votes.voters, [0, 0, 1, 1])
    blocks = Partition.from_item_labels(votes.items, [0, 0, 1, 1, 0])
    grouped = votes.group_by(groups, blocks)
    counts = grouped.counts_by_scale()
    assert counts.shape == (2, 2, 3, 6)
    # Group 0 on block 0: q1 (yes/no) gives levels 2 and 1, q2 and q5 (four) give 4, 1, 3.
    assert counts[0, 0, 0].tolist() == [0, 1, 1, 0, 0, 0]
    assert counts[0, 0, 1].tolist() == [0, 1, 0, 1, 1, 0]
    assert counts[0, 0, 2].tolist() == [0] * 6
    assert counts.sum() == votes.n_shown
    with pytest.raises(InvalidInputError, match="3 scales, not one"):
        grouped.counts()
    with pytest.raises(InvalidInputError, match="3 scales, not one"):
        grouped.vote_counts(0, 0)
    one_scale = votes[:, ["q2", "q5"]].group_by()
    assert one_scale.counts_by_scale()[:, :, 0].tolist() == one_scale.counts().tolist()


def test_the_table_as_counted_has_one_distribution_per_scale() -> None:
    votes = survey()
    table = votes.group_by().to_table()
    assert table.scales == votes.scales
    shares = table.vote_shares_by_scale
    assert [array.shape for array in shares] == [(1, 1, 2), (1, 1, 4), (1, 1, 5)]
    # Yes/no items: 6 votes, two at 0 and four at 1.
    assert shares[0][0, 0].tolist() == pytest.approx([1 / 3, 2 / 3])
    cocluster = table[0, 0]
    assert cocluster.scores_by_scale == ((0, 1), (1, 2, 3, 4), (-2, -1, 0, 1, 2))
    assert cocluster.typical_score_by_scale == (1, 2, 0)
    assert cocluster.n_votes == votes.n_votes
    with pytest.raises(InvalidInputError, match="this co-cluster has 3 scales"):
        _ = cocluster.vote_shares
    with pytest.raises(InvalidInputError, match="this table of co-clusters has 3 scales"):
        _ = table.vote_shares
    with pytest.raises(InvalidInputError, match="3 scales"):
        table.log_probabilities()
    logs = table.log_probabilities_by_scale()
    assert logs.shape == (1, 1, 3, 6)
    probabilities = np.exp(logs)
    possible = votes.item_scales._possible_states
    assert probabilities[0, 0][possible].reshape(-1).sum() == pytest.approx(3.0)
    assert (probabilities[0, 0][~possible] < 1e-299).all()
    data = table.to_dict()
    assert set(data) == {"scales", "vote_rate", "vote_shares_by_scale", "n_votes"}
    assert data["scales"] == [[0, 1], [1, 2, 3, 4], [-2, -1, 0, 1, 2]]
    assert (
        str(table).splitlines()[0] == "CoclusterTable(1 group x 1 block, 3 scales of 2 to 5 levels)"
    )


def test_a_table_of_several_scales_checks_each_of_them() -> None:
    rate, n_votes = [[0.5]], [[2.0]]
    two = [[[0.5, 0.5]]]
    three = [[[0.2, 0.3, 0.5]]]
    table = CoclusterTable(
        (YES_NO, THREE), rate, (two, three), n_votes, thresholds=((0.5,), (0.3, 0.6))
    )
    assert table.thresholds is not None
    assert [grid.tolist() for grid in table.thresholds] == [[0.5], [0.3, 0.6]]
    assert table.to_dict()["thresholds"] == [[0.5], [0.3, 0.6]]
    with pytest.raises(InvalidInputError, match="one array per scale"):
        CoclusterTable((YES_NO, THREE), rate, (two,), n_votes)
    with pytest.raises(InvalidInputError, match=r"\(groups, blocks, levels\) = \(1, 1, 3\)"):
        CoclusterTable((YES_NO, THREE), rate, (two, two), n_votes)
    with pytest.raises(InvalidInputError, match="2 thresholds"):
        CoclusterTable((YES_NO, THREE), rate, (two, three), n_votes, thresholds=((0.5,), (0.3,)))
    with pytest.raises(InvalidInputError, match="strictly increasing"):
        CoclusterTable(THREE, rate, three, n_votes, thresholds=(0.6, 0.3))
    with pytest.raises(InvalidInputError, match="each an OrdinalScale"):
        CoclusterTable((YES_NO, (1, 2)), rate, (two, two), n_votes)  # type: ignore[arg-type]


@pytest.mark.parametrize("model", [LatentBlockModel(), MixedMembershipBlockModel()])
def test_a_model_of_one_scale_answers_several_with_an_explained_absence(
    model: LatentBlockModel | MixedMembershipBlockModel,
) -> None:
    result = model.estimate(survey(), seed=0)
    if type(model).accepts_several_scales:
        pytest.skip(f"{type(model).__name__} reads several scales")
    assert not result.is_available
    assert result.unavailable is not None
    assert result.unavailable.reason is UnavailableReason.UNSUPPORTED_INPUT
    assert "3 scales" in result.unavailable.detail
