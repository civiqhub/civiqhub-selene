# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import re

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene.blocks.data import OrdinalScale, Vote, VoteMatrix
from civiqhub_selene.errors import InvalidInputError, InvalidParameterError

pytestmark = pytest.mark.blocks

RECORDS = [("a", "p1", 2), ("b", "p1", -2), ("a", "p2", 0)]


def shown_pairs(matrix: VoteMatrix) -> list[tuple[str, str]]:
    return [(matrix.voters[row], matrix.items[column]) for row, column in np.argwhere(matrix.shown)]


# OrdinalScale -------------------------------------------------------------------------


def test_default_scale_is_the_five_score_agreement_scale() -> None:
    scale = OrdinalScale()
    assert scale.scores == (-2, -1, 0, 1, 2)
    assert scale.n_levels == 5
    assert [scale.level_of(score) for score in scale.scores] == [1, 2, 3, 4, 5]
    assert [scale.score_of(level) for level in range(1, 6)] == [-2, -1, 0, 1, 2]
    assert [scale.format_score(score) for score in scale.scores] == ["-2", "-1", "0", "+1", "+2"]


def test_a_scale_is_frozen_hashable_and_stores_a_tuple() -> None:
    scale = OrdinalScale([1, 2, 3])  # type: ignore[arg-type]
    assert scale.scores == (1, 2, 3)
    assert scale == OrdinalScale((1, 2, 3))
    assert hash(scale) == hash(OrdinalScale((1, 2, 3)))
    with pytest.raises(dataclasses.FrozenInstanceError):
        scale.scores = (1, 2)  # type: ignore[misc]


@pytest.mark.parametrize("scores", [(1,), (), (1, 1, 2), (2, 1), (1.0, 2.0), (True, 2), "12", 5])
def test_a_scale_needs_two_strictly_increasing_integers(scores: object) -> None:
    with pytest.raises(InvalidParameterError, match=re.escape(repr(scores))):
        OrdinalScale(scores)  # type: ignore[arg-type]


def test_a_score_or_a_level_off_the_scale_is_refused() -> None:
    scale = OrdinalScale()
    with pytest.raises(
        InvalidInputError, match="score 3 is not on the scale -2, -1, 0, \\+1, \\+2"
    ):
        scale.level_of(3)
    with pytest.raises(InvalidInputError, match=r"score 1\.0"):
        scale.level_of(1.0)  # type: ignore[arg-type]
    for level in (0, 6, True):
        with pytest.raises(InvalidInputError, match="rank from 1 to 5"):
            scale.score_of(level)


# Construction -------------------------------------------------------------------------


def test_from_records_without_shown_reads_empty_cells_as_abstentions() -> None:
    matrix = VoteMatrix.from_records(RECORDS)
    assert matrix.shape == (2, 2)
    assert matrix.voters == ("a", "b")
    assert matrix.items == ("p1", "p2")
    assert matrix.n_votes == 3
    assert matrix.n_shown == 4
    assert matrix.n_levels == 5
    assert matrix.levels.tolist() == [[5, 3], [1, 0]]
    # (b, p2): shown, no vote.
    assert bool(matrix.shown[1, 1]) is True
    assert bool(matrix.voted[1, 1]) is False


def test_from_records_with_shown_only_shows_these_cells_and_the_voted_ones() -> None:
    same = VoteMatrix.from_records(RECORDS, shown=[("b", "p2")])
    assert same == VoteMatrix.from_records(RECORDS)
    assert same.n_shown == 4

    hidden = VoteMatrix.from_records(RECORDS, shown=[])
    assert hidden.n_shown == 3
    assert hidden.n_votes == 3
    assert bool(hidden.shown[1, 1]) is False
    assert hidden != same


def test_shown_can_name_a_voter_or_an_item_without_any_vote() -> None:
    matrix = VoteMatrix.from_records(RECORDS, shown=[("c", "p3")])
    assert matrix.voters == ("a", "b", "c")
    assert matrix.items == ("p1", "p2", "p3")
    assert matrix.n_shown == 4
    assert matrix.n_active_voters == 2
    assert matrix.n_active_items == 2


def test_a_neutral_vote_is_a_vote() -> None:
    matrix = VoteMatrix.from_records([("a", "p1", 0)])
    assert matrix.n_votes == 1
    assert matrix.levels.tolist() == [[3]]
    empty_cell = VoteMatrix.from_table([[None]], voters=["a"], items=["p1"])
    assert empty_cell.n_votes == 0
    assert matrix != empty_cell


def test_to_records_gives_the_votes_back_voter_by_voter() -> None:
    matrix = VoteMatrix.from_records(RECORDS)
    records = list(matrix.to_records())
    assert records == [Vote("a", "p1", 2), Vote("a", "p2", 0), Vote("b", "p1", -2)]
    assert sorted(records) == sorted(Vote(*record) for record in RECORDS)
    assert records[0].voter == "a"
    assert records[0].item == "p1"
    assert records[0].score == 2


@pytest.mark.parametrize("shown", [None, [], [("b", "p2")]])
def test_records_round_trip_to_an_equal_matrix(shown: list[tuple[str, str]] | None) -> None:
    matrix = VoteMatrix.from_records(RECORDS, shown=shown)
    again = VoteMatrix.from_records(matrix.to_records(), shown=shown_pairs(matrix))
    assert again == matrix
    assert again.fingerprint() == matrix.fingerprint()


def test_records_round_trip_keeps_the_order_when_identifiers_are_given() -> None:
    matrix = VoteMatrix.from_table(
        [[None, 2, None], [-1, None, None], [None, None, None]],
        voters=["z", "y", "x"],
        items=["q", "p", "o"],
    )
    again = VoteMatrix.from_records(
        matrix.to_records(),
        shown=shown_pairs(matrix),
        voters=matrix.voters,
        items=matrix.items,
        scale=matrix.scale,
    )
    assert again == matrix
    with pytest.raises(InvalidInputError, match="'a' is in the records but not in the voters"):
        VoteMatrix.from_records(RECORDS, voters=["b"])
    with pytest.raises(InvalidInputError, match="'p2' is in the records but not in the items"):
        VoteMatrix.from_records(RECORDS, items=["p1"])


def test_from_table_reads_none_and_nan_as_empty_cells_all_shown() -> None:
    matrix = VoteMatrix.from_table([[2, None, -2], [float("nan"), 0, 1.0]])
    assert matrix.voters == ("v1", "v2")
    assert matrix.items == ("i1", "i2", "i3")
    assert matrix.levels.tolist() == [[5, 0, 1], [0, 3, 4]]
    assert matrix.n_shown == 6
    assert matrix.n_votes == 4
    array = np.array([[2.0, np.nan], [-1.0, 0.0]])
    assert VoteMatrix.from_table(array).levels.tolist() == [[5, 0], [2, 3]]
    with pytest.raises(InvalidInputError, match="same length"):
        VoteMatrix.from_table([[1, 2], [1]])
    with pytest.raises(InvalidInputError, match=r"score 0\.5"):
        VoteMatrix.from_table([[0.5]])


def test_from_arrays_takes_the_internal_encoding() -> None:
    matrix = VoteMatrix.from_arrays([[5, 0], [1, 3]], shown=[[True, False], [True, True]])
    assert matrix.voters == ("v1", "v2")
    assert matrix.items == ("i1", "i2")
    assert matrix.n_votes == 3
    assert matrix.n_shown == 3
    named = VoteMatrix.from_arrays(np.array([[1, 2]]), voters=["a"], items=["p", "q"])
    assert named.shape == (1, 2)
    three = VoteMatrix.from_arrays([[3, 1]], scale=OrdinalScale((1, 2, 3)))
    assert [vote.score for vote in three.to_records()] == [3, 1]


def test_an_empty_matrix_is_a_matrix() -> None:
    matrix = VoteMatrix.from_records([])
    assert matrix.shape == (0, 0)
    assert matrix.n_votes == 0
    assert matrix.n_shown == 0
    assert list(matrix.to_records()) == []
    assert matrix._states().shape == (0, 0, 6)
    assert str(matrix) == "VoteMatrix(0 voters x 0 items, 0 votes, 5 levels)\nno cell"


# Selection ----------------------------------------------------------------------------


def test_selection_by_identifiers_keeps_the_order_and_the_scale() -> None:
    scale = OrdinalScale((1, 2, 3, 4, 5))
    matrix = VoteMatrix.from_table(
        [[1, 2, 3], [4, 5, None], [None, 1, 2]],
        voters=["a", "b", "c"],
        items=["p", "q", "r"],
        scale=scale,
    )
    only_a = matrix[["a"], :]
    assert only_a.voters == ("a",)
    assert only_a.items == ("p", "q", "r")
    assert only_a.scale is scale
    assert only_a.levels.tolist() == [[1, 2, 3]]

    reordered = matrix[["c", "a"], ("r", "p")]
    assert reordered.voters == ("a", "c")
    assert reordered.items == ("p", "r")
    assert reordered.levels.tolist() == [[1, 3], [0, 2]]

    assert matrix["b", "q"].levels.tolist() == [[5]]
    assert matrix[:, :] == matrix
    assert matrix[frozenset({"a", "b"}), {"q"}].shape == (2, 1)
    assert matrix[[], :].shape == (0, 3)


def test_selection_by_position_is_refused() -> None:
    matrix = VoteMatrix.from_records(RECORDS)
    with pytest.raises(TypeError, match="by identifiers, not by positions"):
        matrix[0, :]  # type: ignore[index]
    with pytest.raises(TypeError, match="by identifiers, not by positions"):
        matrix[:, 0:1]
    with pytest.raises(TypeError, match="two sides"):
        matrix["a"]  # type: ignore[index]
    with pytest.raises(TypeError, match="collection of identifiers"):
        matrix[None, :]  # type: ignore[index]
    with pytest.raises(InvalidInputError, match="voter 'z' is not in the matrix"):
        matrix[["z"], :]
    with pytest.raises(InvalidInputError, match="item 'p9' is not in the matrix"):
        matrix[:, "p9"]


# Hold-out -----------------------------------------------------------------------------


def test_hold_out_hides_cells_and_keeps_their_votes() -> None:
    matrix = VoteMatrix.from_records(RECORDS, shown=[])
    mask = np.array([[True, False], [False, True]])
    held = matrix.hold_out(mask)
    assert held.n_shown == matrix.n_shown - int((mask & matrix.shown).sum())
    assert held.n_shown == 2
    assert held.levels.tolist() == matrix.levels.tolist()
    assert held.n_votes == 2
    assert Vote("a", "p1", 2) not in list(held.to_records())
    # The original is unchanged.
    assert matrix.n_shown == 3
    assert matrix == VoteMatrix.from_records(RECORDS, shown=[])
    with pytest.raises(InvalidInputError, match="booleans of shape"):
        matrix.hold_out(np.array([True, False]))
    with pytest.raises(InvalidInputError, match="booleans of shape"):
        matrix.hold_out(np.ones((2, 2), dtype=np.int64))


# Value semantics ----------------------------------------------------------------------


def test_arrays_are_copied_and_read_only() -> None:
    levels = np.array([[5, 0], [1, 3]])
    shown = np.ones((2, 2), dtype=bool)
    matrix = VoteMatrix.from_arrays(levels, shown=shown)
    levels[0, 0] = 1
    shown[0, 0] = False
    assert matrix.levels[0, 0] == 5
    assert bool(matrix.shown[0, 0]) is True
    for array in (matrix.levels, matrix.shown, matrix.voted, matrix._states()):
        with pytest.raises(ValueError, match="read-only"):
            array[0, 0] = 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        matrix.voters = ("x", "y")  # type: ignore[misc]


def test_fingerprint_follows_the_content() -> None:
    matrix = VoteMatrix.from_records(RECORDS)
    same = VoteMatrix.from_arrays([[5, 3], [1, 0]], voters=["a", "b"], items=["p1", "p2"])
    assert re.fullmatch(r"[0-9a-f]{16}", matrix.fingerprint())
    assert matrix.fingerprint() == same.fingerprint()
    assert matrix == same
    assert hash(matrix) == hash(same)
    assert matrix != "matrix"

    other_vote = VoteMatrix.from_records([("a", "p1", 1), *RECORDS[1:]])
    other_voter = VoteMatrix.from_arrays([[5, 3], [1, 0]], voters=["a", "c"], items=["p1", "p2"])
    other_shown = VoteMatrix.from_records(RECORDS, shown=[])
    other_scale = VoteMatrix.from_arrays(
        [[5, 3], [1, 0]],
        voters=["a", "b"],
        items=["p1", "p2"],
        scale=OrdinalScale((1, 2, 3, 4, 5)),
    )
    fingerprints = {
        matrix.fingerprint(),
        other_vote.fingerprint(),
        other_voter.fingerprint(),
        other_shown.fingerprint(),
        other_scale.fingerprint(),
    }
    assert len(fingerprints) == 5


def test_fingerprint_does_not_depend_on_the_integer_type_of_the_input() -> None:
    wide = VoteMatrix.from_arrays(np.array([[5, 0]], dtype=np.int64))
    narrow = VoteMatrix.from_arrays(np.array([[5, 0]], dtype=np.int8))
    assert wide.fingerprint() == narrow.fingerprint()
    assert narrow.levels.dtype == np.int64


# Refusals -----------------------------------------------------------------------------


def test_invalid_records_are_refused_and_named() -> None:
    with pytest.raises(InvalidInputError, match="score 3 is not on the scale"):
        VoteMatrix.from_records([("a", "p1", 3)])
    with pytest.raises(InvalidInputError, match="voter 'a' votes twice on the item 'p1'"):
        VoteMatrix.from_records([("a", "p1", 2), ("a", "p1", -2)])
    with pytest.raises(InvalidInputError, match="triple"):
        VoteMatrix.from_records([("a", "p1")])  # type: ignore[list-item]
    with pytest.raises(InvalidInputError, match="pair"):
        VoteMatrix.from_records(RECORDS, shown=[("a",)])  # type: ignore[list-item]
    with pytest.raises(InvalidInputError, match="voter identifier is text, got 1"):
        VoteMatrix.from_records([(1, "p1", 2)])  # type: ignore[list-item]


def test_invalid_arrays_are_refused_and_named() -> None:
    with pytest.raises(InvalidInputError, match="identifier 'a' is repeated"):
        VoteMatrix.from_arrays([[1, 2]], voters=["a"], items=["a", "a"])
    with pytest.raises(InvalidInputError, match="rank from 0 to 5, got 6"):
        VoteMatrix.from_arrays([[6]])
    with pytest.raises(InvalidInputError, match="rank from 0 to 5, got -1"):
        VoteMatrix.from_arrays([[-1]])
    with pytest.raises(InvalidInputError, match=r"shapes \(1, 2\) and \(2, 1\)"):
        VoteMatrix.from_arrays([[1, 2]], shown=[[True], [True]])
    with pytest.raises(InvalidInputError, match="got shape"):
        VoteMatrix.from_arrays([1, 2])
    with pytest.raises(InvalidInputError, match="2 voters and 2 items do not follow"):
        VoteMatrix.from_arrays([[1, 2]], voters=["a", "b"])
    with pytest.raises(InvalidInputError, match="integers from 0 to 5"):
        VoteMatrix.from_arrays([[1.5, 2.0]])
    with pytest.raises(InvalidInputError, match="boolean array"):
        VoteMatrix.from_arrays([[1, 2]], shown=[[1, 0]])
    with pytest.raises(InvalidInputError, match="OrdinalScale"):
        VoteMatrix.from_arrays([[1]], scale=(1, 2, 3))  # type: ignore[arg-type]


# States -------------------------------------------------------------------------------


def test_states_are_the_indicator_of_the_three_states() -> None:
    matrix = VoteMatrix.from_records(RECORDS, shown=[])
    states = matrix._states()
    assert states.shape == (2, 2, 6)
    assert states.dtype == np.float64
    # Shown cells sum to one, the cell that is not shown is zero everywhere.
    assert states.sum(axis=-1).tolist() == [[1.0, 1.0], [1.0, 0.0]]
    assert states[0, 0].tolist() == [0, 0, 0, 0, 0, 1]
    assert states[0, 1].tolist() == [0, 0, 0, 1, 0, 0]
    assert states[1, 0].tolist() == [0, 1, 0, 0, 0, 0]
    assert matrix._states() is states

    unmasked = matrix._states(ignore_shown=True)
    assert unmasked.sum(axis=-1).tolist() == [[1.0, 1.0], [1.0, 1.0]]
    assert unmasked[1, 1].tolist() == [1, 0, 0, 0, 0, 0]

    abstention = VoteMatrix.from_records(RECORDS)._states()
    assert abstention[1, 1].tolist() == [1, 0, 0, 0, 0, 0]


def test_description() -> None:
    matrix = VoteMatrix.from_records(RECORDS, shown=[])
    assert repr(matrix) == "VoteMatrix(2 voters x 2 items, 3 votes, 5 levels)"
    assert str(matrix).splitlines()[1] == (
        "75.0% of the 4 cells shown, 100.0% of the shown cells voted"
    )
    assert repr(VoteMatrix.from_records([("a", "p", 1)])) == (
        "VoteMatrix(1 voter x 1 item, 1 vote, 5 levels)"
    )
