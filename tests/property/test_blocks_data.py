# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("numpy")

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from civiqhub_selene.blocks import Partition, VoteMatrix

pytestmark = pytest.mark.blocks


@st.composite
def matrices(draw: st.DrawFn) -> VoteMatrix:
    """Small vote matrices with the three states of a cell."""
    n_voters = draw(st.integers(min_value=1, max_value=6))
    n_items = draw(st.integers(min_value=1, max_value=6))
    cells = n_voters * n_items
    levels = draw(st.lists(st.integers(0, 5), min_size=cells, max_size=cells))
    shown = draw(st.lists(st.booleans(), min_size=cells, max_size=cells))
    return VoteMatrix.from_arrays(
        np.array(levels).reshape(n_voters, n_items),
        shown=np.array(shown).reshape(n_voters, n_items),
    )


@st.composite
def labelled(draw: st.DrawFn) -> tuple[VoteMatrix, list[int], list[int], int, int]:
    """A matrix with a label per voter and per item."""
    matrix = draw(matrices())
    n_voters, n_items = matrix.shape
    n_groups = draw(st.integers(1, 3))
    n_blocks = draw(st.integers(1, 3))
    rows = draw(st.lists(st.integers(0, n_groups - 1), min_size=n_voters, max_size=n_voters))
    cols = draw(st.lists(st.integers(0, n_blocks - 1), min_size=n_items, max_size=n_items))
    return matrix, rows, cols, n_groups, n_blocks


@given(matrix=matrices())
def test_a_cell_is_in_exactly_one_state_when_shown_and_none_otherwise(matrix: VoteMatrix) -> None:
    states = matrix._states()
    assert np.array_equal(states.sum(axis=-1), matrix.shown.astype(np.float64))
    assert states[..., 1:].sum() == matrix.n_votes
    assert matrix.n_votes <= matrix.n_shown <= matrix.levels.size


@given(matrix=matrices())
def test_records_and_shown_cells_rebuild_the_matrix(matrix: VoteMatrix) -> None:
    shown = [(matrix.voters[row], matrix.items[col]) for row, col in np.argwhere(matrix.shown)]
    rebuilt = VoteMatrix.from_records(
        matrix.to_records(), shown=shown, voters=matrix.voters, items=matrix.items
    )
    assert np.array_equal(rebuilt.shown, matrix.shown)
    assert np.array_equal(rebuilt.voted, matrix.voted)
    assert list(rebuilt.to_records()) == list(matrix.to_records())


@given(data=labelled())
def test_counts_cover_every_shown_cell_once(
    data: tuple[VoteMatrix, list[int], list[int], int, int],
) -> None:
    matrix, rows, cols, n_groups, n_blocks = data
    grouped = matrix.group_by(
        Partition.from_voter_labels(matrix.voters, rows, n_clusters=n_groups),
        Partition.from_item_labels(matrix.items, cols, n_clusters=n_blocks),
    )
    counts = grouped.counts()
    assert counts.shape == (n_groups, n_blocks, 6)
    assert counts.sum() == matrix.n_shown
    assert np.array_equal(counts.sum(axis=(0, 1)), matrix.group_by().counts()[0, 0])
    table = grouped.to_table()
    assert table.n_votes.sum() == matrix.n_votes
    assert np.exp(table.log_probabilities()).sum(axis=-1) == pytest.approx(
        np.ones((n_groups, n_blocks)), abs=1e-9
    )


@given(data=labelled(), seed=st.integers(0, 1000))
def test_agreement_does_not_depend_on_the_numbering_of_clusters(
    data: tuple[VoteMatrix, list[int], list[int], int, int], seed: int
) -> None:
    matrix, rows, _, n_groups, _ = data
    partition = Partition.from_voter_labels(matrix.voters, rows, n_clusters=n_groups)
    renumbering = np.random.default_rng(seed).permutation(n_groups)
    renumbered = Partition.from_voter_labels(
        matrix.voters, renumbering[np.array(rows)], n_clusters=n_groups
    )
    assert partition.agreement_with(renumbered) == 1.0
    assert renumbered.agreement_with(partition) == 1.0
    assert sum(partition.sizes) == len(matrix.voters)
