# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import re

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene.blocks import (
    CoclusterTable,
    GroupedVotes,
    OrdinalScale,
    Partition,
    VoteMatrix,
)
from civiqhub_selene.errors import InvalidInputError
from civiqhub_selene.estimator import Fingerprinted
from civiqhub_selene.ordinal import VoteCounts

pytestmark = pytest.mark.blocks

VOTERS = ["a", "b", "c", "d"]
ITEMS = ["p", "q", "r", "s"]
# Group 0 = {a, b}, group 1 = {c, d}; block 0 = {p, q}, block 1 = {r, s}.
# (b, s) is shown without a vote; (d, q) is not shown.
RECORDS = [
    ("a", "p", 2), ("a", "q", 2), ("a", "r", -2), ("a", "s", -1),
    ("b", "p", 2), ("b", "q", 1), ("b", "r", -2),
    ("c", "p", -2), ("c", "q", -1), ("c", "r", 2), ("c", "s", 2),
    ("d", "p", -1), ("d", "r", 1), ("d", "s", 2),
]  # fmt: skip
SHOWN = [(voter, item) for voter in VOTERS for item in ITEMS if (voter, item) != ("d", "q")]


def make_grouped() -> GroupedVotes:
    matrix = VoteMatrix.from_records(RECORDS, shown=SHOWN, voters=VOTERS, items=ITEMS)
    groups = Partition.from_voter_labels(VOTERS, [0, 0, 1, 1])
    blocks = Partition.from_item_labels(ITEMS, [0, 0, 1, 1])
    return matrix.group_by(groups, blocks)


def test_counts_cover_the_shown_cells_of_each_cocluster() -> None:
    grouped = make_grouped()
    assert grouped.shape == (2, 2)
    counts = grouped.counts()
    assert counts.shape == (2, 2, 6)
    assert counts.sum(axis=-1).tolist() == [[4.0, 4.0], [3.0, 4.0]]
    # The cell that is not shown is counted nowhere.
    assert counts.sum() == grouped.votes.n_shown == 15
    # Index 0 is the abstention: only (b, s), in the co-cluster (0, 1).
    assert counts[..., 0].tolist() == [[0.0, 1.0], [0.0, 0.0]]
    assert counts[0, 0].tolist() == [0, 0, 0, 0, 1, 3]
    assert counts[1, 0].tolist() == [0, 1, 2, 0, 0, 0]


def test_to_table_is_the_table_as_it_is_counted() -> None:
    table = make_grouped().to_table()
    assert isinstance(table, CoclusterTable)
    assert table.shape == (2, 2)
    first = table[0, 0]
    assert first.vote_rate == 1.0
    assert first.vote_shares == (0.0, 0.0, 0.0, 0.25, 0.75)
    assert first.typical_score == 2
    assert first.n_votes == 4.0
    assert first.latent_mean is None
    assert first.latent_spread is None
    second = table[0, 1]
    assert second.vote_rate == 0.75
    assert second.vote_shares == pytest.approx((2 / 3, 1 / 3, 0.0, 0.0, 0.0))
    assert second.typical_score == -2
    assert table[1, 0].vote_rate == 1.0
    assert table[1, 0].typical_score == -1
    assert table.n_votes.tolist() == [[4.0, 3.0], [3.0, 4.0]]


def test_vote_counts_of_a_cocluster_are_ready_for_the_ordinal_family() -> None:
    grouped = make_grouped()
    counts = grouped.vote_counts(0, 1)
    assert isinstance(counts, VoteCounts)
    assert counts == VoteCounts((2, 1, 0, 0, 0), "group 0 x block 1")
    assert counts.total == grouped[0, 1].n_votes == 3
    assert all(type(count) is int for count in counts.counts)
    with pytest.raises(InvalidInputError, match="2 groups and 2 blocks, got group 2"):
        grouped.vote_counts(2, 0)
    with pytest.raises(InvalidInputError, match="got group True"):
        grouped.vote_counts(True, 0)


def test_the_submatrix_of_a_cocluster() -> None:
    grouped = make_grouped()
    cocluster = grouped[0, 1]
    assert isinstance(cocluster, VoteMatrix)
    assert cocluster.shape == (2, 2)
    assert cocluster.voters == ("a", "b")
    assert cocluster.items == ("r", "s")
    assert cocluster.n_votes == 3
    assert grouped[1, 0].n_shown == 3
    with pytest.raises(IndexError, match="2 groups and 2 blocks"):
        grouped[0, 2]
    with pytest.raises(IndexError, match=r"grouped\[group, block\]"):
        grouped[0]  # type: ignore[index]


def test_group_by_without_argument_is_one_cocluster_over_the_whole_matrix() -> None:
    matrix = make_grouped().votes
    grouped = matrix.group_by()
    assert grouped.shape == (1, 1)
    assert grouped[0, 0] == matrix
    assert grouped.counts().sum() == matrix.n_shown
    assert grouped.vote_counts(0, 0).total == matrix.n_votes
    assert grouped.groups[0].voters == matrix.voters
    assert grouped.blocks[0].items == matrix.items

    by_groups = matrix.group_by(groups=Partition.from_voter_labels(VOTERS, [0, 0, 1, 1]))
    assert by_groups.shape == (2, 1)
    by_blocks = matrix.group_by(blocks=Partition.from_item_labels(ITEMS, [0, 1, 1, 1]))
    assert by_blocks.shape == (1, 2)


def test_group_by_an_empty_matrix() -> None:
    grouped = VoteMatrix.from_records([]).group_by()
    assert grouped.shape == (1, 1)
    assert grouped.counts().tolist() == [[[0.0] * 6]]
    table = grouped.to_table()
    assert table[0, 0].vote_rate == 0.0
    assert table[0, 0].vote_shares == (0.2, 0.2, 0.2, 0.2, 0.2)
    assert table[0, 0].typical_score == 0


def test_a_partition_that_does_not_cover_the_matrix_is_refused() -> None:
    matrix = make_grouped().votes
    with pytest.raises(InvalidInputError, match="does not cover the voters") as caught:
        matrix.group_by(groups=Partition.from_voter_labels(["a", "b", "c"], [0, 0, 1]))
    assert "Slice the votes first" in str(caught.value)
    with pytest.raises(InvalidInputError, match="does not cover the voters"):
        matrix.group_by(groups=Partition.from_voter_labels(["b", "a", "c", "d"], [0, 0, 1, 1]))
    with pytest.raises(InvalidInputError, match="does not cover the items"):
        matrix.group_by(blocks=Partition.from_item_labels(["p", "q"], [0, 1]))
    with pytest.raises(InvalidInputError, match="Partition of VoterGroup"):
        matrix.group_by(groups=Partition.from_item_labels(VOTERS, [0, 0, 1, 1]))  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError, match="votes are a VoteMatrix"):
        GroupedVotes("votes", None, None)  # type: ignore[arg-type]


def test_lineage_cites_the_computations_the_partitions_come_from() -> None:
    matrix = make_grouped().votes
    assert make_grouped().lineage == ()
    groups = Partition.from_voter_labels(VOTERS, [0, 0, 1, 1], computation="aaaaaaaaaaaaaaaa")
    blocks = Partition.from_item_labels(ITEMS, [0, 0, 1, 1], computation="bbbbbbbbbbbbbbbb")
    assert matrix.group_by(groups).lineage == ("aaaaaaaaaaaaaaaa",)
    assert matrix.group_by(groups, blocks).lineage == ("aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb")
    same = Partition.from_item_labels(ITEMS, [0, 0, 1, 1], computation="aaaaaaaaaaaaaaaa")
    assert matrix.group_by(groups, same).lineage == ("aaaaaaaaaaaaaaaa",)


def test_vote_counts_need_a_scale_of_five_scores() -> None:
    three = VoteMatrix.from_table([[1, 3], [2, None]], scale=OrdinalScale((1, 2, 3)))
    with pytest.raises(InvalidInputError, match=r"five scores, got 3 scores: \(1, 2, 3\)"):
        three.group_by().vote_counts(0, 0)
    assert three.group_by().to_table()[0, 0].vote_shares == pytest.approx((1 / 3, 1 / 3, 1 / 3))


def test_fingerprint_follows_the_votes_and_the_two_partitions() -> None:
    grouped = make_grouped()
    readable: Fingerprinted = grouped
    assert re.fullmatch(r"[0-9a-f]{16}", readable.fingerprint())
    assert grouped.fingerprint() == make_grouped().fingerprint()
    matrix = grouped.votes
    other_groups = matrix.group_by(
        Partition.from_voter_labels(VOTERS, [0, 1, 1, 1]), grouped.blocks
    )
    other_votes = (
        matrix[:, :].hold_out(np.eye(4, dtype=bool)).group_by(grouped.groups, grouped.blocks)
    )
    assert len({grouped.fingerprint(), other_groups.fingerprint(), other_votes.fingerprint()}) == 3


def test_description() -> None:
    assert repr(make_grouped()) == (
        "GroupedVotes(4 voters in 2 groups x 4 items in 2 blocks, 14 votes)"
    )
