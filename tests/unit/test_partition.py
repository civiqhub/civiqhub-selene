# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import re
from typing import Any

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene.blocks.partition import Cluster, ItemBlock, Partition, VoterGroup
from civiqhub_selene.errors import InvalidInputError, InvalidParameterError

pytestmark = pytest.mark.blocks


def test_a_partition_of_voters_is_a_sequence_of_groups() -> None:
    partition = Partition.from_voter_labels(["a", "b", "c"], [0, 1, 0])
    assert len(partition) == 2
    assert partition.n_clusters == 2
    assert partition[0].voters == ("a", "c")
    assert partition[0].share == 2 / 3
    assert "a" in partition[0]
    assert "b" not in partition[0]
    assert partition.cluster_of("b").index == 1
    assert isinstance(partition[0], VoterGroup)
    assert partition.kind is VoterGroup
    assert partition.sizes == (2, 1)
    assert [group.index for group in partition] == [0, 1]
    assert partition[-1] is partition[1]
    assert partition[0:2] == (partition[0], partition[1])
    assert partition[0] is partition[0]
    with pytest.raises(IndexError):
        partition[2]


def test_a_partition_of_items_is_a_sequence_of_blocks() -> None:
    partition = Partition.from_item_labels(("p", "q", "r"), np.array([1, 1, 0]))
    assert isinstance(partition[1], ItemBlock)
    assert partition[1].items == ("p", "q")
    assert partition[0].items == ("r",)
    assert partition.kind is ItemBlock
    assert not hasattr(partition[0], "voters")


def test_a_cluster_is_a_set_of_identifiers() -> None:
    partition = Partition.from_voter_labels(["a", "b", "c", "d"], [0, 0, 1, 1])
    first, second = partition
    assert len(first) == 2
    assert list(first) == ["a", "b"]
    assert first & {"a", "z"} == frozenset({"a"})
    assert first | second == frozenset({"a", "b", "c", "d"})
    assert isinstance(first | second, frozenset)
    assert first - {"a"} == frozenset({"b"})
    assert first.isdisjoint(second)
    assert first <= {"a", "b", "c"}
    assert first == VoterGroup(0, ("a", "b"), 0.5)
    assert first != VoterGroup(1, ("a", "b"), 0.5)
    assert hash(first) == hash(VoterGroup(0, ("a", "b"), 0.5))
    assert repr(first) == "VoterGroup(index=0, 2 voters, share=0.500)"
    assert repr(ItemBlock(1, ("p",), 0.25)) == "ItemBlock(index=1, 1 item, share=0.250)"
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.index = 3  # type: ignore[misc]


def test_cluster_of_an_unknown_identifier_is_refused() -> None:
    partition = Partition.from_voter_labels(["a", "b"], [0, 1])
    with pytest.raises(InvalidInputError, match="voter 'z' is not in the partition"):
        partition.cluster_of("z")
    with pytest.raises(InvalidInputError, match="voter 3 is not in the partition"):
        partition.cluster_of(3)  # type: ignore[arg-type]


def test_agreement_ignores_the_numbering_of_clusters() -> None:
    ids = ["a", "b", "c"]
    partition = Partition.from_voter_labels(ids, [0, 1, 0])
    permuted = Partition.from_voter_labels(ids, [1, 0, 1])
    moved = Partition.from_voter_labels(ids, [0, 1, 1])
    assert partition.agreement_with(permuted) == 1.0
    assert partition.agreement_with(partition) == 1.0
    assert partition.agreement_with(moved) == 2 / 3
    assert moved.agreement_with(partition) == 2 / 3


def test_agreement_between_partitions_with_different_numbers_of_clusters() -> None:
    ids = ["a", "b", "c", "d"]
    two = Partition.from_voter_labels(ids, [0, 0, 1, 1])
    three = Partition.from_voter_labels(ids, [2, 2, 0, 1])
    one = Partition.from_voter_labels(ids, [0, 0, 0, 0])
    assert two.agreement_with(three) == 0.75
    assert three.agreement_with(two) == 0.75
    assert two.agreement_with(one) == 0.5


def test_agreement_needs_the_same_identifiers_and_at_most_eight_clusters() -> None:
    partition = Partition.from_voter_labels(["a", "b", "c"], [0, 1, 0])
    other_order = Partition.from_voter_labels(["b", "a", "c"], [1, 0, 0])
    with pytest.raises(InvalidInputError, match="same identifiers in the same order"):
        partition.agreement_with(other_order)
    with pytest.raises(InvalidInputError, match="same identifiers"):
        partition.agreement_with("partition")  # type: ignore[arg-type]
    ids = [f"v{number}" for number in range(9)]
    nine = Partition.from_voter_labels(ids, list(range(9)))
    with pytest.raises(InvalidParameterError, match="stops at 8 clusters, got 9"):
        nine.agreement_with(nine)
    eight = Partition.from_voter_labels(ids[:8], list(range(8)))
    assert eight.agreement_with(eight) == 1.0
    empty = Partition.from_voter_labels([], [])
    with pytest.raises(InvalidInputError, match="without identifier"):
        empty.agreement_with(empty)


@pytest.mark.parametrize(
    ("ids", "labels", "options", "named"),
    [
        (["a", "b"], [0, 2], {"n_clusters": 2}, "got the label 2"),
        (["a", "b"], [0, -1], {}, "got the label -1"),
        (["a", "b"], [0], {}, "one label per voter"),
        (["a", "b", "c"], [0, 1], {}, "one label per voter"),
        (["a", "a"], [0, 1], {}, "'a' is repeated"),
        (["a", 2], [0, 1], {}, "identifier is text, got 2"),
        ("ab", [0, 1], {}, "collection"),
        (["a", "b"], [0.0, 1.0], {}, "labels are integers"),
        (["a", "b"], [[0, 1]], {}, "one label per voter"),
        (["a", "b"], [0, 1], {"n_clusters": 2.0}, "n_clusters is an integer"),
        (["a", "b"], [0, 1], {"memberships": [[1.0, 0.0], [0.4, 0.5]]}, "'b' sum to 0.9"),
        (["a", "b"], [0, 1], {"memberships": [[1.0, 0.0]]}, "memberships have the shape"),
        (["a", "b"], [0, 1], {"memberships": [[1.5, -0.5], [0.0, 1.0]]}, "not negative"),
        (["a", "b"], [0, 1], {"shares": [1.0]}, "one share per cluster"),
        (["a", "b"], [0, 1], {"shares": [1.5, -0.5]}, "not negative"),
        (["a", "b"], [0, 1], {"computation": 12}, "fingerprint in text"),
    ],
)
def test_invalid_partitions_are_refused_and_named(
    ids: list[str], labels: list[int], options: dict[str, object], named: str
) -> None:
    with pytest.raises(InvalidInputError, match=re.escape(named)):
        Partition.from_voter_labels(ids, labels, **options)  # type: ignore[arg-type]


def test_the_kind_is_a_cluster_class() -> None:
    with pytest.raises(InvalidInputError, match="VoterGroup or ItemBlock"):
        Partition(str, ("a",), np.array([0]), 1)  # type: ignore[type-var]
    generic = Partition(Cluster, ("a", "b"), np.array([0, 0]), 1)
    assert repr(generic) == "Partition(2 ids in 1 cluster: 2)"


def test_an_empty_cluster_exists_with_a_zero_share() -> None:
    partition = Partition.from_voter_labels(["a", "b", "c"], [0, 1, 0], n_clusters=3)
    assert len(partition) == 3
    assert partition[2].voters == ()
    assert len(partition[2]) == 0
    assert partition[2].share == 0.0
    assert partition.sizes == (2, 1, 0)


def test_a_partition_is_mixed_when_memberships_are_given() -> None:
    plain = Partition.from_voter_labels(["a", "b"], [0, 1])
    assert plain.is_mixed is False
    assert plain.memberships is None
    mixed = Partition.from_voter_labels(
        ["a", "b"], [0, 1], memberships=[[0.75, 0.25], [0.25, 0.75]]
    )
    assert mixed.is_mixed is True
    assert mixed.memberships is not None
    assert mixed.memberships.tolist() == [[0.75, 0.25], [0.25, 0.75]]
    # Shares default to the mean of the memberships.
    assert mixed[0].share == 0.5
    # Every voter leans to group 0, yet the memberships say there are two groups.
    skewed = Partition.from_voter_labels(["a", "b"], [0, 0], memberships=[[0.6, 0.4], [0.8, 0.2]])
    assert skewed.n_clusters == 2
    assert skewed.sizes == (2, 0)
    assert skewed[1].share == pytest.approx(0.3)


def test_shares_default_to_sizes_or_follow_the_ones_given() -> None:
    sized = Partition.from_item_labels(["p", "q", "r", "s"], [0, 0, 0, 1])
    assert sized.shares is not None
    assert sized.shares.tolist() == [0.75, 0.25]
    given = Partition.from_item_labels(["p", "q", "r", "s"], [0, 0, 0, 1], shares=[0.6, 0.4])
    assert given[0].share == 0.6
    mixed = Partition.from_item_labels(
        ["p", "q"], [0, 0], n_clusters=2, memberships=[[0.6, 0.4], [0.8, 0.2]]
    )
    assert mixed.shares is not None
    assert mixed.shares.tolist() == pytest.approx([0.7, 0.3])


def test_fingerprint_follows_the_arrangement() -> None:
    partition = Partition.from_voter_labels(["a", "b", "c"], [0, 1, 0])
    same = Partition.from_voter_labels(("a", "b", "c"), np.array([0, 1, 0], dtype=np.int8))
    assert re.fullmatch(r"[0-9a-f]{16}", partition.fingerprint())
    assert partition.fingerprint() == same.fingerprint()
    assert partition == same
    assert hash(partition) == hash(same)
    others: list[Partition[Any]] = [
        Partition.from_voter_labels(["a", "b", "c"], [0, 1, 1]),
        Partition.from_voter_labels(["a", "b", "d"], [0, 1, 0]),
        Partition.from_voter_labels(["a", "b", "c"], [0, 1, 0], n_clusters=3),
        Partition.from_item_labels(["a", "b", "c"], [0, 1, 0]),
    ]
    assert len({partition.fingerprint(), *(other.fingerprint() for other in others)}) == 5
    assert all(partition != other for other in others)
    assert partition != "partition"


def test_equality_also_reads_memberships_shares_and_computation() -> None:
    plain = Partition.from_voter_labels(["a", "b"], [0, 1])
    mixed = Partition.from_voter_labels(["a", "b"], [0, 1], memberships=[[0.9, 0.1], [0.2, 0.8]])
    computed = Partition.from_voter_labels(["a", "b"], [0, 1], computation="0123456789abcdef")
    assert plain.fingerprint() == mixed.fingerprint() == computed.fingerprint()
    assert plain != mixed
    assert plain != computed
    assert computed.computation == "0123456789abcdef"
    assert plain.computation is None


def test_arrays_are_copied_and_read_only() -> None:
    labels = np.array([0, 1, 0])
    partition = Partition.from_voter_labels(["a", "b", "c"], labels)
    labels[0] = 1
    assert partition.labels.tolist() == [0, 1, 0]
    assert partition.shares is not None
    for array in (partition.labels, partition.shares):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        partition.n_clusters = 5  # type: ignore[misc]


def test_description() -> None:
    labels = [0] * 20 + [1] * 16 + [2] * 12
    ids = [f"v{number}" for number in range(1, 49)]
    partition = Partition.from_voter_labels(ids, labels)
    assert repr(partition) == "Partition(48 voters in 3 groups: 20, 16, 12)"
    assert str(partition) == repr(partition)
    blocks = Partition.from_item_labels(["p"], [0])
    assert repr(blocks) == "Partition(1 item in 1 block: 1)"
    mixed = Partition.from_voter_labels(["a", "b"], [0, 1], memberships=[[0.9, 0.1], [0.2, 0.8]])
    assert repr(mixed) == "Partition(2 voters in 2 mixed groups: 1, 1)"
    assert repr(Partition.from_voter_labels([], [])) == "Partition(0 voters in 0 groups: )"
