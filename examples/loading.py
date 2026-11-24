# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Read the CSV files of the examples into the types of the engine.

The engine reads no file: it receives a ``VoteMatrix``. These two functions show how
to build one from a table of votes, keeping the three states of a cell:

- a row with a score is a vote;
- a row with an empty score is a cell shown without a vote;
- a pair (voter, item) without a row was not shown.
"""

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from civiqhub_selene.blocks import ItemBlock, Partition, VoteMatrix, VoterGroup

__all__ = ["planted_blocks", "planted_groups", "read_memberships", "read_votes"]


def read_votes(path: str | Path) -> VoteMatrix:
    """Read a ``voter,item,score`` file into a vote matrix.

    Voters and items follow their order of first appearance in the file.
    """
    votes: list[tuple[str, str, int]] = []
    shown: list[tuple[str, str]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            shown.append((row["voter"], row["item"]))
            if row["score"].strip():
                votes.append((row["voter"], row["item"], int(row["score"])))
    return VoteMatrix.from_records(votes, shown=shown)


def read_memberships(path: str | Path) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    """Read a file of planted memberships: one row per identifier, one column per cluster.

    Returns:
        The identifiers, and the memberships of shape (n, number of clusters).
    """
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))[1:]
    identifiers = tuple(row[0] for row in rows)
    memberships = np.array([[float(value) for value in row[1:]] for row in rows])
    return identifiers, memberships


def planted_groups(votes: VoteMatrix, path: str | Path) -> Partition[VoterGroup]:
    """Return the main planted group of each voter of ``votes``, as a partition."""
    identifiers, memberships = read_memberships(path)
    position = {identifier: index for index, identifier in enumerate(identifiers)}
    labels = [int(memberships[position[voter]].argmax()) for voter in votes.voters]
    return Partition.from_voter_labels(votes.voters, labels)


def planted_blocks(votes: VoteMatrix, path: str | Path) -> Partition[ItemBlock]:
    """Return the main planted block of each item of ``votes``, as a partition."""
    identifiers, memberships = read_memberships(path)
    position = {identifier: index for index, identifier in enumerate(identifiers)}
    labels = [int(memberships[position[item]].argmax()) for item in votes.items]
    return Partition.from_item_labels(votes.items, labels)
