# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Draw the synthetic collectives of the examples and write them as CSV files.

The data are synthetic: they are drawn by ``make_votes`` from a planted truth, and no
identifier resembles real data. Run this script from the repository root to write the
files again; it overwrites them with the same content::

    python examples/data/make_data.py

Two collectives are written, each as three files:

- ``<name>.votes.csv``: one row per shown cell, ``voter,item,score``; an empty score is
  a cell shown without a vote; a cell that is not shown has no row;
- ``<name>.voters.csv``: the planted membership of each voter to each group;
- ``<name>.items.csv``: the planted membership of each item to each block.

``consultation`` has 240 voters in 3 groups and 24 items in 3 blocks, each voter seeing
between 35% and 100% of the items. On the third block, groups 0 and 1 have the same mean
opinion, one in agreement and one split. ``assembly`` has 150 voters, 30 of whom share
their membership between two groups, and 20 items, 4 of which share theirs between the
two blocks.
"""

import csv
from pathlib import Path

import numpy as np

from civiqhub_selene.blocks import VoteMatrix
from civiqhub_selene.blocks.synthetic import make_votes, single_memberships

HERE = Path(__file__).resolve().parent


def write(name: str, votes: VoteMatrix, voters: np.ndarray, items: np.ndarray) -> None:
    with (HERE / f"{name}.votes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["voter", "item", "score"])
        for row, column in np.argwhere(votes.shown):
            level = int(votes.levels[row, column])
            score = votes.scale.score_of(level) if level else ""
            writer.writerow([votes.voters[row], votes.items[column], score])
    for side, ids, memberships, prefix in (
        ("voters", votes.voters, voters, "group"),
        ("items", votes.items, items, "block"),
    ):
        with (HERE / f"{name}.{side}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow([side[:-1], *(f"{prefix}_{k}" for k in range(memberships.shape[1]))])
            for identifier, row in zip(ids, memberships, strict=True):
                writer.writerow([identifier, *(f"{weight:g}" for weight in row)])


def consultation() -> None:
    voters = single_memberships((100, 80, 60))
    items = single_memberships((8, 8, 8))
    votes, _ = make_votes(
        voters,
        items,
        vote_rate=[[0.85, 0.45, 0.70], [0.55, 0.90, 0.70], [0.40, 0.40, 0.80]],
        latent_mean=[[4.7, 1.4, 3.0], [1.3, 4.6, 3.0], [3.0, 3.0, 4.5]],
        latent_spread=[[0.6, 0.7, 0.5], [0.7, 0.6, 2.2], [1.0, 1.0, 0.6]],
        seed=11,
    )
    # Uneven exposure: each voter sees each item with its own probability.
    rng = np.random.default_rng(12)
    exposure = 0.35 + 0.65 * rng.beta(2.0, 1.5, size=votes.shape[0])
    shown = rng.random(votes.shape) < exposure[:, np.newaxis]
    write("consultation", VoteMatrix.from_arrays(votes.levels, shown=shown), voters, items)


def assembly() -> None:
    voters = np.vstack(
        [single_memberships((40, 40, 40)), [[0.5, 0.5, 0.0]] * 15, [[0.0, 0.5, 0.5]] * 15]
    )
    items = np.vstack([single_memberships((8, 8)), [[0.5, 0.5]] * 4])
    votes, _ = make_votes(voters, items, seed=21)
    rng = np.random.default_rng(22)
    shown = rng.random(votes.shape) < 0.9
    write("assembly", VoteMatrix.from_arrays(votes.levels, shown=shown), voters, items)


if __name__ == "__main__":
    consultation()
    assembly()
