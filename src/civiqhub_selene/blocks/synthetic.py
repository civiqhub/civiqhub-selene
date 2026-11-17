# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Synthetic votes with a planted truth.

The engine is tested and documented on synthetic data only. :func:`make_votes` draws a
collective whose truth is known: who belongs to which group, which item to which block,
and how each group answers each block. No identifier resembles real data: voters are
``v1, v2, ...`` and items ``i1, i2, ...``.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from civiqhub_selene.blocks._arrays import FloatArray, IntArray
from civiqhub_selene.blocks.data import OrdinalScale, VoteMatrix
from civiqhub_selene.errors import InvalidParameterError

__all__ = [
    "DEFAULT_LATENT_MEAN",
    "DEFAULT_LATENT_SPREAD",
    "DEFAULT_VOTE_RATE",
    "PlantedTruth",
    "make_votes",
    "single_memberships",
]

DEFAULT_VOTE_RATE = ((0.85, 0.45), (0.55, 0.90), (0.40, 0.40))
"""Share of cells that carry a vote, for 3 groups (rows) and 2 blocks (columns)."""

DEFAULT_LATENT_MEAN = ((4.7, 1.4), (1.3, 4.6), (3.0, 3.0))
"""Mean of the latent opinion on the axis of levels, from 1 to 5.

Group 0 approves block 0 and rejects block 1, group 1 does the opposite, group 2 is
neutral.
"""

DEFAULT_LATENT_SPREAD = ((0.6, 0.7), (0.7, 0.6), (1.0, 1.0))
"""Standard deviation of the latent opinion, for 3 groups and 2 blocks."""

_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True, eq=False)
class PlantedTruth:
    """The truth planted in a synthetic collective.

    Attributes:
        voter_memberships: Shape (M, Q): degree of membership of each voter to each group.
        item_memberships: Shape (P, L): degree of membership of each item to each block.
        roles: Shape (M, L): the group each voter follows on each block.
    """

    voter_memberships: FloatArray
    item_memberships: FloatArray
    roles: IntArray


def single_memberships(sizes: tuple[int, ...]) -> FloatArray:
    """Return the memberships of identifiers that each belong to a single cluster.

    Rows are indicators, cluster after cluster: ``single_memberships((2, 1))`` is
    ``[[1, 0], [1, 0], [0, 1]]``.

    Raises:
        InvalidParameterError: If a size is not a non-negative integer.
    """
    for size in sizes:
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 0:
            raise InvalidParameterError(
                f"the size of a cluster is a non-negative integer, got {size!r} in {sizes!r}"
            )
    labels = np.repeat(np.arange(len(sizes)), sizes)
    memberships: FloatArray = np.zeros((labels.shape[0], len(sizes)), dtype=np.float64)
    memberships[np.arange(labels.shape[0]), labels] = 1.0
    return memberships


def _memberships(received: ArrayLike, name: str) -> FloatArray:
    try:
        memberships: FloatArray = np.array(received, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InvalidParameterError(f"{name} is a table of numbers: {error}") from error
    if memberships.ndim != 2 or memberships.shape[1] == 0:
        raise InvalidParameterError(
            f"{name} has one row per identifier and one column per cluster, "
            f"got the shape {memberships.shape}"
        )
    if not np.isfinite(memberships).all() or (memberships < 0).any():
        raise InvalidParameterError(f"{name} holds finite numbers that are not negative")
    sums = memberships.sum(axis=1)
    wrong = np.flatnonzero(np.abs(sums - 1.0) > _SUM_TOLERANCE)
    if wrong.size:
        row = int(wrong[0])
        raise InvalidParameterError(
            f"each row of {name} sums to 1, but row {row} sums to {float(sums[row])!r}"
        )
    return memberships


def _parameter(received: ArrayLike, name: str, shape: tuple[int, int]) -> FloatArray:
    try:
        parameter: FloatArray = np.array(received, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InvalidParameterError(f"{name} is a table of numbers: {error}") from error
    if parameter.shape != shape:
        raise InvalidParameterError(
            f"{name} has the shape (groups, blocks) = {shape}, got {parameter.shape}"
        )
    if not np.isfinite(parameter).all():
        raise InvalidParameterError(f"{name} holds finite numbers")
    return parameter


def _draw(uniforms: FloatArray, laws: FloatArray) -> IntArray:
    """Draw one index per uniform number from the law on its row, by inverse cumulative sum."""
    thresholds = np.cumsum(laws, axis=1)[:, np.newaxis, :-1]
    drawn: IntArray = (uniforms[..., np.newaxis] >= thresholds).sum(axis=-1).astype(np.int64)
    return drawn


def make_votes(
    voter_memberships: ArrayLike,
    item_memberships: ArrayLike,
    *,
    vote_rate: ArrayLike = DEFAULT_VOTE_RATE,
    latent_mean: ArrayLike = DEFAULT_LATENT_MEAN,
    latent_spread: ArrayLike = DEFAULT_LATENT_SPREAD,
    n_levels: int = 5,
    seed: int = 0,
) -> tuple[VoteMatrix, PlantedTruth]:
    """Draw synthetic votes from a planted truth. These are not real data.

    With memberships to a single cluster on both sides, this is exactly the latent
    block model; with spread memberships, it is the mixed-membership model.

    The draw is made with ``numpy.random.default_rng(seed)``, in this exact order,
    which is part of the reproducibility contract:

    1. the roles: for each voter then each block, the group the voter follows on this
       block, drawn from the memberships of the voter (uniform numbers of shape (M, L));
    2. the block of each cell: for each item then each voter, drawn from the
       memberships of the item (uniform numbers of shape (P, M));
    3. the co-cluster of a cell is (role of the voter on the block of the cell, block
       of the cell);
    4. whether the cell carries a vote, with the vote rate of its co-cluster (uniform
       numbers of shape (M, P));
    5. the latent opinion of the cell, normal with the mean and the spread of its
       co-cluster (standard normal numbers of shape (M, P));
    6. the level of the vote: 1 plus the number of thresholds 1.5, 2.5, ...,
       ``n_levels - 0.5`` below the opinion; 0 when the cell carries no vote.

    Every cell of the matrix is shown: a cell without a vote is an abstention.

    Args:
        voter_memberships: Shape (M, Q): for each voter, its degree of membership to
            each group. Rows sum to 1. Use :func:`single_memberships` for plain groups.
        item_memberships: Shape (P, L): for each item, its degree of membership to
            each block. Rows sum to 1.
        vote_rate: Shape (Q, L): share of the cells of each co-cluster that carry a
            vote, between 0 and 1.
        latent_mean: Shape (Q, L): where each group stands on each block, on the axis
            of levels from 1 to ``n_levels``.
        latent_spread: Shape (Q, L): how much the opinions of a group vary on a block,
            strictly positive.
        n_levels: The number of scores of the scale. With 5, the scale is the
            agreement scale from -2 to +2; otherwise the scores are 1 to ``n_levels``.
        seed: Seed of every draw, a non-negative integer.

    Returns:
        The vote matrix, with voters ``v1, v2, ...`` and items ``i1, i2, ...``, and
        the planted truth.

    Raises:
        InvalidParameterError: If a row of memberships does not sum to 1, if a
            parameter table does not have the shape (Q, L), if a rate is outside
            [0, 1], if a spread is not strictly positive, or if ``n_levels`` or
            ``seed`` is out of range.
    """
    voters = _memberships(voter_memberships, "voter_memberships")
    items = _memberships(item_memberships, "item_memberships")
    shape = (voters.shape[1], items.shape[1])
    rates = _parameter(vote_rate, "vote_rate", shape)
    means = _parameter(latent_mean, "latent_mean", shape)
    spreads = _parameter(latent_spread, "latent_spread", shape)
    if ((rates < 0) | (rates > 1)).any():
        raise InvalidParameterError("a vote rate is between 0 and 1")
    if (spreads <= 0).any():
        raise InvalidParameterError("a latent spread is strictly positive")
    if isinstance(n_levels, bool) or not isinstance(n_levels, int) or n_levels < 2:
        raise InvalidParameterError(f"n_levels is an integer of at least 2, got {n_levels!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise InvalidParameterError(f"seed must be a non-negative integer, got {seed!r}")

    n_voters, n_items = voters.shape[0], items.shape[0]
    n_blocks = shape[1]
    rng = np.random.default_rng(seed)
    roles = _draw(rng.random((n_voters, n_blocks)), voters)
    cell_block = _draw(rng.random((n_items, n_voters)), items).T
    cell_group = np.take_along_axis(roles, cell_block, axis=1)
    carries_vote = rng.random((n_voters, n_items)) < rates[cell_group, cell_block]
    opinion = means[cell_group, cell_block] + spreads[cell_group, cell_block] * (
        rng.standard_normal((n_voters, n_items))
    )
    thresholds = np.arange(1, n_levels) + 0.5
    levels = 1 + (opinion[..., np.newaxis] > thresholds).sum(axis=-1)
    levels = np.where(carries_vote, levels, 0).astype(np.int64)

    scale = OrdinalScale() if n_levels == 5 else OrdinalScale(tuple(range(1, n_levels + 1)))
    matrix = VoteMatrix.from_arrays(levels, scale=scale)
    for array in (voters, items, roles):
        array.setflags(write=False)
    return matrix, PlantedTruth(voters, items, roles)
