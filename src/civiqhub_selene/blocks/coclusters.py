# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Co-clusters and their table.

A co-cluster is the pair of a group of voters and a block of items: the zone of the
matrix where this group answers this block. :class:`CoclusterTable` summarises each
co-cluster by a vote rate and a distribution of the answers.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from civiqhub_selene.blocks._arrays import FloatArray
from civiqhub_selene.blocks.data import OrdinalScale
from civiqhub_selene.errors import InvalidInputError
from civiqhub_selene.types import Displayable

__all__ = ["Cocluster", "CoclusterTable"]

_MIN_PROBABILITY = 1e-300
_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True, repr=False)
class Cocluster:
    """One co-cluster: how a group of voters answers a block of items.

    Attributes:
        group: Index of the group of voters.
        block: Index of the block of items.
        vote_rate: Share of the shown cells that carry a vote.
        vote_shares: Distribution of the votes over the scores of the scale, summing to 1.
        n_votes: Number of votes: an integer for plain groups, fractional for a mixed
            membership.
        latent_mean: Mean of the latent opinion under the ordinal probit response
            model, on the axis of levels; ``None`` under another response model.
        latent_spread: Standard deviation of the latent opinion, likewise.
        scores: The scores of the scale.
    """

    group: int
    block: int
    vote_rate: float
    vote_shares: tuple[float, ...]
    n_votes: float
    latent_mean: float | None
    latent_spread: float | None
    scores: tuple[int, ...]

    @property
    def typical_score(self) -> int:
        """The score with the largest share of votes.

        On a tie, the score closest to the middle of the scale; if two tied scores are
        equally close, the lower one.
        """
        middle = (len(self.scores) - 1) / 2
        best = max(
            range(len(self.scores)),
            key=lambda rank: (self.vote_shares[rank], -abs(rank - middle), -rank),
        )
        return self.scores[best]

    def __repr__(self) -> str:
        typical = self.typical_score
        return (
            f"Cocluster(group={self.group}, block={self.block}, "
            f"vote_rate={self.vote_rate:.3f}, typical_score={typical:+d}, "
            f"n_votes={self.n_votes:g})"
        )


def state_log_probabilities(vote_rate: FloatArray, vote_shares: FloatArray) -> FloatArray:
    """Return the log-probability of each state of a shown cell, per co-cluster.

    Shape (Q, L, K + 1) from a vote rate of shape (Q, L) and vote shares of shape
    (Q, L, K). State 0 (shown without a vote): ``log(1 - vote_rate)``; state ``k``:
    ``log(vote_rate) + log(vote_shares[k])``. Probabilities are floored at 1e-300
    before the logarithm.
    """
    rate = vote_rate[..., np.newaxis]
    probabilities = np.concatenate([1.0 - rate, rate * vote_shares], axis=-1)
    logs: FloatArray = np.log(np.maximum(probabilities, _MIN_PROBABILITY))
    return logs


def _array(values: ArrayLike, name: str) -> FloatArray:
    try:
        array: FloatArray = np.array(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InvalidInputError(f"{name} is an array of numbers: {error}") from error
    if not np.isfinite(array).all():
        raise InvalidInputError(f"{name} holds a value that is not finite")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False, repr=False)
class CoclusterTable(Displayable):
    """The table of the co-clusters of Q groups and L blocks.

    ``table[q, l]`` is one :class:`Cocluster`; iteration yields every co-cluster,
    group by group. Arrays are copied and made read-only.

    Attributes:
        scale: The scale of scores, with K levels.
        vote_rate: Shape (Q, L): share of the shown cells that carry a vote.
        vote_shares: Shape (Q, L, K): distribution of the votes over the scores.
        n_votes: Shape (Q, L): number of votes.
        latent_mean: Shape (Q, L): mean of the latent opinion under the ordinal probit
            response model; ``None`` otherwise.
        latent_spread: Shape (Q, L): standard deviation of the latent opinion, likewise.

    Raises:
        InvalidInputError: If the shapes do not agree, if a rate is outside [0, 1], if
            a distribution does not sum to 1 within 1e-6, or if a count is negative.
    """

    scale: OrdinalScale
    vote_rate: FloatArray
    vote_shares: FloatArray
    n_votes: FloatArray
    latent_mean: FloatArray | None = None
    latent_spread: FloatArray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scale, OrdinalScale):
            raise InvalidInputError(f"the scale is an OrdinalScale, got {self.scale!r}")
        vote_rate = _array(self.vote_rate, "vote_rate")
        vote_shares = _array(self.vote_shares, "vote_shares")
        n_votes = _array(self.n_votes, "n_votes")
        if vote_rate.ndim != 2:
            raise InvalidInputError(
                f"vote_rate has the shape (groups, blocks), got {vote_rate.shape}"
            )
        expected = (*vote_rate.shape, self.scale.n_levels)
        if vote_shares.shape != expected:
            raise InvalidInputError(
                f"vote_shares has the shape (groups, blocks, levels) = {expected}, "
                f"got {vote_shares.shape}"
            )
        if n_votes.shape != vote_rate.shape:
            raise InvalidInputError(
                f"n_votes has the shape {vote_rate.shape} of vote_rate, got {n_votes.shape}"
            )
        if ((vote_rate < 0) | (vote_rate > 1)).any():
            raise InvalidInputError("a vote rate is between 0 and 1")
        if (vote_shares < 0).any() or (
            np.abs(vote_shares.sum(axis=-1) - 1.0) > _SUM_TOLERANCE
        ).any():
            raise InvalidInputError("the vote shares of a co-cluster are not negative and sum to 1")
        if (n_votes < 0).any():
            raise InvalidInputError("a number of votes is not negative")
        object.__setattr__(self, "vote_rate", vote_rate)
        object.__setattr__(self, "vote_shares", vote_shares)
        object.__setattr__(self, "n_votes", n_votes)
        for name in ("latent_mean", "latent_spread"):
            received = getattr(self, name)
            if received is None:
                continue
            latent = _array(received, name)
            if latent.shape != vote_rate.shape:
                raise InvalidInputError(
                    f"{name} has the shape {vote_rate.shape} of vote_rate, got {latent.shape}"
                )
            object.__setattr__(self, name, latent)

    @property
    def shape(self) -> tuple[int, int]:
        """The number of groups and the number of blocks."""
        n_groups, n_blocks = self.vote_rate.shape
        return (n_groups, n_blocks)

    def __getitem__(self, key: tuple[int, int]) -> Cocluster:
        """Return the co-cluster of the group ``q`` and the block ``l``.

        Raises:
            IndexError: If the group or the block does not exist.
        """
        n_groups, n_blocks = self.shape
        try:
            group, block = key
            inside = 0 <= group < n_groups and 0 <= block < n_blocks
        except (TypeError, ValueError):
            inside = False
        if not inside:
            raise IndexError(
                f"a co-cluster is read as table[group, block] with group < {n_groups} "
                f"and block < {n_blocks}, got {key!r}"
            )
        return Cocluster(
            group=int(group),
            block=int(block),
            vote_rate=float(self.vote_rate[group, block]),
            vote_shares=tuple(float(share) for share in self.vote_shares[group, block]),
            n_votes=float(self.n_votes[group, block]),
            latent_mean=None if self.latent_mean is None else float(self.latent_mean[group, block]),
            latent_spread=(
                None if self.latent_spread is None else float(self.latent_spread[group, block])
            ),
            scores=self.scale.scores,
        )

    def __iter__(self) -> Iterator[Cocluster]:
        n_groups, n_blocks = self.shape
        for group in range(n_groups):
            for block in range(n_blocks):
                yield self[group, block]

    def __len__(self) -> int:
        n_groups, n_blocks = self.shape
        return n_groups * n_blocks

    def log_probabilities(self) -> FloatArray:
        """Return the log-probability of each state of a shown cell, per co-cluster.

        Shape (Q, L, K + 1). State 0 (shown without a vote): ``log(1 - vote_rate)``;
        state ``k``: ``log(vote_rate) + log(vote_shares[k])``. Probabilities are
        floored at 1e-300 before the logarithm. With ``nu`` the vote rate and ``p``
        the vote shares, this is ``log F[q, l, s]``.

        See the mathematical model: :doc:`/model/response-models`, equation
        :eq:`eq-response-states`.
        """
        return state_log_probabilities(self.vote_rate, self.vote_shares)

    def to_dict(self) -> dict[str, Any]:
        """Return plain lists: the scores, the three tables, and the latent parameters if any."""
        data: dict[str, Any] = {
            "scores": list(self.scale.scores),
            "vote_rate": self.vote_rate.tolist(),
            "vote_shares": self.vote_shares.tolist(),
            "n_votes": self.n_votes.tolist(),
        }
        if self.latent_mean is not None:
            data["latent_mean"] = self.latent_mean.tolist()
        if self.latent_spread is not None:
            data["latent_spread"] = self.latent_spread.tolist()
        return data

    def _describe(self) -> list[str]:
        n_groups, n_blocks = self.shape
        lines = [
            f"CoclusterTable({n_groups} group{'' if n_groups == 1 else 's'} x "
            f"{n_blocks} block{'' if n_blocks == 1 else 's'}, {self.scale.n_levels} levels)"
        ]
        for cocluster in self:
            typical = self.scale.format_score(cocluster.typical_score)
            lines.append(
                f"group {cocluster.group} x block {cocluster.block}: "
                f"vote rate {cocluster.vote_rate:.2f}, typical score {typical}, "
                f"{cocluster.n_votes:g} votes"
            )
        return lines
