# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Co-clusters and their table.

A co-cluster is the pair of a group of voters and a block of items: the zone of the
matrix where this group answers this block. :class:`CoclusterTable` summarises each
co-cluster by a vote rate and a distribution of the answers on each scale.

When the items use several scales, a co-cluster has one distribution per scale: the
readings of one scale (``scale``, ``vote_shares``, ``typical_score``) refuse, and the
readings by scale (``scales``, ``vote_shares_by_scale``, ``typical_score_by_scale``)
answer.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike

from civiqhub_selene.blocks._arrays import BoolArray, FloatArray
from civiqhub_selene.blocks.data import OrdinalScale, _only_scale
from civiqhub_selene.errors import InvalidInputError
from civiqhub_selene.types import Displayable

__all__ = ["Cocluster", "CoclusterTable"]

_MIN_PROBABILITY = 1e-300
_SUM_TOLERANCE = 1e-6


def _typical(scores: tuple[int, ...], shares: tuple[float, ...]) -> int:
    """The score with the largest share; on a tie, the closest to the middle, then the lower."""
    middle = (len(scores) - 1) / 2
    best = max(range(len(scores)), key=lambda rank: (shares[rank], -abs(rank - middle), -rank))
    return scores[best]


def _is_by_scale(values: Sequence[Any]) -> bool:
    return len(values) > 0 and all(isinstance(value, (tuple, list)) for value in values)


@dataclass(frozen=True, repr=False, init=False)
class Cocluster:
    """One co-cluster: how a group of voters answers a block of items.

    Args:
        group: Index of the group of voters.
        block: Index of the block of items.
        vote_rate: Share of the shown cells that carry a vote.
        vote_shares: Distribution of the votes over the scores of the scale, summing to
            1; with several scales, one distribution per scale.
        n_votes: Number of votes: an integer for plain groups, fractional for a mixed
            membership.
        latent_mean: Mean of the latent opinion under the ordinal probit response
            model, on the axis of its thresholds; ``None`` under another response model.
        latent_spread: Standard deviation of the latent opinion, likewise.
        scores: The scores of the scale; with several scales, the scores of each.
        has_fixed_spread: Whether the spread was held fixed rather than estimated,
            because every vote of the co-cluster is on a scale of two levels. Defaults
            to ``False``.

    Attributes:
        group: As given.
        block: As given.
        vote_rate: As given.
        vote_shares_by_scale: One distribution of the votes per scale.
        n_votes: As given.
        latent_mean: As given.
        latent_spread: As given.
        scores_by_scale: The scores of each scale.
        has_fixed_spread: As given.
    """

    group: int
    block: int
    vote_rate: float
    vote_shares_by_scale: tuple[tuple[float, ...], ...]
    n_votes: float
    latent_mean: float | None
    latent_spread: float | None
    scores_by_scale: tuple[tuple[int, ...], ...]
    has_fixed_spread: bool

    def __init__(
        self,
        group: int,
        block: int,
        vote_rate: float,
        vote_shares: Sequence[float] | Sequence[Sequence[float]],
        n_votes: float,
        latent_mean: float | None,
        latent_spread: float | None,
        scores: Sequence[int] | Sequence[Sequence[int]],
        *,
        has_fixed_spread: bool = False,
    ) -> None:
        by_scale = _is_by_scale(scores)
        if by_scale != _is_by_scale(vote_shares) and len(scores) and len(vote_shares):
            raise InvalidInputError(
                "vote_shares and scores of a co-cluster are both for one scale, or both "
                "given scale by scale"
            )
        shares: tuple[tuple[float, ...], ...]
        scales: tuple[tuple[int, ...], ...]
        if by_scale:
            nested_shares = cast(Sequence[Sequence[float]], vote_shares)
            nested_scores = cast(Sequence[Sequence[int]], scores)
            shares = tuple(tuple(float(share) for share in one) for one in nested_shares)
            scales = tuple(tuple(int(score) for score in one) for one in nested_scores)
        else:
            shares = (tuple(float(share) for share in cast(Sequence[float], vote_shares)),)
            scales = (tuple(int(score) for score in cast(Sequence[int], scores)),)
        for name, value in (
            ("group", group),
            ("block", block),
            ("vote_rate", vote_rate),
            ("vote_shares_by_scale", shares),
            ("n_votes", n_votes),
            ("latent_mean", latent_mean),
            ("latent_spread", latent_spread),
            ("scores_by_scale", scales),
            ("has_fixed_spread", bool(has_fixed_spread)),
        ):
            object.__setattr__(self, name, value)

    @property
    def vote_shares(self) -> tuple[float, ...]:
        """The distribution of the votes over the scores, when the block has one scale.

        Raises:
            InvalidInputError: If there are several scales: read
                :attr:`vote_shares_by_scale`.
        """
        self._one_scale()
        return self.vote_shares_by_scale[0]

    @property
    def scores(self) -> tuple[int, ...]:
        """The scores of the scale, when there is one.

        Raises:
            InvalidInputError: If there are several scales: read :attr:`scores_by_scale`.
        """
        self._one_scale()
        return self.scores_by_scale[0]

    @property
    def typical_score(self) -> int:
        """The score with the largest share of votes, when there is one scale.

        On a tie, the score closest to the middle of the scale; if two tied scores are
        equally close, the lower one.

        Raises:
            InvalidInputError: If there are several scales: read
                :attr:`typical_score_by_scale`.
        """
        self._one_scale()
        return self.typical_score_by_scale[0]

    @property
    def typical_score_by_scale(self) -> tuple[int, ...]:
        """The typical score on each scale, chosen as :attr:`typical_score` chooses it."""
        return tuple(
            _typical(scores, shares)
            for scores, shares in zip(self.scores_by_scale, self.vote_shares_by_scale, strict=True)
        )

    def _one_scale(self) -> None:
        if len(self.scores_by_scale) != 1:
            _only_scale(tuple(OrdinalScale(s) for s in self.scores_by_scale), "this co-cluster")

    def __repr__(self) -> str:
        if len(self.scores_by_scale) == 1:
            typical = f"typical_score={self.typical_score:+d}"
        else:
            scores = ", ".join(f"{score:+d}" for score in self.typical_score_by_scale)
            typical = f"typical_score_by_scale=({scores})"
        return (
            f"Cocluster(group={self.group}, block={self.block}, "
            f"vote_rate={self.vote_rate:.3f}, {typical}, n_votes={self.n_votes:g})"
        )


def state_log_probabilities(vote_rate: FloatArray, vote_shares: FloatArray) -> FloatArray:
    """Return the log-probability of each state of a shown cell, per co-cluster.

    From a vote rate of shape (Q, L) and vote shares of shape (Q, L, K), the result
    has the shape (Q, L, K + 1); from shares by scale of shape (Q, L, C, K), it has the
    shape (Q, L, C, K + 1). State 0 (shown without a vote): ``log(1 - vote_rate)``;
    state ``k``: ``log(vote_rate) + log(vote_shares[k])``. Probabilities are floored at
    1e-300 before the logarithm, so a state past the levels of a scale, whose share is
    0, has a finite logarithm.
    """
    rate = vote_rate.reshape(vote_rate.shape + (1,) * (vote_shares.ndim - vote_rate.ndim))
    probabilities = np.concatenate(
        [np.broadcast_to(1.0 - rate, (*vote_shares.shape[:-1], 1)), rate * vote_shares], axis=-1
    )
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


def _scales(received: object) -> tuple[tuple[OrdinalScale, ...], bool]:
    """Return the scales of a table, and whether they were given as a collection."""
    if isinstance(received, OrdinalScale):
        return (received,), False
    if isinstance(received, (str, bytes)) or not isinstance(received, Sequence) or not received:
        raise InvalidInputError(
            f"the scale is an OrdinalScale, or a collection of OrdinalScale, got {received!r}"
        )
    for scale in received:
        if not isinstance(scale, OrdinalScale):
            raise InvalidInputError(
                f"the scales of a table are each an OrdinalScale, got {scale!r}"
            )
    return tuple(received), True


@dataclass(frozen=True, eq=False, repr=False, init=False)
class CoclusterTable(Displayable):
    """The table of the co-clusters of Q groups and L blocks.

    ``table[q, l]`` is one :class:`Cocluster`; iteration yields every co-cluster,
    group by group. Arrays are copied and made read-only.

    When the items use several scales, give ``scale`` as the collection of the scales,
    and ``vote_shares`` as one array per scale, of shape (Q, L, K_c) for a scale of
    ``K_c`` levels. The vote rate and the latent opinion stay one per co-cluster.

    Args:
        scale: The scale of scores, with K levels, or the scales of the items.
        vote_rate: Shape (Q, L): share of the shown cells that carry a vote.
        vote_shares: Shape (Q, L, K): distribution of the votes over the scores; with
            several scales, one array per scale.
        n_votes: Shape (Q, L): number of votes.
        latent_mean: Shape (Q, L): mean of the latent opinion under the ordinal probit
            response model; ``None`` otherwise.
        latent_spread: Shape (Q, L): standard deviation of the latent opinion, likewise.
        thresholds: The thresholds of the ordinal probit response model, one grid of
            ``K_c - 1`` strictly increasing numbers per scale (a single grid for a
            single scale); ``None`` under another response model.
        has_fixed_spread: Shape (Q, L), booleans: the latent spread of the co-cluster
            was held fixed, because all its votes are on scales of two levels; ``None``
            under another response model.

    Attributes:
        scales: The scales, in the order of the arrays.
        vote_rate: Shape (Q, L), as given.
        vote_shares_by_scale: One array of vote shares per scale, of shape (Q, L, K_c).
        n_votes: Shape (Q, L), as given.
        latent_mean: Shape (Q, L), as given, or ``None``.
        latent_spread: Shape (Q, L), as given, or ``None``.
        thresholds: One grid of thresholds per scale, or ``None``.
        has_fixed_spread: Shape (Q, L), as given, or ``None``.

    Raises:
        InvalidInputError: If the shapes do not agree, if a rate is outside [0, 1], if
            a distribution does not sum to 1 within 1e-6, if a count is negative, or if
            a grid of thresholds does not fit its scale.
    """

    scales: tuple[OrdinalScale, ...]
    vote_rate: FloatArray
    vote_shares_by_scale: tuple[FloatArray, ...]
    n_votes: FloatArray
    latent_mean: FloatArray | None
    latent_spread: FloatArray | None
    thresholds: tuple[FloatArray, ...] | None
    has_fixed_spread: BoolArray | None

    def __init__(
        self,
        scale: OrdinalScale | Sequence[OrdinalScale],
        vote_rate: ArrayLike,
        vote_shares: ArrayLike | Sequence[ArrayLike],
        n_votes: ArrayLike,
        latent_mean: ArrayLike | None = None,
        latent_spread: ArrayLike | None = None,
        *,
        thresholds: ArrayLike | Sequence[ArrayLike] | None = None,
        has_fixed_spread: ArrayLike | None = None,
    ) -> None:
        scales, by_scale = _scales(scale)
        rate = _array(vote_rate, "vote_rate")
        if rate.ndim != 2:
            raise InvalidInputError(f"vote_rate has the shape (groups, blocks), got {rate.shape}")
        if by_scale:
            if not isinstance(vote_shares, Sequence) or len(vote_shares) != len(scales):
                raise InvalidInputError(
                    f"vote_shares holds one array per scale, {len(scales)} here"
                )
            received_shares = list(vote_shares)
        else:
            received_shares = [vote_shares]
        shares: list[FloatArray] = []
        for one_scale, received in zip(scales, received_shares, strict=True):
            array = _array(received, "vote_shares")
            expected = (*rate.shape, one_scale.n_levels)
            if array.shape != expected:
                raise InvalidInputError(
                    f"vote_shares has the shape (groups, blocks, levels) = {expected}, "
                    f"got {array.shape}"
                )
            if (array < 0).any() or (np.abs(array.sum(axis=-1) - 1.0) > _SUM_TOLERANCE).any():
                raise InvalidInputError(
                    "the vote shares of a co-cluster are not negative and sum to 1"
                )
            shares.append(array)
        counts = _array(n_votes, "n_votes")
        if counts.shape != rate.shape:
            raise InvalidInputError(
                f"n_votes has the shape {rate.shape} of vote_rate, got {counts.shape}"
            )
        if ((rate < 0) | (rate > 1)).any():
            raise InvalidInputError("a vote rate is between 0 and 1")
        if (counts < 0).any():
            raise InvalidInputError("a number of votes is not negative")
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "vote_rate", rate)
        object.__setattr__(self, "vote_shares_by_scale", tuple(shares))
        object.__setattr__(self, "n_votes", counts)
        for name, received_latent in (
            ("latent_mean", latent_mean),
            ("latent_spread", latent_spread),
        ):
            latent = None if received_latent is None else _array(received_latent, name)
            if latent is not None and latent.shape != rate.shape:
                raise InvalidInputError(
                    f"{name} has the shape {rate.shape} of vote_rate, got {latent.shape}"
                )
            object.__setattr__(self, name, latent)
        object.__setattr__(self, "thresholds", _grids(thresholds, scales, by_scale))
        fixed = None
        if has_fixed_spread is not None:
            fixed = np.array(has_fixed_spread, dtype=np.bool_)
            if fixed.shape != rate.shape:
                raise InvalidInputError(
                    f"has_fixed_spread has the shape {rate.shape} of vote_rate, got {fixed.shape}"
                )
            fixed.setflags(write=False)
        object.__setattr__(self, "has_fixed_spread", fixed)

    @property
    def scale(self) -> OrdinalScale:
        """The scale of scores, when the table has one.

        Raises:
            InvalidInputError: If the table has several scales: read :attr:`scales`.
        """
        return _only_scale(self.scales, "this table of co-clusters")

    @property
    def vote_shares(self) -> FloatArray:
        """Shape (Q, L, K): the distribution of the votes over the scores, for one scale.

        Raises:
            InvalidInputError: If the table has several scales: read
                :attr:`vote_shares_by_scale`.
        """
        _only_scale(self.scales, "this table of co-clusters")
        return self.vote_shares_by_scale[0]

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
        shares = tuple(
            tuple(float(share) for share in array[group, block])
            for array in self.vote_shares_by_scale
        )
        scores = tuple(scale.scores for scale in self.scales)
        one = len(self.scales) == 1
        return Cocluster(
            group=int(group),
            block=int(block),
            vote_rate=float(self.vote_rate[group, block]),
            vote_shares=shares[0] if one else shares,
            n_votes=float(self.n_votes[group, block]),
            latent_mean=None if self.latent_mean is None else float(self.latent_mean[group, block]),
            latent_spread=(
                None if self.latent_spread is None else float(self.latent_spread[group, block])
            ),
            scores=scores[0] if one else scores,
            has_fixed_spread=(
                False
                if self.has_fixed_spread is None
                else bool(self.has_fixed_spread[group, block])
            ),
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

        Shape (Q, L, K + 1), for a table of one scale. State 0 (shown without a vote):
        ``log(1 - vote_rate)``; state ``k``: ``log(vote_rate) + log(vote_shares[k])``.
        Probabilities are floored at 1e-300 before the logarithm.

        Raises:
            InvalidInputError: If the table has several scales: read
                :meth:`log_probabilities_by_scale`.
        """
        return state_log_probabilities(self.vote_rate, self.vote_shares)

    def log_probabilities_by_scale(self) -> FloatArray:
        """Return the log-probability of each state of a shown cell, per co-cluster and scale.

        Shape (Q, L, C, K + 1), with C the scales and K the levels of the longest one.
        Entry ``[q, l, c, s]`` is the logarithm of the probability that a shown cell of
        an item of the scale ``c`` in the co-cluster ``(q, l)`` is in the state ``s``.
        The states past the levels of a scale have the probability 0, floored at
        1e-300 before the logarithm. With one scale, ``[:, :, 0]`` is
        :meth:`log_probabilities`.
        """
        return state_log_probabilities(self.vote_rate, self._padded_shares)

    @cached_property
    def _padded_shares(self) -> FloatArray:
        """Shape (Q, L, C, K): the shares of each scale, followed by zeros up to K."""
        n_levels = max(scale.n_levels for scale in self.scales)
        padded: FloatArray = np.zeros((*self.vote_rate.shape, len(self.scales), n_levels))
        for position, array in enumerate(self.vote_shares_by_scale):
            padded[:, :, position, : array.shape[-1]] = array
        padded.setflags(write=False)
        return padded

    def to_dict(self) -> dict[str, Any]:
        """Return plain lists: the scales, the tables, and what the response model adds.

        With one scale: ``scores``, ``vote_rate``, ``vote_shares`` and ``n_votes``. With
        several, ``scores`` gives way to ``scales`` (the scores of each scale) and
        ``vote_shares`` to ``vote_shares_by_scale`` (one table per scale), so that a
        reader of one scale never reads the shares of several as one. Then
        ``latent_mean`` and ``latent_spread``, ``thresholds`` (one grid per scale) and
        ``has_fixed_spread``, when the table has them.
        """
        data: dict[str, Any]
        if len(self.scales) == 1:
            data = {
                "scores": list(self.scales[0].scores),
                "vote_rate": self.vote_rate.tolist(),
                "vote_shares": self.vote_shares_by_scale[0].tolist(),
                "n_votes": self.n_votes.tolist(),
            }
        else:
            data = {
                "scales": [list(scale.scores) for scale in self.scales],
                "vote_rate": self.vote_rate.tolist(),
                "vote_shares_by_scale": [array.tolist() for array in self.vote_shares_by_scale],
                "n_votes": self.n_votes.tolist(),
            }
        if self.latent_mean is not None:
            data["latent_mean"] = self.latent_mean.tolist()
        if self.latent_spread is not None:
            data["latent_spread"] = self.latent_spread.tolist()
        if self.thresholds is not None:
            data["thresholds"] = [grid.tolist() for grid in self.thresholds]
        if self.has_fixed_spread is not None:
            data["has_fixed_spread"] = self.has_fixed_spread.tolist()
        return data

    def _describe(self) -> list[str]:
        n_groups, n_blocks = self.shape
        if len(self.scales) == 1:
            reading = f"{self.scales[0].n_levels} levels"
        else:
            levels = sorted({scale.n_levels for scale in self.scales})
            span = f"{levels[0]}" if len(levels) == 1 else f"{levels[0]} to {levels[-1]}"
            reading = f"{len(self.scales)} scales of {span} levels"
        lines = [
            f"CoclusterTable({n_groups} group{'' if n_groups == 1 else 's'} x "
            f"{n_blocks} block{'' if n_blocks == 1 else 's'}, {reading})"
        ]
        for cocluster in self:
            typical = ", ".join(
                scale.format_score(score)
                for scale, score in zip(self.scales, cocluster.typical_score_by_scale, strict=True)
            )
            label = "typical score" if len(self.scales) == 1 else "typical scores"
            lines.append(
                f"group {cocluster.group} x block {cocluster.block}: "
                f"vote rate {cocluster.vote_rate:.2f}, {label} {typical}, "
                f"{cocluster.n_votes:g} votes"
            )
        return lines


def _grids(
    received: ArrayLike | Sequence[ArrayLike] | None,
    scales: tuple[OrdinalScale, ...],
    by_scale: bool,
) -> tuple[FloatArray, ...] | None:
    """Return one grid of thresholds per scale, checked, or ``None``."""
    if received is None:
        return None
    if by_scale:
        if not isinstance(received, Sequence) or len(received) != len(scales):
            raise InvalidInputError(f"thresholds hold one grid per scale, {len(scales)} here")
        grids = list(received)
    else:
        grids = [received]
    checked: list[FloatArray] = []
    for scale, grid in zip(scales, grids, strict=True):
        array = _array(grid, "thresholds")
        if array.shape != (scale.n_levels - 1,):
            raise InvalidInputError(
                f"a scale of {scale.n_levels} scores has {scale.n_levels - 1} thresholds, "
                f"got the shape {array.shape}"
            )
        if any(low >= high for low, high in pairwise(array.tolist())):
            raise InvalidInputError(f"thresholds are strictly increasing, got {array.tolist()}")
        checked.append(array)
    return tuple(checked)
