# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Votes of voters on items, as a matrix with three states per cell.

:class:`VoteMatrix` is the input of all co-clustering. Its central rule: a cell is
voted, shown without a vote, or not shown. An empty cell is never a zero nor a neutral
answer, and only shown cells enter a computation.

Each item is asked on a scale (:class:`OrdinalScale`). Items may use different scales,
as the questions of a survey do: :class:`ItemScales` says which scale each item uses.
"""

import hashlib
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from civiqhub_selene.blocks._arrays import BoolArray, FloatArray, IntArray
from civiqhub_selene.blocks._numerics import count_states, count_states_by_scale
from civiqhub_selene.blocks.partition import ItemBlock, Partition, VoterGroup
from civiqhub_selene.errors import InvalidInputError, InvalidParameterError
from civiqhub_selene.ordinal.counts import SCORES, VoteCounts
from civiqhub_selene.types import Displayable

if TYPE_CHECKING:
    from civiqhub_selene.blocks.coclusters import CoclusterTable

__all__ = ["GroupedVotes", "ItemScales", "OrdinalScale", "Vote", "VoteMatrix"]

Selector: TypeAlias = slice | str | Iterable[str]
"""One side of a selection: ``:``, one identifier, or a collection of identifiers."""


def _is_integer(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


@dataclass(frozen=True)
class OrdinalScale:
    """An ordered scale of scores. Defaults to the five-score agreement scale.

    A score is the answer of a vote, for example from -2 to +2. A level is the rank of
    a score on the scale, from 1 to ``n_levels`` (K); level 0 means "no vote" and is
    internal to matrices.

    See the mathematical model: :doc:`/model/data`.

    Attributes:
        scores: The scores, strictly increasing integers, at least two.

    Raises:
        InvalidParameterError: If the scores are not strictly increasing integers, or
            if there are fewer than two.
    """

    scores: tuple[int, ...] = (-2, -1, 0, 1, 2)

    def __post_init__(self) -> None:
        received = self.scores
        if isinstance(received, (str, bytes)) or not isinstance(received, Iterable):
            raise InvalidParameterError(f"the scores of a scale are integers, got {received!r}")
        scores = tuple(received)
        if len(scores) < 2 or not all(_is_integer(score) for score in scores):
            raise InvalidParameterError(
                f"a scale has at least two integer scores, got {received!r}"
            )
        if any(low >= high for low, high in pairwise(scores)):
            raise InvalidParameterError(
                f"the scores of a scale are strictly increasing, got {received!r}"
            )
        object.__setattr__(self, "scores", tuple(int(score) for score in scores))

    @property
    def n_levels(self) -> int:
        """The number of scores on the scale."""
        return len(self.scores)

    def level_of(self, score: int) -> int:
        """Return the rank of a score, from 1 to ``n_levels``.

        Raises:
            InvalidInputError: If the score is not on the scale.
        """
        if not _is_integer(score) or score not in self.scores:
            raise InvalidInputError(f"the score {score!r} is not on the scale {self._formatted()}")
        return self.scores.index(int(score)) + 1

    def score_of(self, level: int) -> int:
        """Return the score at a rank from 1 to ``n_levels``.

        Raises:
            InvalidInputError: If the level is not a rank of the scale.
        """
        if not _is_integer(level) or not 1 <= level <= self.n_levels:
            raise InvalidInputError(
                f"a level is a rank from 1 to {self.n_levels} on the scale, got {level!r}"
            )
        return self.scores[int(level) - 1]

    def format_score(self, score: int) -> str:
        """Return the score with its sign: ``"+2"``, ``"0"``, ``"-1"``."""
        return "0" if score == 0 else f"{score:+d}"

    def _formatted(self) -> str:
        return ", ".join(self.format_score(score) for score in self.scores)


def _only_scale(scales: tuple[OrdinalScale, ...], owner: str) -> OrdinalScale:
    """Return the one scale of ``owner``, or say how to read several.

    Every one-scale reading (``scale``, ``vote_shares``, ``typical_score``...) goes
    through here, so that they all refuse in the same words.

    Raises:
        InvalidInputError: If there is more than one scale.
    """
    if len(scales) == 1:
        return scales[0]
    raise InvalidInputError(
        f"{owner} has {len(scales)} scales, not one",
        hint=(
            "Read the form by scale instead (item_scales, scales, or the name that ends "
            "with _by_scale), or select the items of one scale first."
        ),
    )


@dataclass(frozen=True, eq=False, repr=False)
class ItemScales(Displayable):
    """The scale of each item of a vote matrix.

    The questions of a survey use different scales: two answers, four, seven. This
    object says which scale each item uses. Its ``scales`` are the distinct scales, in
    the order in which the items first use them, and ``scale_index[j]`` is the position
    in ``scales`` of the scale of the item ``j``. Two scales with the same scores are
    the same scale: response models give each scale its own grid of thresholds, and
    the items of one scale share it.

    Build it with :meth:`from_scales`, from the scale of each item in the order of the
    columns, or with :meth:`uniform` when every item uses one scale. It reads by
    position, like the columns of a matrix: ``len(item_scales)`` is the number of
    items, ``item_scales[j]`` the scale of the item ``j``, and iteration yields the
    scale of each item in turn. Two values with the same content are equal.

    Attributes:
        scales: The distinct scales, at least one, in the order of their first item.
        scale_index: Integers of shape (P,): for each item, the position of its scale
            in ``scales``.

    Raises:
        InvalidInputError: If a scale is not an :class:`OrdinalScale`, if two scales
            have the same scores, if an index does not name a scale, if a scale has no
            item, or if the scales are not in the order of their first item. Without
            any item, there is exactly one scale.
    """

    scales: tuple[OrdinalScale, ...]
    scale_index: IntArray

    def __post_init__(self) -> None:
        received = self.scales
        if isinstance(received, OrdinalScale) or not isinstance(received, Iterable):
            raise InvalidInputError(
                f"scales are a collection of OrdinalScale, got {received!r}",
                hint="Build the value with ItemScales.from_scales or ItemScales.uniform.",
            )
        scales = tuple(received)
        if not scales:
            raise InvalidInputError("an ItemScales has at least one scale, got none")
        for scale in scales:
            if not isinstance(scale, OrdinalScale):
                raise InvalidInputError(f"a scale is an OrdinalScale, got {scale!r}")
        if len({scale.scores for scale in scales}) != len(scales):
            raise InvalidInputError(
                "two scales of an ItemScales have the same scores: they are one scale",
                hint="Build the value with ItemScales.from_scales, which merges them.",
            )
        raw = np.asarray(self.scale_index)
        if raw.ndim != 1 or (raw.size and raw.dtype.kind not in "iu"):
            raise InvalidInputError(
                f"scale_index holds one integer per item, got an array of {raw.dtype} "
                f"and shape {raw.shape}"
            )
        index: IntArray = np.array(raw, dtype=np.int64)
        if ((index < 0) | (index >= len(scales))).any():
            raise InvalidInputError(
                f"scale_index holds positions from 0 to {len(scales) - 1} in scales"
            )
        if index.size == 0 and len(scales) != 1:
            raise InvalidInputError(
                f"without any item, an ItemScales keeps exactly one scale, got {len(scales)}"
            )
        _, first_items = np.unique(index, return_index=True)
        if index.size and len(first_items) != len(scales):
            raise InvalidInputError(
                f"each scale of an ItemScales has at least one item, but only "
                f"{len(first_items)} of the {len(scales)} scales have one"
            )
        if index.size and (index[np.sort(first_items)] != np.arange(len(scales))).any():
            raise InvalidInputError(
                "the scales of an ItemScales are in the order of their first item",
                hint="Build the value with ItemScales.from_scales, which orders them.",
            )
        _frozen(index)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "scale_index", index)

    @classmethod
    def uniform(cls, scale: OrdinalScale, n_items: int) -> "ItemScales":
        """Return the scales of ``n_items`` items that all use ``scale``.

        Raises:
            InvalidInputError: If ``scale`` is not an OrdinalScale or ``n_items`` is
                not a non-negative integer.
        """
        if not isinstance(scale, OrdinalScale):
            raise InvalidInputError(f"the scale is an OrdinalScale, got {scale!r}")
        if not _is_integer(n_items) or n_items < 0:
            raise InvalidInputError(f"n_items is a non-negative integer, got {n_items!r}")
        return cls((scale,), np.zeros(int(n_items), dtype=np.int64))

    @classmethod
    def from_scales(cls, scales: Iterable[OrdinalScale]) -> "ItemScales":
        """Return the scales of items from the scale of each item, in column order.

        Equal scales become one scale; the scales keep the order of their first item.
        Without any item, the scale is the five-score agreement scale.

        Raises:
            InvalidInputError: If an element is not an OrdinalScale.
        """
        if isinstance(scales, OrdinalScale):
            raise InvalidInputError(
                "from_scales reads the scale of each item, got a single scale",
                hint="Use ItemScales.uniform(scale, n_items) when every item has one scale.",
            )
        positions: dict[tuple[int, ...], int] = {}
        distinct: list[OrdinalScale] = []
        index: list[int] = []
        for scale in scales:
            if not isinstance(scale, OrdinalScale):
                raise InvalidInputError(f"the scale of an item is an OrdinalScale, got {scale!r}")
            if scale.scores not in positions:
                positions[scale.scores] = len(distinct)
                distinct.append(scale)
            index.append(positions[scale.scores])
        if not distinct:
            return cls.uniform(OrdinalScale(), 0)
        return cls(tuple(distinct), np.array(index, dtype=np.int64))

    # Reading --------------------------------------------------------------------------

    @property
    def n_items(self) -> int:
        """The number of items."""
        return int(self.scale_index.shape[0])

    @property
    def n_scales(self) -> int:
        """The number of distinct scales."""
        return len(self.scales)

    @property
    def is_uniform(self) -> bool:
        """Whether every item uses the same scale."""
        return len(self.scales) == 1

    @property
    def scale(self) -> OrdinalScale:
        """The scale of every item, when they all use one.

        Raises:
            InvalidInputError: If the items use several scales.
        """
        return _only_scale(self.scales, "these items")

    @property
    def n_levels(self) -> int:
        """The number of levels of the longest scale."""
        return max(scale.n_levels for scale in self.scales)

    @cached_property
    def levels_by_item(self) -> IntArray:
        """Integers of shape (P,): the number of levels of the scale of each item."""
        by_scale = np.array([scale.n_levels for scale in self.scales], dtype=np.int64)
        levels: IntArray = by_scale[self.scale_index]
        _frozen(levels)
        return levels

    def __len__(self) -> int:
        return self.n_items

    def __iter__(self) -> Iterator[OrdinalScale]:
        for position in self.scale_index:
            yield self.scales[int(position)]

    def __getitem__(self, position: int) -> OrdinalScale:
        """Return the scale of the item at ``position``, from 0.

        Raises:
            TypeError: If ``position`` is not an integer.
            IndexError: If there is no item at ``position``.
        """
        if not _is_integer(position):
            raise TypeError(f"the scale of an item is read by its position, got {position!r}")
        if not -self.n_items <= position < self.n_items:
            raise IndexError(f"there are {self.n_items} items, got the position {position!r}")
        return self.scales[int(self.scale_index[position])]

    def to_dict(self) -> dict[str, Any]:
        """Return the scores of each scale and the scale of each item, as plain lists."""
        return {
            "scales": [list(scale.scores) for scale in self.scales],
            "scale_index": self.scale_index.tolist(),
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ItemScales):
            return NotImplemented
        return self._key == other._key

    def __hash__(self) -> int:
        return hash(self._key)

    # Internals ------------------------------------------------------------------------

    @cached_property
    def _key(self) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
        return (tuple(scale.scores for scale in self.scales), tuple(self.scale_index.tolist()))

    @cached_property
    def _possible_states(self) -> BoolArray:
        """Booleans of shape (C, K + 1), K the levels of the longest scale.

        Entry ``[c, s]`` says whether a shown cell of an item of the scale ``c`` can be in
        the state ``s``: state 0 (no vote) always, a vote at rank ``s`` up to the levels
        of the scale.
        """
        levels = np.array([scale.n_levels for scale in self.scales], dtype=np.int64)
        possible: BoolArray = np.arange(self.n_levels + 1)[np.newaxis, :] <= levels[:, np.newaxis]
        _frozen(possible)
        return possible

    def _select(self, columns: Sequence[int]) -> "ItemScales":
        """Return the scales of the items at ``columns``, in that order."""
        if self.is_uniform:
            return ItemScales.uniform(self.scales[0], len(columns))
        if len(columns) == 0:
            return ItemScales.uniform(self.scales[int(self.scale_index[0])], 0)
        return ItemScales.from_scales(self[column] for column in columns)

    def _describe(self) -> list[str]:
        n_items, n_scales = self.n_items, self.n_scales
        lines = [
            f"ItemScales({n_scales} scale{_plural(n_scales)} of {_span(self.scales)} levels over "
            f"{n_items} item{_plural(n_items)})"
        ]
        counts = np.bincount(self.scale_index, minlength=n_scales)
        for position, scale in enumerate(self.scales):
            count = int(counts[position])
            lines.append(f"scale {position}: {scale._formatted()} ({count} item{_plural(count)})")
        return lines


class Vote(NamedTuple):
    """One vote: a voter gave a score to an item."""

    voter: str
    item: str
    score: int


def _identifiers(received: Iterable[str], kind: str) -> tuple[str, ...]:
    """Check that identifiers are unique texts and return them as a tuple."""
    if isinstance(received, str):
        raise InvalidInputError(f"{kind} identifiers are a collection of texts, got {received!r}")
    identifiers = tuple(received)
    seen: set[str] = set()
    for identifier in identifiers:
        if not isinstance(identifier, str):
            raise InvalidInputError(f"a {kind} identifier is text, got {identifier!r}")
        if identifier in seen:
            raise InvalidInputError(f"the {kind} identifier {identifier!r} is repeated")
        seen.add(identifier)
    return identifiers


def _numbered(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}{number}" for number in range(1, count + 1))


ScalesOfItems: TypeAlias = Mapping[str, OrdinalScale] | ItemScales
"""The scales of the items: by item identifier, or as an :class:`ItemScales` in column order."""


def _resolved_item_scales(
    columns: tuple[str, ...],
    scale: OrdinalScale | None,
    item_scales: ScalesOfItems | None,
) -> ItemScales:
    """Return the scale of each column from the two ways of giving it.

    ``scale`` gives every item one scale; ``item_scales`` gives each item its own, by
    identifier or in column order. Given together, they must agree.

    Raises:
        InvalidInputError: If a value has the wrong type, if ``item_scales`` does not
            cover exactly the columns, or if ``scale`` disagrees with ``item_scales``.
    """
    if scale is not None and not isinstance(scale, OrdinalScale):
        raise InvalidInputError(f"the scale is an OrdinalScale, got {scale!r}")
    if item_scales is None:
        return ItemScales.uniform(OrdinalScale() if scale is None else scale, len(columns))
    if isinstance(item_scales, ItemScales):
        if item_scales.n_items != len(columns):
            raise InvalidInputError(
                f"item_scales gives the scales of {item_scales.n_items} items "
                f"to {len(columns)} columns"
            )
        resolved = item_scales
    elif isinstance(item_scales, Mapping):
        if set(item_scales) != set(columns):
            missing = [item for item in columns if item not in item_scales]
            extra = [item for item in item_scales if item not in set(columns)]
            named = f"; no scale for {missing[0]!r}" if missing else f"; {extra[0]!r} is extra"
            raise InvalidInputError(
                "item_scales must give a scale to exactly the items of the matrix" + named
            )
        for item in columns:
            if not isinstance(item_scales[item], OrdinalScale):
                raise InvalidInputError(
                    f"the scale of item {item!r} is an OrdinalScale, got {item_scales[item]!r}"
                )
        resolved = ItemScales.from_scales(item_scales[item] for item in columns)
        if not columns and scale is not None:
            resolved = ItemScales.uniform(scale, 0)
    else:
        raise InvalidInputError(
            "item_scales maps each item identifier to its OrdinalScale, or is an ItemScales, "
            f"got {item_scales!r}"
        )
    if scale is not None:
        if not resolved.is_uniform:
            raise InvalidInputError(
                f"scale gives every item one scale, but item_scales gives {resolved.n_scales}",
                hint="Omit scale when the items use several scales.",
            )
        if resolved.scales[0].scores != scale.scores:
            raise InvalidInputError(
                f"the given scale {scale._formatted()} differs from the item scale "
                f"{resolved.scales[0]._formatted()}",
                hint="Omit scale, or pass the same OrdinalScale as every item.",
            )
    return resolved


def _frozen(array: NDArray[Any]) -> None:
    array.setflags(write=False)


def _check_levels(levels: IntArray, item_scales: ItemScales, items: tuple[str, ...]) -> None:
    """Refuse a level outside ``0..K_j``, ``K_j`` the levels of the scale of its column."""
    if item_scales.is_uniform:
        n_levels = item_scales.n_levels
        outside = levels[(levels < 0) | (levels > n_levels)]
        if outside.size:
            raise InvalidInputError(
                f"a level is a rank from 0 to {n_levels}, got {int(outside[0])}"
            )
        return
    limits = item_scales.levels_by_item
    rows, columns = np.nonzero((levels < 0) | (levels > limits[np.newaxis, :]))
    if rows.size:
        column = int(columns[0])
        name = repr(items[column]) if column < len(items) else f"at column {column}"
        raise InvalidInputError(
            f"item {name} is on a scale of {int(limits[column])} levels: a level is a rank "
            f"from 0 to {int(limits[column])}, got {int(levels[rows[0], column])}"
        )


@dataclass(frozen=True, eq=False, repr=False, init=False)
class VoteMatrix(Displayable):
    """Votes of M voters on P items, by opaque identifiers.

    A cell has three states. It is *not shown* when ``shown`` is false, whatever
    ``levels`` holds; an *abstention* when it is shown and ``levels`` is 0; *voted*
    when it is shown and ``levels`` is positive. A vote at the neutral score is a vote.
    The models read the state ``s[i, j]`` of each shown cell, 0 for an abstention and
    ``k`` for a vote at rank ``k``, through its indicator::

        S[i, j, s] = 1 if the cell (i, j) is shown and s[i, j] = s, else 0

    so that a cell that is not shown is zero in every state.

    See the mathematical model: :doc:`/model/data`, equation :eq:`eq-data-states`.

    Each item has a scale. By default every item uses the five-score agreement scale;
    ``scale`` gives every item another one, and ``item_scales`` gives each item its
    own. ``levels[i, j]`` is a rank on the scale of the item ``j``: a level means the
    same answer only between items of the same scale.

    The arrays are copied and made read-only. Two matrices with the same content are
    equal: equality and hash follow :meth:`fingerprint`.

    Args:
        levels: Integers of shape (M, P). 0: no vote; ``k`` from 1 to the number of
            levels of the scale of the column: vote at rank ``k`` of that scale.
        shown: Booleans of shape (M, P): the item was shown to the voter.
        voters: Opaque identifiers of the rows, unique.
        items: Opaque identifiers of the columns, unique.
        scale: The scale of every item. Defaults to the five-score agreement scale.
        voter_weights: Optional finite, positive survey weights, in row order. The
            models read them scaled to average 1, as ``w[i]``.
        item_scales: The scale of each item, as an :class:`ItemScales` in column order,
            in place of ``scale``. Given with ``scale``, they must agree.

    Attributes:
        levels: As given, read-only.
        shown: As given, read-only.
        voters: As given, as a tuple.
        items: As given, as a tuple.
        item_scales: The scale of each item, always set.
        voter_weights: As given, read-only, or ``None``.

    Raises:
        InvalidInputError: If the arrays are not two matrices of the same shape, if a
            level is outside the levels of the scale of its column, if the identifiers
            or the scales do not follow the shape, or if an identifier is repeated.
    """

    levels: IntArray
    shown: BoolArray
    voters: tuple[str, ...]
    items: tuple[str, ...]
    item_scales: ItemScales
    voter_weights: FloatArray | None

    def __init__(
        self,
        levels: IntArray,
        shown: BoolArray,
        voters: Iterable[str],
        items: Iterable[str],
        scale: OrdinalScale | None = None,
        voter_weights: FloatArray | None = None,
        *,
        item_scales: ItemScales | None = None,
    ) -> None:
        if item_scales is not None and not isinstance(item_scales, ItemScales):
            raise InvalidInputError(f"item_scales is an ItemScales, got {item_scales!r}")
        if scale is not None and not isinstance(scale, OrdinalScale):
            raise InvalidInputError(f"the scale is an OrdinalScale, got {scale!r}")
        n_levels = (
            item_scales.n_levels
            if item_scales is not None
            else (OrdinalScale() if scale is None else scale).n_levels
        )
        raw_levels = np.asarray(levels)
        if raw_levels.size and raw_levels.dtype.kind not in "iu":
            raise InvalidInputError(
                f"levels are integers from 0 to {n_levels}, got an array of {raw_levels.dtype}"
            )
        raw_shown = np.asarray(shown)
        if raw_shown.size and raw_shown.dtype.kind != "b":
            raise InvalidInputError(f"shown is a boolean array, got an array of {raw_shown.dtype}")
        if raw_levels.ndim != 2 or raw_shown.ndim != 2 or raw_levels.shape != raw_shown.shape:
            raise InvalidInputError(
                "levels and shown are two matrices of the same shape (voters, items), "
                f"got shapes {raw_levels.shape} and {raw_shown.shape}"
            )
        level_array: IntArray = np.array(raw_levels, dtype=np.int64)
        shown_array: BoolArray = np.array(raw_shown, dtype=np.bool_)
        if item_scales is None:
            item_scales = ItemScales.uniform(
                OrdinalScale() if scale is None else scale, level_array.shape[1]
            )
        if item_scales.n_items != level_array.shape[1]:
            raise InvalidInputError(
                f"item_scales gives the scales of {item_scales.n_items} items "
                f"to a matrix of {level_array.shape[1]} columns"
            )
        if scale is not None and (
            not item_scales.is_uniform or item_scales.scales[0].scores != scale.scores
        ):
            raise InvalidInputError(
                "scale and item_scales disagree: scale gives every item one scale",
                hint="Pass one of the two.",
            )
        voter_ids = _identifiers(voters, "voter")
        item_ids = _identifiers(items, "item")
        _check_levels(level_array, item_scales, item_ids)
        if (len(voter_ids), len(item_ids)) != level_array.shape:
            raise InvalidInputError(
                f"{len(voter_ids)} voters and {len(item_ids)} items do not follow "
                f"the shape {level_array.shape} of the matrix"
            )
        _frozen(level_array)
        _frozen(shown_array)
        object.__setattr__(self, "levels", level_array)
        object.__setattr__(self, "shown", shown_array)
        object.__setattr__(self, "voters", voter_ids)
        object.__setattr__(self, "items", item_ids)
        object.__setattr__(self, "item_scales", item_scales)
        object.__setattr__(self, "voter_weights", None)
        if voter_weights is not None:
            received = np.asarray(voter_weights)
            if received.shape != (len(voter_ids),) or received.dtype.kind not in "fiu":
                raise InvalidInputError("voter_weights must contain one numeric weight per voter")
            weights: FloatArray = np.array(received, dtype=np.float64)
            if not np.isfinite(weights).all() or (weights <= 0).any():
                raise InvalidInputError("voter_weights must be finite and strictly positive")
            _frozen(weights)
            object.__setattr__(self, "voter_weights", weights)

    # Scales ---------------------------------------------------------------------------

    @property
    def scale(self) -> OrdinalScale:
        """The scale of every item, when they all use one.

        Raises:
            InvalidInputError: If the items use several scales: read
                :attr:`item_scales` or :meth:`scale_of`.
        """
        return _only_scale(self.item_scales.scales, "this vote matrix")

    @property
    def scales(self) -> tuple[OrdinalScale, ...]:
        """The distinct scales of the items, in the order of their first item."""
        return self.item_scales.scales

    def scale_of(self, item: str) -> OrdinalScale:
        """Return the scale of an item, by its identifier.

        Raises:
            InvalidInputError: If the item is not in the matrix.
        """
        if not isinstance(item, str) or item not in self._columns:
            raise InvalidInputError(f"the item {item!r} is not in the matrix")
        return self.item_scales[self._columns[item]]

    @cached_property
    def _columns(self) -> dict[str, int]:
        return {item: position for position, item in enumerate(self.items)}

    # Constructors ---------------------------------------------------------------------

    @classmethod
    def from_records(
        cls,
        votes: Iterable[tuple[str, str, int]],
        *,
        shown: Iterable[tuple[str, str]] | None = None,
        voters: Iterable[str] | None = None,
        items: Iterable[str] | None = None,
        scale: OrdinalScale | None = None,
        item_scales: ScalesOfItems | None = None,
    ) -> "VoteMatrix":
        """Build a matrix from ``(voter, item, score)`` triples.

        Rows and columns follow the order of first appearance, in the votes and then
        in ``shown``, unless ``voters`` or ``items`` fix it. A mapping ``item_scales``
        fixes the columns too, in the order of its keys, when ``items`` is left out.

        Without ``shown``, every cell counts as shown: an empty cell is then read as
        an abstention. With ``shown``, only these ``(voter, item)`` pairs and the
        voted cells are shown; the other cells are not shown.

        Args:
            votes: The votes, as ``(voter, item, score)`` triples. Each score is on the
                scale of its item.
            shown: The ``(voter, item)`` pairs where the item was shown to the voter.
            voters: The voters in the order of the rows. It may name voters without
                any vote; it must name every voter of the records.
            items: The items in the order of the columns, likewise.
            scale: The scale of every item. Defaults to the five-score agreement scale.
            item_scales: The scale of each item, in place of ``scale``: a mapping from
                each item identifier to its :class:`OrdinalScale`, or an
                :class:`ItemScales` that follows ``items``.

        Raises:
            InvalidInputError: If a score is not on the scale of its item, if a voter
                votes twice on the same item, if a record names an identifier that
                ``voters`` or ``items`` leaves out, or if the scales do not cover
                exactly the items.
        """
        if isinstance(item_scales, Mapping) and items is None:
            items = tuple(item_scales)
        if isinstance(item_scales, ItemScales) and items is None:
            raise InvalidInputError(
                "an ItemScales follows the order of the columns: give items with it",
                hint="Pass items=..., or give item_scales as a mapping by item identifier.",
            )
        rows: dict[str, int] = {}
        columns: dict[str, int] = {}
        if voters is not None:
            rows = {voter: position for position, voter in enumerate(_identifiers(voters, "voter"))}
        if items is not None:
            columns = {item: position for position, item in enumerate(_identifiers(items, "item"))}
        fixed_items = items is not None
        resolved = (
            None
            if item_scales is None
            else _resolved_item_scales(tuple(columns), scale, item_scales)
        )
        default = OrdinalScale() if scale is None else scale

        def position(index: dict[str, int], identifier: object, kind: str, fixed: bool) -> int:
            if not isinstance(identifier, str):
                raise InvalidInputError(f"a {kind} identifier is text, got {identifier!r}")
            if identifier not in index:
                if fixed:
                    raise InvalidInputError(
                        f"the {kind} {identifier!r} is in the records but not in the {kind}s given"
                    )
                index[identifier] = len(index)
            return index[identifier]

        cells: dict[tuple[int, int], int] = {}
        for record in votes:
            try:
                voter, item, score = record
            except (TypeError, ValueError):
                raise InvalidInputError(
                    f"a vote is a (voter, item, score) triple, got {record!r}"
                ) from None
            cell = (
                position(rows, voter, "voter", voters is not None),
                position(columns, item, "item", fixed_items),
            )
            if cell in cells:
                raise InvalidInputError(f"the voter {voter!r} votes twice on the item {item!r}")
            if resolved is None or resolved.is_uniform:
                cells[cell] = (default if resolved is None else resolved.scales[0]).level_of(score)
                continue
            try:
                cells[cell] = resolved[cell[1]].level_of(score)
            except InvalidInputError as error:
                raise InvalidInputError(f"item {item!r}: {error}", hint=error.hint) from error
        shown_cells: list[tuple[int, int]] = []
        if shown is not None:
            for pair in shown:
                try:
                    voter, item = pair
                except (TypeError, ValueError):
                    raise InvalidInputError(
                        f"a shown cell is a (voter, item) pair, got {pair!r}"
                    ) from None
                shown_cells.append(
                    (
                        position(rows, voter, "voter", voters is not None),
                        position(columns, item, "item", fixed_items),
                    )
                )
        shape = (len(rows), len(columns))
        levels = np.zeros(shape, dtype=np.int64)
        shown_array = np.full(shape, shown is None, dtype=np.bool_)
        for (row, column), level in cells.items():
            levels[row, column] = level
            shown_array[row, column] = True
        for row, column in shown_cells:
            shown_array[row, column] = True
        if resolved is None:
            return cls(levels, shown_array, tuple(rows), tuple(columns), default)
        return cls(levels, shown_array, tuple(rows), tuple(columns), item_scales=resolved)

    @classmethod
    def from_table(
        cls,
        table: Iterable[Iterable[float | None]],
        *,
        voters: Iterable[str] | None = None,
        items: Iterable[str] | None = None,
        scale: OrdinalScale | None = None,
    ) -> "VoteMatrix":
        """Build a matrix from a voters by items table of scores.

        ``None`` or NaN marks an empty cell. Every cell is shown: an empty cell is an
        abstention. Without identifiers, voters are ``v1, v2, ...`` and items
        ``i1, i2, ...``.

        Raises:
            InvalidInputError: If the rows have different lengths or if a score is not
                on the scale.
        """
        return cls.from_responses(table, voters=voters, items=items, scale=scale)

    @classmethod
    def from_responses(
        cls,
        table: Iterable[Iterable[float | None]],
        *,
        voters: Iterable[str] | None = None,
        items: Iterable[str] | None = None,
        item_scales: ScalesOfItems | None = None,
        scale: OrdinalScale | None = None,
        shown: ArrayLike | None = None,
        voter_weights: ArrayLike | None = None,
    ) -> "VoteMatrix":
        """Build a survey matrix from a table of scores in voter and item order.

        With ``item_scales``, provide the scale of every item: a mapping keyed by item
        identifier, whose keys supply the column order if ``items`` is omitted, or an
        :class:`ItemScales` in column order. Items may use different scales: a question
        with two answers and a question with seven sit side by side. The matrix keeps
        the original ranks on the scale of each item, in the direction of agreement,
        and does not recode them onto a common grid.

        A shown cell of item ``j`` has ``K_j + 1`` modalities (shown without a vote, or
        a vote at one of the ``K_j`` ranks of its scale). A table on one scale matches
        :meth:`from_records` on the same scores, voters and items.

        ``None`` and NaN indicate a cell without a response. The ``shown`` mask
        distinguishes an abstention from a question that was not shown. A score in a
        cell marked as not shown is refused. Pass ``voter_weights`` in row order.
        Each weight must be finite and positive.

        Raises:
            InvalidInputError: If the dimensions or identifiers do not match, if a
                score is outside the scale of its item, if a scored cell is marked as
                not shown, or if the scales do not cover exactly the items.
        """
        rows = [list(row) for row in table]
        widths = {len(row) for row in rows}
        if len(widths) > 1:
            raise InvalidInputError(
                f"the rows of a table have the same length, got lengths {sorted(widths)}"
            )
        if isinstance(item_scales, Mapping) and items is None:
            items = tuple(item_scales)
        identifiers = None if items is None else _identifiers(items, "item")
        width = (
            widths.pop()
            if widths
            else len(identifiers or ())
            or (item_scales.n_items if isinstance(item_scales, ItemScales) else 0)
        )
        columns = _numbered("i", width) if identifiers is None else identifiers
        if len(columns) != width:
            raise InvalidInputError("the item identifiers must match the response table columns")
        resolved = _resolved_item_scales(columns, scale, item_scales)

        n_rows = len(rows)
        row_ids = _numbered("v", n_rows) if voters is None else _identifiers(voters, "voter")
        if len(row_ids) != n_rows:
            raise InvalidInputError(
                f"{len(row_ids)} voters do not follow the {n_rows} rows of the table"
            )
        scales_of_columns = list(resolved)
        shown_array = None if shown is None else np.asarray(shown)
        levels = np.zeros((n_rows, width), dtype=np.int64)
        for row_position, row in enumerate(rows):
            for column_position, cell in enumerate(row):
                if cell is None:
                    continue
                if isinstance(cell, (float, np.floating)):
                    if math.isnan(cell):
                        continue
                    if float(cell).is_integer():
                        cell = int(cell)
                item = columns[column_position]
                try:
                    level = scales_of_columns[column_position].level_of(cell)  # type: ignore[arg-type]
                except InvalidInputError as error:
                    raise InvalidInputError(f"item {item!r}: {error}", hint=error.hint) from error
                if (
                    shown_array is not None
                    and shown_array.shape == (n_rows, width)
                    and not bool(shown_array[row_position, column_position])
                ):
                    raise InvalidInputError(
                        f"voter {row_ids[row_position]!r} has a score for item {item!r} "
                        "in a cell marked as not shown"
                    )
                levels[row_position, column_position] = level
        return cls.from_arrays(
            levels,
            shown=shown,
            voters=row_ids,
            items=columns,
            item_scales=resolved,
            voter_weights=voter_weights,
        )

    @classmethod
    def from_arrays(
        cls,
        levels: ArrayLike,
        *,
        shown: ArrayLike | None = None,
        voters: Iterable[str] | None = None,
        items: Iterable[str] | None = None,
        scale: OrdinalScale | None = None,
        item_scales: ScalesOfItems | None = None,
        voter_weights: ArrayLike | None = None,
    ) -> "VoteMatrix":
        """Build a matrix from its internal encoding, as NumPy arrays or lists.

        Args:
            levels: Shape (M, P). 0: no vote; ``k``: vote at rank ``k`` of the scale of
                the column.
            shown: Shape (M, P). Defaults to every cell shown.
            voters: Identifiers of the rows. Defaults to ``v1, v2, ...``.
            items: Identifiers of the columns. Defaults to ``i1, i2, ...``.
            scale: The scale of every item. Defaults to the five-score agreement scale.
            item_scales: The scale of each item, in place of ``scale``: an
                :class:`ItemScales` in column order, or a mapping by item identifier.
            voter_weights: Optional survey weights, one per row.
        """
        level_array = np.asarray(levels)
        if level_array.ndim != 2:
            raise InvalidInputError(
                f"levels are a matrix of shape (voters, items), got shape {level_array.shape}"
            )
        shown_array = (
            np.ones(level_array.shape, dtype=np.bool_) if shown is None else np.asarray(shown)
        )
        columns = _numbered("i", level_array.shape[1]) if items is None else tuple(items)
        weights = None if voter_weights is None else np.asarray(voter_weights)
        rows = _numbered("v", level_array.shape[0]) if voters is None else tuple(voters)
        if item_scales is None:
            return cls(
                level_array,
                shown_array,
                rows,
                columns,
                OrdinalScale() if scale is None else scale,
                weights,
            )
        resolved = _resolved_item_scales(_identifiers(columns, "item"), scale, item_scales)
        return cls(level_array, shown_array, rows, columns, None, weights, item_scales=resolved)

    # Counts ---------------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """The number of voters and the number of items."""
        return (len(self.voters), len(self.items))

    @property
    def n_levels(self) -> int:
        """The number of scores of the longest scale: of the scale, when there is one.

        The arrays over the states of a cell have ``n_levels + 1`` entries; a state
        past the levels of the scale of an item never occurs in its column.
        """
        return self.item_scales.n_levels

    @cached_property
    def _likelihood_weights(self) -> FloatArray | None:
        """Mean-one survey weights: they average 1, so they sum to the number of voters."""
        weights = self.voter_weights
        if weights is None or not weights.size:
            return weights
        relative = weights / np.max(weights)
        scaled: FloatArray = relative / relative.mean()
        _frozen(scaled)
        return scaled

    @cached_property
    def voted(self) -> BoolArray:
        """Booleans of shape (M, P): the cell is shown and carries a vote."""
        voted: BoolArray = self.shown & (self.levels > 0)
        _frozen(voted)
        return voted

    @property
    def n_votes(self) -> int:
        """The number of voted cells."""
        return int(self.voted.sum())

    @property
    def n_shown(self) -> int:
        """The number of shown cells, voted or not."""
        return int(self.shown.sum())

    @property
    def n_active_voters(self) -> int:
        """The number of voters with at least one vote."""
        return int(self.voted.any(axis=1).sum())

    @property
    def n_active_items(self) -> int:
        """The number of items with at least one vote."""
        return int(self.voted.any(axis=0).sum())

    # Reading --------------------------------------------------------------------------

    def to_records(self) -> Iterator[Vote]:
        """Yield the votes one by one, voter by voter, in scores of the scale of each item.

        Only voted cells are yielded: a vote held out by :meth:`hold_out` is not.
        """
        for row, column in np.argwhere(self.voted):
            yield Vote(
                self.voters[row],
                self.items[column],
                self.item_scales[int(column)].score_of(int(self.levels[row, column])),
            )

    def __getitem__(self, key: tuple[Selector, Selector]) -> "VoteMatrix":
        """Return the sub-matrix of some voters and some items, selected by identifiers.

        Each side accepts ``:``, one identifier, or a collection of identifiers (a
        ``VoterGroup`` or an ``ItemBlock`` included). The order of the matrix is kept,
        and each item keeps its scale.

        Raises:
            TypeError: If a side is an integer position or a slice other than ``:``.
            InvalidInputError: If an identifier is not in the matrix.
        """
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(
                "select a sub-matrix with two sides, matrix[voters, items]; "
                "use ':' to keep a whole side"
            )
        rows = _selection(key[0], self.voters, "voter")
        columns = _selection(key[1], self.items, "item")
        return VoteMatrix(
            self.levels[np.ix_(rows, columns)],
            self.shown[np.ix_(rows, columns)],
            tuple(self.voters[row] for row in rows),
            tuple(self.items[column] for column in columns),
            None,
            None if self.voter_weights is None else self.voter_weights[rows],
            item_scales=self.item_scales._select(columns),
        )

    def group_by(
        self,
        groups: Partition[VoterGroup] | None = None,
        blocks: Partition[ItemBlock] | None = None,
    ) -> "GroupedVotes":
        """Read the matrix through groups of voters and blocks of items.

        A side left out counts as a single cluster that holds everyone.

        Raises:
            InvalidInputError: If a partition does not cover the voters, or the items,
                of the matrix in the same order.
        """
        n_voters, n_items = self.shape
        if groups is None:
            groups = Partition.from_voter_labels(
                self.voters, np.zeros(n_voters, dtype=np.int64), n_clusters=1
            )
        if blocks is None:
            blocks = Partition.from_item_labels(
                self.items, np.zeros(n_items, dtype=np.int64), n_clusters=1
            )
        return GroupedVotes(self, groups, blocks)

    def hold_out(self, cells: BoolArray) -> "VoteMatrix":
        """Return the same matrix where ``cells`` are treated as never shown.

        Their votes stay in ``levels``, to be predicted; this matrix is unchanged.

        See the mathematical model: :doc:`/model/choosing-the-shape`, equations
        :eq:`eq-shape-held-out-single` and :eq:`eq-shape-held-out-mixed`.

        Raises:
            InvalidInputError: If ``cells`` is not a boolean array of the same shape.
        """
        mask = np.asarray(cells)
        if mask.dtype.kind != "b" or mask.shape != self.levels.shape:
            raise InvalidInputError(
                f"cells to hold out are booleans of shape {self.levels.shape}, "
                f"got {mask.dtype} of shape {mask.shape}"
            )
        return VoteMatrix(
            self.levels,
            self.shown & ~mask,
            self.voters,
            self.items,
            None,
            self.voter_weights,
            item_scales=self.item_scales,
        )

    def fingerprint(self) -> str:
        """Return a stable fingerprint of the content: 16 hexadecimal characters.

        SHA-256 of the shape, the scores, the identifiers, then the bytes of ``levels``
        (little-endian 64-bit integers), of ``shown``, and of the survey weights when
        they are present. With one scale, the scores are those of the scale; with
        several, the scores of each scale followed by the scale of each item, so that a
        matrix on one scale keeps the fingerprint it had before items could have
        several.
        """
        return self._fingerprint

    @cached_property
    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        scales = self.item_scales
        if scales.is_uniform:
            head: tuple[object, ...] = (
                self.shape,
                scales.scales[0].scores,
                self.voters,
                self.items,
            )
        else:
            head = (
                self.shape,
                tuple(scale.scores for scale in scales.scales),
                tuple(scales.scale_index.tolist()),
                self.voters,
                self.items,
            )
        digest.update(repr(head).encode())
        digest.update(np.ascontiguousarray(self.levels, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(self.shown, dtype=np.uint8).tobytes())
        if self.voter_weights is not None:
            digest.update(np.ascontiguousarray(self.voter_weights, dtype="<f8").tobytes())
        return digest.hexdigest()[:16]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VoteMatrix):
            return NotImplemented
        return self.fingerprint() == other.fingerprint()

    def __hash__(self) -> int:
        return hash(self.fingerprint())

    def _states(self, *, ignore_shown: bool = False) -> FloatArray:
        """Return the indicator of the state of each cell, of shape (M, P, K + 1).

        Index 0: shown without a vote; index ``k``: vote at rank ``k`` of the scale of
        the column. ``K`` is the number of levels of the longest scale: in a column on a
        shorter scale, the indices past its levels are always 0. A cell that is not
        shown is zero everywhere, unless ``ignore_shown`` is true: the mask is then left
        out and every cell is read through ``levels`` alone.

        Computed once and read-only. Models read the matrix through this array only.
        """
        return self._unmasked_states if ignore_shown else self._masked_states

    @cached_property
    def _unmasked_states(self) -> FloatArray:
        states: FloatArray = np.zeros((*self.levels.shape, self.n_levels + 1), dtype=np.float64)
        np.put_along_axis(states, self.levels[..., np.newaxis], 1.0, axis=-1)
        _frozen(states)
        return states

    @cached_property
    def _masked_states(self) -> FloatArray:
        states: FloatArray = self._unmasked_states * self.shown[..., np.newaxis]
        _frozen(states)
        return states

    def _describe(self) -> list[str]:
        n_voters, n_items = self.shape
        n_cells = n_voters * n_items
        scales = self.item_scales
        if scales.is_uniform:
            reading = f"{self.n_levels} levels"
        else:
            reading = f"{scales.n_scales} scales of {_span(scales.scales)} levels"
        identity = (
            f"VoteMatrix({n_voters} voter{_plural(n_voters)} x {n_items} item{_plural(n_items)}, "
            f"{self.n_votes} vote{_plural(self.n_votes)}, {reading})"
        )
        if n_cells == 0:
            return [identity, "no cell"]
        shown_share = self.n_shown / n_cells
        voted_share = self.n_votes / self.n_shown if self.n_shown else 0.0
        return [
            identity,
            f"{shown_share:.1%} of the {n_cells} cells shown, "
            f"{voted_share:.1%} of the shown cells voted",
        ]


@dataclass(frozen=True, eq=False, repr=False)
class GroupedVotes(Displayable):
    """A vote matrix read through groups of voters and blocks of items.

    The groups and the blocks can come from a model or from anywhere else. Only the
    plain labels of the partitions are read here: a degree of membership is not.

    Attributes:
        votes: The vote matrix.
        groups: A partition of the voters of the matrix, in the same order.
        blocks: A partition of the items of the matrix, in the same order.

    Raises:
        InvalidInputError: If a partition does not cover the voters, or the items, of
            the matrix in the same order.
    """

    votes: VoteMatrix
    groups: Partition[VoterGroup]
    blocks: Partition[ItemBlock]

    def __post_init__(self) -> None:
        if not isinstance(self.votes, VoteMatrix):
            raise InvalidInputError(f"votes are a VoteMatrix, got {self.votes!r}")
        for partition, kind, side, expected in (
            (self.groups, VoterGroup, "voters", self.votes.voters),
            (self.blocks, ItemBlock, "items", self.votes.items),
        ):
            if not isinstance(partition, Partition) or not issubclass(partition.kind, kind):
                raise InvalidInputError(
                    f"the {side} are grouped by a Partition of {kind.__name__}, got {partition!r}"
                )
            if partition.ids != expected:
                raise InvalidInputError(
                    f"the partition does not cover the {side} of the matrix in the same order: "
                    f"{len(partition.ids)} identifiers for {len(expected)} {side}. "
                    f"Slice the votes first, matrix[voters, items], or rebuild the partition "
                    f"on the {side} of the matrix."
                )

    @property
    def shape(self) -> tuple[int, int]:
        """The number of groups and the number of blocks."""
        return (self.groups.n_clusters, self.blocks.n_clusters)

    def counts(self) -> FloatArray:
        """Count the shown cells of each co-cluster in each state, on a matrix of one scale.

        Shape (Q, L, K + 1). Index 0: shown without a vote; index ``k``: vote at rank
        ``k``. A cell that is not shown is counted nowhere. Without survey weights::

            N[q, l, s] = sum_{i in group q} sum_{j in block l} S[i, j, s]

        See the mathematical model: :doc:`/model/data`, equation :eq:`eq-data-counts`.

        Raises:
            InvalidInputError: If the items use several scales: a level then means
                another answer from one scale to the next; read
                :meth:`counts_by_scale`.
        """
        _only_scale(self.votes.scales, "this vote matrix")
        return count_states(
            self.votes._states(), self.groups.labels, self.blocks.labels, self.shape
        )

    def counts_by_scale(self) -> FloatArray:
        """Count the shown cells of each co-cluster in each state, scale by scale.

        Shape (Q, L, C, K + 1), with C the scales of the items and K the levels of the
        longest one. ``counts[q, l, c, s]`` is the number of shown cells of the items
        of the scale ``c`` in the co-cluster ``(q, l)`` in the state ``s``: 0 for no
        vote, ``k`` for a vote at rank ``k`` of that scale. The states past the levels
        of a scale are 0. With one scale, ``counts_by_scale()[:, :, 0]`` is
        :meth:`counts`.
        """
        scales = self.votes.item_scales
        return count_states_by_scale(
            self.votes._states(),
            self.groups.labels,
            self.blocks.labels,
            self.shape,
            scales.scale_index,
            scales.n_scales,
        )

    def to_table(self) -> "CoclusterTable":
        """Return the table of co-clusters as it is counted, without a model.

        ``vote_rate`` is the number of votes over the number of shown cells (0 when no
        cell is shown). On each scale, the shares of the votes are the count of each
        level over the number of votes of the co-cluster on that scale (a uniform
        distribution when there is none).
        """
        from civiqhub_selene.blocks.coclusters import CoclusterTable

        counts = self.counts_by_scale()
        scales = self.votes.scales
        n_votes = counts[..., 1:].sum(axis=-1).sum(axis=-1)
        n_shown = counts.sum(axis=-1).sum(axis=-1)
        vote_rate = np.divide(n_votes, n_shown, out=np.zeros_like(n_votes), where=n_shown > 0)
        shares: list[FloatArray] = []
        for position, scale in enumerate(scales):
            by_level = counts[:, :, position, 1 : scale.n_levels + 1]
            on_scale = by_level.sum(axis=-1)
            uniform = np.full_like(by_level, 1.0 / scale.n_levels)
            shares.append(
                np.divide(
                    by_level,
                    on_scale[..., np.newaxis],
                    out=uniform,
                    where=on_scale[..., np.newaxis] > 0,
                )
            )
        if len(scales) == 1:
            return CoclusterTable(scales[0], vote_rate, shares[0], n_votes)
        return CoclusterTable(scales, vote_rate, tuple(shares), n_votes)

    def vote_counts(self, group: int, block: int) -> VoteCounts:
        """Return the five vote counts of a co-cluster, ready for the ordinal family.

        The counts are named ``"group {group} x block {block}"``. They follow the rank
        of the scores on the scale: ``N[q, l, 1]`` to ``N[q, l, 5]``, without the
        abstentions ``N[q, l, 0]``.

        See the mathematical model: :doc:`/model/data`, equation :eq:`eq-data-counts`.

        Raises:
            InvalidInputError: If the items use several scales, if the scale does not
                have five scores, or if the group or the block does not exist.
        """
        scale = _only_scale(self.votes.scales, "this vote matrix")
        if scale.n_levels != len(SCORES):
            raise InvalidInputError(
                f"vote counts are defined on a scale of five scores, "
                f"got {scale.n_levels} scores: {scale.scores!r}"
            )
        self._check(group, block, InvalidInputError)
        counts = self.counts()[group, block, 1:]
        first, second, third, fourth, fifth = (round(float(count)) for count in counts)
        return VoteCounts(
            (first, second, third, fourth, fifth), name=f"group {group} x block {block}"
        )

    def __getitem__(self, key: tuple[int, int]) -> VoteMatrix:
        """Return the sub-matrix of the co-cluster of a group and a block.

        Raises:
            IndexError: If the group or the block does not exist.
        """
        try:
            group, block = key
        except (TypeError, ValueError):
            raise IndexError(
                f"a co-cluster is read as grouped[group, block], got {key!r}"
            ) from None
        self._check(group, block, IndexError)
        return self.votes[self.groups[group], self.blocks[block]]

    @property
    def lineage(self) -> tuple[str, ...]:
        """Fingerprints of the computations the two partitions come from, without repeat.

        ``Estimator`` copies it into the provenance: a result computed on groups cites
        the computation that produced them.
        """
        computations = (self.groups.computation, self.blocks.computation)
        return tuple(dict.fromkeys(c for c in computations if c is not None))

    def fingerprint(self) -> str:
        """Return a stable fingerprint of the votes and the two partitions."""
        parts = (self.votes.fingerprint(), self.groups.fingerprint(), self.blocks.fingerprint())
        return hashlib.sha256(repr(parts).encode()).hexdigest()[:16]

    def _check(self, group: object, block: object, error: type[Exception]) -> None:
        n_groups, n_blocks = self.shape
        if not (_is_integer(group) and _is_integer(block)) or not (
            0 <= group < n_groups and 0 <= block < n_blocks  # type: ignore[operator]
        ):
            raise error(
                f"there are {n_groups} groups and {n_blocks} blocks, "
                f"got group {group!r} and block {block!r}"
            )

    def _describe(self) -> list[str]:
        n_voters, n_items = self.votes.shape
        n_groups, n_blocks = self.shape
        return [
            f"GroupedVotes({n_voters} voter{_plural(n_voters)} in {n_groups} "
            f"group{_plural(n_groups)} x {n_items} item{_plural(n_items)} in {n_blocks} "
            f"block{_plural(n_blocks)}, {self.votes.n_votes} vote{_plural(self.votes.n_votes)})"
        ]


def _plural(count: int) -> str:
    return "" if count == 1 else "s"


def _span(scales: tuple[OrdinalScale, ...]) -> str:
    """Say how many levels the scales have: ``"5"`` or ``"2 to 7"``."""
    levels = sorted({scale.n_levels for scale in scales})
    return f"{levels[0]}" if len(levels) == 1 else f"{levels[0]} to {levels[-1]}"


def _selection(selector: object, identifiers: tuple[str, ...], kind: str) -> list[int]:
    """Translate one side of a selection into positions, in the order of the matrix."""
    if isinstance(selector, slice):
        if selector != slice(None):
            raise TypeError(
                f"select {kind}s by identifiers, not by positions: got the slice {selector!r}"
            )
        return list(range(len(identifiers)))
    if _is_integer(selector) or isinstance(selector, bool):
        raise TypeError(
            f"select {kind}s by identifiers, not by positions: got the position {selector!r}"
        )
    if isinstance(selector, str):
        wanted = {selector}
    elif isinstance(selector, Iterable):
        wanted = set(selector)
    else:
        raise TypeError(
            f"select {kind}s with ':', one identifier or a collection of identifiers, "
            f"got {selector!r}"
        )
    known = set(identifiers)
    for identifier in wanted:
        if identifier not in known:
            raise InvalidInputError(f"the {kind} {identifier!r} is not in the matrix")
    return [position for position, identifier in enumerate(identifiers) if identifier in wanted]
