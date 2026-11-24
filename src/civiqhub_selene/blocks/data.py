# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Votes of voters on items, as a matrix with three states per cell.

:class:`VoteMatrix` is the input of all co-clustering. Its central rule: a cell is
voted, shown without a vote, or not shown. An empty cell is never a zero nor a neutral
answer, and only shown cells enter a computation.
"""

import hashlib
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from itertools import pairwise
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from civiqhub_selene.blocks._arrays import BoolArray, FloatArray, IntArray
from civiqhub_selene.blocks._numerics import count_states
from civiqhub_selene.blocks.partition import ItemBlock, Partition, VoterGroup
from civiqhub_selene.errors import InvalidInputError, InvalidParameterError
from civiqhub_selene.ordinal.counts import SCORES, VoteCounts
from civiqhub_selene.types import Displayable

if TYPE_CHECKING:
    from civiqhub_selene.blocks.coclusters import CoclusterTable
    from civiqhub_selene.blocks.covariates import VoterCovariates as VoterCovariates

__all__ = ["GroupedVotes", "OrdinalScale", "Vote", "VoteMatrix"]

Selector: TypeAlias = slice | str | Iterable[str]
"""One side of a selection: ``:``, one identifier, or a collection of identifiers."""


def _is_integer(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


@dataclass(frozen=True)
class OrdinalScale:
    """An ordered scale of scores. Defaults to the five-score agreement scale.

    A score is the answer of a vote, for example from -2 to +2. A level is the rank of
    a score on the scale, from 1 to ``n_levels``; level 0 means "no vote" and is
    internal to matrices.

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


def _shared_item_scale(
    item_scales: Mapping[str, OrdinalScale],
    columns: tuple[str, ...],
    scale: OrdinalScale | None,
) -> OrdinalScale:
    """Return the unique observation scale of every item, or refuse the first mismatch."""
    if set(item_scales) != set(columns):
        raise InvalidInputError("item_scales must have exactly the response table items")
    sources = tuple(item_scales[item] for item in columns)
    if not all(isinstance(source, OrdinalScale) for source in sources):
        raise InvalidInputError("item_scales must contain an OrdinalScale for each item")
    if not sources:
        return OrdinalScale() if scale is None else scale
    reference_item = columns[0]
    reference = sources[0]
    for item, source in zip(columns[1:], sources[1:], strict=True):
        if source.scores != reference.scores:
            raise InvalidInputError(
                f"item {item!r} uses the scale {source._formatted()}, "
                f"which differs from {reference_item!r} ({reference._formatted()})",
                hint=(
                    "Give every item the same OrdinalScale; per-item observation "
                    "scales are SEL-8.2 (ItemScales)."
                ),
            )
    if scale is None:
        return reference
    if not isinstance(scale, OrdinalScale):
        raise InvalidInputError(f"the scale is an OrdinalScale, got {scale!r}")
    if scale.scores != reference.scores:
        raise InvalidInputError(
            f"the given scale {scale._formatted()} differs from the item scale "
            f"{reference._formatted()}",
            hint="Omit scale, or pass the same OrdinalScale as every item.",
        )
    return scale


def _frozen(array: NDArray[Any]) -> None:
    array.setflags(write=False)


@dataclass(frozen=True, eq=False, repr=False)
class VoteMatrix(Displayable):
    """Votes of M voters on P items, by opaque identifiers.

    A cell has three states. It is *not shown* when ``shown`` is false, whatever
    ``levels`` holds; an *abstention* when it is shown and ``levels`` is 0; *voted*
    when it is shown and ``levels`` is positive. A vote at the neutral score is a vote.

    The arrays are copied and made read-only. Two matrices with the same content are
    equal: equality and hash follow :meth:`fingerprint`.

    Attributes:
        levels: Integers of shape (M, P). 0: no vote; ``k`` from 1 to ``n_levels``:
            vote at rank ``k`` of the scale.
        shown: Booleans of shape (M, P): the item was shown to the voter.
        voters: Opaque identifiers of the rows, unique.
        items: Opaque identifiers of the columns, unique.
        scale: The scale of scores. Defaults to the five-score agreement scale.
        voter_weights: Optional finite, positive survey weights, in row order.

    Raises:
        InvalidInputError: If the arrays are not two matrices of the same shape, if a
            level is outside ``0..n_levels``, if the identifiers do not follow the
            shape, or if an identifier is repeated.
    """

    levels: IntArray
    shown: BoolArray
    voters: tuple[str, ...]
    items: tuple[str, ...]
    scale: OrdinalScale = field(default_factory=OrdinalScale)
    voter_weights: FloatArray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scale, OrdinalScale):
            raise InvalidInputError(f"the scale is an OrdinalScale, got {self.scale!r}")
        raw_levels = np.asarray(self.levels)
        if raw_levels.size and raw_levels.dtype.kind not in "iu":
            raise InvalidInputError(
                f"levels are integers from 0 to {self.scale.n_levels}, "
                f"got an array of {raw_levels.dtype}"
            )
        raw_shown = np.asarray(self.shown)
        if raw_shown.size and raw_shown.dtype.kind != "b":
            raise InvalidInputError(f"shown is a boolean array, got an array of {raw_shown.dtype}")
        if raw_levels.ndim != 2 or raw_shown.ndim != 2 or raw_levels.shape != raw_shown.shape:
            raise InvalidInputError(
                "levels and shown are two matrices of the same shape (voters, items), "
                f"got shapes {raw_levels.shape} and {raw_shown.shape}"
            )
        levels: IntArray = np.array(raw_levels, dtype=np.int64)
        shown: BoolArray = np.array(raw_shown, dtype=np.bool_)
        outside = levels[(levels < 0) | (levels > self.scale.n_levels)]
        if outside.size:
            raise InvalidInputError(
                f"a level is a rank from 0 to {self.scale.n_levels}, got {int(outside[0])}"
            )
        voters = _identifiers(self.voters, "voter")
        items = _identifiers(self.items, "item")
        if (len(voters), len(items)) != levels.shape:
            raise InvalidInputError(
                f"{len(voters)} voters and {len(items)} items do not follow "
                f"the shape {levels.shape} of the matrix"
            )
        _frozen(levels)
        _frozen(shown)
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "shown", shown)
        object.__setattr__(self, "voters", voters)
        object.__setattr__(self, "items", items)
        if self.voter_weights is not None:
            received = np.asarray(self.voter_weights)
            if received.shape != (len(voters),) or received.dtype.kind not in "fiu":
                raise InvalidInputError("voter_weights must contain one numeric weight per voter")
            weights: FloatArray = np.array(received, dtype=np.float64)
            if not np.isfinite(weights).all() or (weights <= 0).any():
                raise InvalidInputError("voter_weights must be finite and strictly positive")
            _frozen(weights)
            object.__setattr__(self, "voter_weights", weights)

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
    ) -> "VoteMatrix":
        """Build a matrix from ``(voter, item, score)`` triples.

        Rows and columns follow the order of first appearance, in the votes and then
        in ``shown``, unless ``voters`` or ``items`` fix it.

        Without ``shown``, every cell counts as shown: an empty cell is then read as
        an abstention. With ``shown``, only these ``(voter, item)`` pairs and the
        voted cells are shown; the other cells are not shown.

        Args:
            votes: The votes, as ``(voter, item, score)`` triples.
            shown: The ``(voter, item)`` pairs where the item was shown to the voter.
            voters: The voters in the order of the rows. It may name voters without
                any vote; it must name every voter of the records.
            items: The items in the order of the columns, likewise.
            scale: The scale of scores. Defaults to the five-score agreement scale.

        Raises:
            InvalidInputError: If a score is not on the scale, if a voter votes twice
                on the same item, or if a record names an identifier that ``voters``
                or ``items`` leaves out.
        """
        scale = OrdinalScale() if scale is None else scale
        rows: dict[str, int] = {}
        columns: dict[str, int] = {}
        if voters is not None:
            rows = {voter: position for position, voter in enumerate(_identifiers(voters, "voter"))}
        if items is not None:
            columns = {item: position for position, item in enumerate(_identifiers(items, "item"))}

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
                position(columns, item, "item", items is not None),
            )
            if cell in cells:
                raise InvalidInputError(f"the voter {voter!r} votes twice on the item {item!r}")
            cells[cell] = scale.level_of(score)
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
                        position(columns, item, "item", items is not None),
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
        return cls(levels, shown_array, tuple(rows), tuple(columns), scale)

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
        item_scales: Mapping[str, OrdinalScale] | None = None,
        scale: OrdinalScale | None = None,
        shown: ArrayLike | None = None,
        voter_weights: ArrayLike | None = None,
    ) -> "VoteMatrix":
        """Build a survey matrix from a table of scores in voter and item order.

        With ``item_scales``, provide one scale for every item, keyed by its identifier.
        The order of the keys supplies the column order if ``items`` is omitted. The
        matrix keeps the original ranks on that scale, in the direction of agreement,
        and does not recode them onto a common grid.

        Every item must share the same observation scale. A shown cell of item ``j``
        has ``K_j + 1`` modalities (shown without a vote, or a vote at one of the
        ``K_j`` ranks). Distinct ``K_j`` wait on SEL-8.2 (ItemScales): the response
        models still use one shared ``K``. A homogeneous table matches
        :meth:`from_records` on the same scores, voters and items.

        ``None`` and NaN indicate a cell without a response. The ``shown`` mask
        distinguishes an abstention from a question that was not shown. A score in a
        cell marked as not shown is refused. Pass ``voter_weights`` in row order.
        Each weight must be finite and positive.

        Raises:
            InvalidInputError: If the dimensions or identifiers do not match, a
                score is outside its declared scale, a scored cell is marked as not
                shown, or the item scales are not all the same.
        """
        rows = [list(row) for row in table]
        widths = {len(row) for row in rows}
        if len(widths) > 1:
            raise InvalidInputError(
                f"the rows of a table have the same length, got lengths {sorted(widths)}"
            )
        if item_scales is not None and items is None:
            items = tuple(item_scales)
        identifiers = None if items is None else _identifiers(items, "item")
        width = widths.pop() if widths else len(identifiers or ())
        columns = _numbered("i", width) if identifiers is None else identifiers
        if len(columns) != width:
            raise InvalidInputError("the item identifiers must match the response table columns")

        if item_scales is not None:
            scale = _shared_item_scale(item_scales, columns, scale)
        scale = OrdinalScale() if scale is None else scale
        if not isinstance(scale, OrdinalScale):
            raise InvalidInputError(f"the scale is an OrdinalScale, got {scale!r}")

        n_rows = len(rows)
        row_ids = _numbered("v", n_rows) if voters is None else _identifiers(voters, "voter")
        if len(row_ids) != n_rows:
            raise InvalidInputError(
                f"{len(row_ids)} voters do not follow the {n_rows} rows of the table"
            )
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
                    level = scale.level_of(cell)  # type: ignore[arg-type]
                except InvalidInputError as error:
                    raise InvalidInputError(f"item {item!r}: {error}") from error
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
            scale=scale,
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
        voter_weights: ArrayLike | None = None,
    ) -> "VoteMatrix":
        """Build a matrix from its internal encoding, as NumPy arrays or lists.

        Args:
            levels: Shape (M, P). 0: no vote; ``k``: vote at rank ``k`` of the scale.
            shown: Shape (M, P). Defaults to every cell shown.
            voters: Identifiers of the rows. Defaults to ``v1, v2, ...``.
            items: Identifiers of the columns. Defaults to ``i1, i2, ...``.
            scale: The scale of scores. Defaults to the five-score agreement scale.
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
        return cls(
            level_array,
            shown_array,
            _numbered("v", level_array.shape[0]) if voters is None else tuple(voters),
            _numbered("i", level_array.shape[1]) if items is None else tuple(items),
            OrdinalScale() if scale is None else scale,
            None if voter_weights is None else np.asarray(voter_weights),
        )

    # Counts ---------------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """The number of voters and the number of items."""
        return (len(self.voters), len(self.items))

    @property
    def n_levels(self) -> int:
        """The number of scores on the scale."""
        return self.scale.n_levels

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
        """Yield the votes one by one, voter by voter, in scores of the scale.

        Only voted cells are yielded: a vote held out by :meth:`hold_out` is not.
        """
        for row, column in np.argwhere(self.voted):
            yield Vote(
                self.voters[row],
                self.items[column],
                self.scale.score_of(int(self.levels[row, column])),
            )

    def __getitem__(self, key: tuple[Selector, Selector]) -> "VoteMatrix":
        """Return the sub-matrix of some voters and some items, selected by identifiers.

        Each side accepts ``:``, one identifier, or a collection of identifiers (a
        ``VoterGroup`` or an ``ItemBlock`` included). The order of the matrix is kept.

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
            self.scale,
            None if self.voter_weights is None else self.voter_weights[rows],
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
            self.levels, self.shown & ~mask, self.voters, self.items, self.scale, self.voter_weights
        )

    def fingerprint(self) -> str:
        """Return a stable fingerprint of the content: 16 hexadecimal characters.

        SHA-256 of the shape, the scores, the identifiers, then the bytes of ``levels``
        (little-endian 64-bit integers), of ``shown``, and of the survey weights when
        they are present.
        """
        return self._fingerprint

    @cached_property
    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(repr((self.shape, self.scale.scores, self.voters, self.items)).encode())
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

        Index 0: shown without a vote; index ``k``: vote at rank ``k``. A cell that is
        not shown is zero everywhere, unless ``ignore_shown`` is true: the mask is then
        left out and every cell is read through ``levels`` alone.

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
        identity = (
            f"VoteMatrix({n_voters} voter{_plural(n_voters)} x {n_items} item{_plural(n_items)}, "
            f"{self.n_votes} vote{_plural(self.n_votes)}, {self.n_levels} levels)"
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
        """Count the shown cells of each co-cluster in each state.

        Shape (Q, L, K + 1). Index 0: shown without a vote; index ``k``: vote at rank
        ``k``. A cell that is not shown is counted nowhere.
        """
        return count_states(
            self.votes._states(), self.groups.labels, self.blocks.labels, self.shape
        )

    def to_table(self) -> "CoclusterTable":
        """Return the table of co-clusters as it is counted, without a model.

        ``vote_rate`` is the number of votes over the number of shown cells (0 when no
        cell is shown); ``vote_shares`` is the count of each level over the number of
        votes (a uniform distribution when there is no vote).
        """
        from civiqhub_selene.blocks.coclusters import CoclusterTable

        counts = self.counts()
        by_level = counts[..., 1:]
        n_votes = by_level.sum(axis=-1)
        n_shown = counts.sum(axis=-1)
        vote_rate = np.divide(n_votes, n_shown, out=np.zeros_like(n_votes), where=n_shown > 0)
        uniform = np.full_like(by_level, 1.0 / self.votes.n_levels)
        vote_shares = np.divide(
            by_level, n_votes[..., np.newaxis], out=uniform, where=n_votes[..., np.newaxis] > 0
        )
        return CoclusterTable(self.votes.scale, vote_rate, vote_shares, n_votes)

    def vote_counts(self, group: int, block: int) -> VoteCounts:
        """Return the five vote counts of a co-cluster, ready for the ordinal family.

        The counts are named ``"group {group} x block {block}"``. They follow the rank
        of the scores on the scale.

        Raises:
            InvalidInputError: If the scale does not have five scores, or if the group
                or the block does not exist.
        """
        if self.votes.n_levels != len(SCORES):
            raise InvalidInputError(
                f"vote counts are defined on a scale of five scores, "
                f"got {self.votes.n_levels} scores: {self.votes.scale.scores!r}"
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


def __getattr__(name: str) -> Any:
    """Re-export :class:`VoterCovariates` from :mod:`civiqhub_selene.blocks.covariates`."""
    if name == "VoterCovariates":
        from civiqhub_selene.blocks.covariates import VoterCovariates

        return VoterCovariates
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
