# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Partitions of identifiers into clusters: voters into groups, items into blocks.

A partition can come from a model of the engine or from anywhere else (a k-means, a
list written by hand): this is what lets votes be read through groups whatever their
origin. A partition does not know the vote matrix; the link is made by
``GroupedVotes``.

Everywhere in the engine, "group" means a cluster of voters and "block" a cluster of
items.
"""

import hashlib
import itertools
from collections.abc import Iterable, Iterator, Sequence, Set
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar, Generic, TypeVar, overload

import numpy as np
from numpy.typing import ArrayLike

from civiqhub_selene.blocks._arrays import FloatArray, IntArray
from civiqhub_selene.errors import InvalidInputError, InvalidParameterError
from civiqhub_selene.types import Displayable

__all__ = ["Cluster", "ItemBlock", "Partition", "VoterGroup"]

_MAX_CLUSTERS_FOR_AGREEMENT = 8
_ROW_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True, repr=False)
class Cluster(Set[str]):
    """One cluster of a partition: a set of identifiers with its index and its share.

    It behaves as a set of identifiers: ``in``, ``len``, iteration and the set
    operations, which return a ``frozenset``. Two clusters are equal when they have the
    same index, the same identifiers and the same share.

    Attributes:
        index: Index of the cluster in its partition.
        ids: The identifiers of the cluster, in the order of the partition.
        share: Share of the cluster in the whole, between 0 and 1.
    """

    index: int
    ids: tuple[str, ...]
    share: float

    _member_noun: ClassVar[str] = "id"
    _cluster_noun: ClassVar[str] = "cluster"

    @cached_property
    def _members(self) -> frozenset[str]:
        return frozenset(self.ids)

    def __contains__(self, identifier: object) -> bool:
        return identifier in self._members

    def __iter__(self) -> Iterator[str]:
        return iter(self.ids)

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def _from_iterable(cls, it: Iterable[Any]) -> frozenset[Any]:
        return frozenset(it)

    def __repr__(self) -> str:
        count = len(self.ids)
        noun = self._member_noun + ("" if count == 1 else "s")
        return f"{type(self).__name__}(index={self.index}, {count} {noun}, share={self.share:.3f})"


@dataclass(frozen=True, repr=False)
class VoterGroup(Cluster):
    """A group: a cluster of voters."""

    _member_noun: ClassVar[str] = "voter"
    _cluster_noun: ClassVar[str] = "group"

    @property
    def voters(self) -> tuple[str, ...]:
        """The voters of the group."""
        return self.ids


@dataclass(frozen=True, repr=False)
class ItemBlock(Cluster):
    """A block: a cluster of items."""

    _member_noun: ClassVar[str] = "item"
    _cluster_noun: ClassVar[str] = "block"

    @property
    def items(self) -> tuple[str, ...]:
        """The items of the block."""
        return self.ids


ClusterT = TypeVar("ClusterT", bound=Cluster)


def _read_only(array: ArrayLike, dtype: type[Any], name: str) -> Any:
    try:
        copy = np.array(array, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise InvalidInputError(f"{name} is an array of numbers: {error}") from error
    copy.setflags(write=False)
    return copy


@dataclass(frozen=True, eq=False, repr=False)
class Partition(Displayable, Sequence[ClusterT], Generic[ClusterT]):
    """Identifiers arranged in clusters. A sequence of its clusters.

    ``len(partition)`` is the number of clusters, ``partition[k]`` is a cluster of the
    type ``kind``, built once. Arrays are copied and made read-only. Build a partition
    with :meth:`from_voter_labels` or :meth:`from_item_labels`.

    Attributes:
        kind: The type of the clusters: ``VoterGroup`` or ``ItemBlock``.
        ids: Every identifier, in order.
        labels: For each identifier, the index of its cluster.
        n_clusters: The number of clusters, empty clusters included.
        memberships: Shape (n, n_clusters): degree of membership of each identifier to
            each cluster, rows summing to 1. ``None`` for a plain partition.
        shares: The share of each cluster. Defaults to the sizes divided by n, or to
            the mean of the memberships when they are given.
        computation: Fingerprint of the computation the partition comes from, if any.

    Raises:
        InvalidInputError: If there are not as many labels as identifiers, if a label
            is not an integer from 0 to ``n_clusters - 1``, if an identifier is
            repeated, or if the memberships have the wrong shape, a negative value or
            a row that does not sum to 1 within 1e-6.
    """

    kind: type[ClusterT]
    ids: tuple[str, ...]
    labels: IntArray
    n_clusters: int
    memberships: FloatArray | None = None
    shares: FloatArray | None = None
    computation: str | None = None

    def __post_init__(self) -> None:
        if not (isinstance(self.kind, type) and issubclass(self.kind, Cluster)):
            raise InvalidInputError(
                f"the kind of a partition is VoterGroup or ItemBlock, got {self.kind!r}"
            )
        noun = self.kind._member_noun
        if isinstance(self.ids, str):
            raise InvalidInputError(f"{noun} identifiers are a collection, got {self.ids!r}")
        ids = tuple(self.ids)
        seen: set[str] = set()
        for identifier in ids:
            if not isinstance(identifier, str):
                raise InvalidInputError(f"a {noun} identifier is text, got {identifier!r}")
            if identifier in seen:
                raise InvalidInputError(f"the {noun} identifier {identifier!r} is repeated")
            seen.add(identifier)
        n_ids = len(ids)

        raw_labels = np.asarray(self.labels)
        if raw_labels.size and raw_labels.dtype.kind not in "iu":
            raise InvalidInputError(
                f"labels are integers, the index of a cluster, got an array of {raw_labels.dtype}"
            )
        if raw_labels.shape != (n_ids,):
            raise InvalidInputError(
                f"there is one label per {noun}: {n_ids} {noun}s, "
                f"labels of shape {raw_labels.shape}"
            )
        labels: IntArray = _read_only(raw_labels, np.int64, "labels")
        n_clusters = self.n_clusters
        if isinstance(n_clusters, bool) or not isinstance(n_clusters, (int, np.integer)):
            raise InvalidInputError(f"n_clusters is an integer, got {n_clusters!r}")
        n_clusters = int(n_clusters)
        outside = labels[(labels < 0) | (labels >= n_clusters)]
        if n_clusters < 0 or outside.size:
            detail = f"got the label {int(outside[0])}" if outside.size else f"got {n_clusters}"
            raise InvalidInputError(
                f"a label is an integer from 0 to n_clusters - 1 = {n_clusters - 1}, {detail}"
            )

        memberships: FloatArray | None = None
        if self.memberships is not None:
            memberships = _read_only(self.memberships, np.float64, "memberships")
            if memberships.shape != (n_ids, n_clusters):
                raise InvalidInputError(
                    f"memberships have the shape ({noun}s, clusters) = ({n_ids}, {n_clusters}), "
                    f"got {memberships.shape}"
                )
            if not np.isfinite(memberships).all() or (memberships < 0).any():
                raise InvalidInputError("memberships are finite and not negative")
            sums = memberships.sum(axis=1)
            wrong = np.flatnonzero(np.abs(sums - 1.0) > _ROW_SUM_TOLERANCE)
            if wrong.size:
                row = int(wrong[0])
                raise InvalidInputError(
                    f"the memberships of {ids[row]!r} sum to {float(sums[row])!r}, not to 1"
                )

        shares: FloatArray
        if self.shares is not None:
            shares = _read_only(self.shares, np.float64, "shares")
            if shares.shape != (n_clusters,):
                raise InvalidInputError(
                    f"there is one share per cluster: {n_clusters} clusters, "
                    f"shares of shape {shares.shape}"
                )
            if not np.isfinite(shares).all() or (shares < 0).any():
                raise InvalidInputError("shares are finite and not negative")
        elif n_ids == 0:
            shares = _read_only(np.zeros(n_clusters), np.float64, "shares")
        elif memberships is not None:
            shares = _read_only(memberships.mean(axis=0), np.float64, "shares")
        else:
            sizes = np.bincount(labels, minlength=n_clusters)
            shares = _read_only(sizes / n_ids, np.float64, "shares")
        if self.computation is not None and not isinstance(self.computation, str):
            raise InvalidInputError(
                f"computation is a fingerprint in text, got {self.computation!r}"
            )

        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "n_clusters", n_clusters)
        object.__setattr__(self, "memberships", memberships)
        object.__setattr__(self, "shares", shares)

    # Constructors ---------------------------------------------------------------------

    @classmethod
    def from_voter_labels(
        cls,
        ids: Iterable[str],
        labels: ArrayLike,
        *,
        n_clusters: int | None = None,
        memberships: ArrayLike | None = None,
        shares: ArrayLike | None = None,
        computation: str | None = None,
    ) -> "Partition[VoterGroup]":
        """Build a partition of voters into groups from one label per voter.

        Args:
            ids: The voters, in order.
            labels: For each voter, the index of its group.
            n_clusters: The number of groups. Defaults to the number of columns of
                ``memberships`` when they are given, to ``max(labels) + 1`` otherwise.
            memberships: Degree of membership of each voter to each group, if any.
            shares: The share of each group, if it is not the one of the sizes.
            computation: Fingerprint of the computation the partition comes from.
        """
        return _from_labels(VoterGroup, ids, labels, n_clusters, memberships, shares, computation)

    @classmethod
    def from_item_labels(
        cls,
        ids: Iterable[str],
        labels: ArrayLike,
        *,
        n_clusters: int | None = None,
        memberships: ArrayLike | None = None,
        shares: ArrayLike | None = None,
        computation: str | None = None,
    ) -> "Partition[ItemBlock]":
        """Build a partition of items into blocks from one label per item.

        Same arguments as :meth:`from_voter_labels`.
        """
        return _from_labels(ItemBlock, ids, labels, n_clusters, memberships, shares, computation)

    # Sequence of clusters -------------------------------------------------------------

    @cached_property
    def _clusters(self) -> tuple[ClusterT, ...]:
        assert self.shares is not None
        members: list[list[str]] = [[] for _ in range(self.n_clusters)]
        for identifier, label in zip(self.ids, self.labels.tolist(), strict=True):
            members[label].append(identifier)
        return tuple(
            self.kind(index, tuple(members[index]), float(self.shares[index]))
            for index in range(self.n_clusters)
        )

    @cached_property
    def _positions(self) -> dict[str, int]:
        return {identifier: position for position, identifier in enumerate(self.ids)}

    def __len__(self) -> int:
        return self.n_clusters

    @overload
    def __getitem__(self, index: int) -> ClusterT: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[ClusterT, ...]: ...
    def __getitem__(self, index: int | slice) -> ClusterT | tuple[ClusterT, ...]:
        return self._clusters[index]

    def cluster_of(self, identifier: str) -> ClusterT:
        """Return the cluster of an identifier.

        Raises:
            InvalidInputError: If the identifier is not in the partition.
        """
        position = self._positions.get(identifier) if isinstance(identifier, str) else None
        if position is None:
            raise InvalidInputError(
                f"the {self.kind._member_noun} {identifier!r} is not in the partition"
            )
        return self._clusters[int(self.labels[position])]

    @property
    def sizes(self) -> tuple[int, ...]:
        """The number of identifiers in each cluster."""
        return tuple(len(cluster) for cluster in self._clusters)

    @property
    def is_mixed(self) -> bool:
        """Whether degrees of membership are given."""
        return self.memberships is not None

    def agreement_with(self, other: "Partition[Any]") -> float:
        """Return the share of identifiers the two partitions arrange in the same way.

        The clusters of ``other`` are renumbered in the best possible way first, by
        exhaustive search over the permutations. With ``T[q, r]`` the number of
        identifiers in the cluster ``q`` of this partition and ``r`` of ``other``::

            agreement = max_chi sum_q T[q, chi(q)] / len(ids)

        where ``chi`` runs over the renumberings of the clusters of ``other``.

        See the mathematical model: :doc:`/model/reading-results`, equation
        :eq:`eq-results-match`.

        Raises:
            InvalidInputError: If the two partitions do not cover the same identifiers
                in the same order, or if they are empty.
            InvalidParameterError: If a partition has more than 8 clusters.
        """
        if not isinstance(other, Partition) or self.ids != other.ids:
            raise InvalidInputError(
                "agreement compares two partitions of the same identifiers in the same order"
            )
        if not self.ids:
            raise InvalidInputError("agreement is undefined on a partition without identifier")
        n_clusters = max(self.n_clusters, other.n_clusters)
        if n_clusters > _MAX_CLUSTERS_FOR_AGREEMENT:
            raise InvalidParameterError(
                f"agreement searches every renumbering and stops at "
                f"{_MAX_CLUSTERS_FOR_AGREEMENT} clusters, got {n_clusters}"
            )
        table = np.zeros((n_clusters, n_clusters), dtype=np.int64)
        np.add.at(table, (self.labels, other.labels), 1)
        rows = range(n_clusters)
        best = max(
            sum(int(table[row, column]) for row, column in zip(rows, columns, strict=True))
            for columns in itertools.permutations(rows)
        )
        return best / len(self.ids)

    def fingerprint(self) -> str:
        """Return a stable fingerprint: 16 hexadecimal characters.

        SHA-256 of the kind, the identifiers and the number of clusters, then the bytes
        of ``labels``. Memberships and shares are left out: it identifies the
        arrangement of identifiers in clusters.
        """
        return self._fingerprint

    @cached_property
    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(repr((self.kind.__name__, self.ids, self.n_clusters)).encode())
        digest.update(np.ascontiguousarray(self.labels, dtype="<i8").tobytes())
        return digest.hexdigest()[:16]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Partition):
            return NotImplemented
        return (
            self.fingerprint() == other.fingerprint()
            and self.computation == other.computation
            and _same(self.memberships, other.memberships)
            and _same(self.shares, other.shares)
        )

    def __hash__(self) -> int:
        return hash(self.fingerprint())

    def _describe(self) -> list[str]:
        kind = self.kind
        count = len(self.ids)
        members = kind._member_noun + ("" if count == 1 else "s")
        clusters = kind._cluster_noun + ("" if self.n_clusters == 1 else "s")
        mixed = "mixed " if self.is_mixed else ""
        sizes = ", ".join(str(size) for size in self.sizes)
        return [f"Partition({count} {members} in {self.n_clusters} {mixed}{clusters}: {sizes})"]


def _same(left: FloatArray | None, right: FloatArray | None) -> bool:
    if left is None or right is None:
        return left is right
    return bool(np.array_equal(left, right))


def _from_labels(
    kind: type[ClusterT],
    ids: Iterable[str],
    labels: ArrayLike,
    n_clusters: int | None,
    memberships: ArrayLike | None,
    shares: ArrayLike | None,
    computation: str | None,
) -> Partition[ClusterT]:
    label_array = np.asarray(labels)
    membership_array = None if memberships is None else np.asarray(memberships)
    if n_clusters is None:
        if membership_array is not None and membership_array.ndim == 2:
            n_clusters = int(membership_array.shape[1])
        elif label_array.size > 0 and label_array.dtype.kind in "iu":
            n_clusters = int(label_array.max(initial=-1)) + 1
        else:
            n_clusters = 0
    return Partition(
        kind,
        tuple(ids) if not isinstance(ids, str) else ids,  # type: ignore[arg-type]
        label_array,
        n_clusters,
        membership_array,
        None if shares is None else np.asarray(shares),
        computation,
    )
