# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""How a block model starts: the arrangement of voters and items of each start.

A block model runs several starts and keeps the best one. Each start begins from an
arrangement: a group for each voter and a block for each item. The arrangement is
given by a start object, a setting of the model like its response model::

    LatentBlockModel(n_groups=3, n_blocks=2, initialization=SpectralStart())

Three start objects are shipped:

- :class:`RandomStart`, the default: a random arrangement where every cluster has
  members;
- :class:`SpectralStart`: the arrangement of regularised spectral co-clustering
  (DI-SIM, RSC), see :mod:`civiqhub_selene.blocks.spectral`;
- :class:`FixedStart`: an arrangement given by the caller, for example the one a
  single-membership model found, to start a mixed-membership model from it.

A start object is a frozen dataclass: its settings enter the provenance of the model
that carries it, as they are. Any object with the method ``arrangement`` of
:class:`Initialization` can be used.
"""

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from civiqhub_selene.blocks._arrays import IntArray
from civiqhub_selene.blocks.data import VoteMatrix
from civiqhub_selene.blocks.spectral import (
    ENCODINGS,
    Encoding,
    n_singular_values,
    spectral_embedding,
)
from civiqhub_selene.errors import InvalidInputError, InvalidParameterError

__all__ = ["FixedStart", "Initialization", "RandomStart", "SpectralStart"]


class Initialization(Protocol):
    """What a block model asks of the object that gives its starting arrangements."""

    def arrangement(
        self, votes: VoteMatrix, shape: tuple[int, int], rng: np.random.Generator
    ) -> tuple[IntArray, IntArray]:
        """Return the group of each voter and the block of each item for one start.

        Args:
            votes: The vote matrix.
            shape: ``(Q, L)``, the number of groups and of blocks.
            rng: The generator of the start; any draw goes through it.

        Returns:
            Two integer arrays, of shapes (M,) and (P,), with labels from 0 to Q - 1
            and from 0 to L - 1. No group and no block is empty.
        """
        ...


@dataclass(frozen=True)
class RandomStart:
    """A random arrangement where every group and every block has members.

    Definition. The voters are shuffled and dealt in turn to the groups: with ``psi``
    a random permutation of the M voters (``rng.permutation(M)``), the group of the
    voter ``i`` is ``R[i] = psi(i) mod Q``, counted from 0. The items are dealt to the
    blocks in the same way. The groups then differ in size by one at most. The two
    permutations are drawn in this order with the generator of the start.

    See the mathematical model: :doc:`/model/initialization`.
    """

    def arrangement(
        self, votes: VoteMatrix, shape: tuple[int, int], rng: np.random.Generator
    ) -> tuple[IntArray, IntArray]:
        """Return a random arrangement, groups first."""
        n_voters, n_items = votes.shape
        n_groups, n_blocks = shape
        groups: IntArray = (rng.permutation(n_voters) % n_groups).astype(np.int64)
        blocks: IntArray = (rng.permutation(n_items) % n_blocks).astype(np.int64)
        return groups, blocks


@dataclass(frozen=True)
class SpectralStart:
    """The arrangement of regularised spectral co-clustering of the votes.

    Definition. The votes are written as a matrix (``encoding``), divided by the square
    roots of the regularised degrees of rows and columns, and decomposed into singular
    values; the leading singular vectors place each voter and each item in a space of Q
    or L dimensions, where k-means finds the groups and the blocks. It is DI-SIM (Rohe,
    Qin and Yu 2016) for a vote matrix, with the regulariser and the projection onto the
    sphere of regularised spectral clustering (Qin and Rohe 2013). See
    :func:`~civiqhub_selene.blocks.spectral.spectral_embedding` for the matrix and
    :meth:`~civiqhub_selene.blocks.spectral.SpectralEmbedding.labels` for the k-means.

    The decomposition does not depend on the start: it is computed once for a vote
    matrix and kept in a small cache. Only the seeding of k-means changes from one
    start to the next, so the starts of a model are close to one another.

    References: Qin and Rohe (2013), Regularized spectral clustering under the
    degree-corrected stochastic blockmodel, NeurIPS 26; Rohe, Qin and Yu (2016),
    Co-clustering directed graphs to discover asymmetries and directional communities,
    PNAS 113(45).

    See the mathematical model: :doc:`/model/initialization`, equation
    :eq:`eq-starts-laplacian`.

    Attributes:
        encoding: ``"states"`` (default: one column per item and state, so that the
            abstentions and the spread of the votes count) or ``"scores"`` (one column
            per item, of centred scores).
        regularization: The regulariser ``tau`` as a multiple of the average degree of
            each side, a finite number of at least 0. Defaults to 1, the average
            degree; 0 is unregularised spectral clustering.
        project: Whether the rows of the singular vectors are projected onto the unit
            sphere before k-means. Defaults to True.
        leverage_threshold: The threshold ``gamma`` of thresholded spectral clustering
            (t-RSC): k-means only reads the rows of leverage at least
            ``gamma / sqrt(M)`` for the voters and ``gamma / sqrt(P)`` for the items,
            and the others take the nearest centroid. A finite number of at least 0.
            Defaults to 0, every row.
        n_init: The number of runs of k-means in each start, at least 1. Defaults to 10.

    Raises:
        InvalidParameterError: If a setting is out of range.
    """

    encoding: Encoding = "states"
    regularization: float = 1.0
    project: bool = True
    leverage_threshold: float = 0.0
    n_init: int = 10

    def __post_init__(self) -> None:
        if self.encoding not in ENCODINGS:
            raise InvalidParameterError(f"encoding is one of {ENCODINGS}, got {self.encoding!r}")
        for name in ("regularization", "leverage_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InvalidParameterError(
                    f"{name} is a finite number of at least 0, got {value!r}"
                )
            if not 0 <= value < math.inf:
                raise InvalidParameterError(
                    f"{name} is a finite number of at least 0, got {value!r}"
                )
        if not isinstance(self.project, bool):
            raise InvalidParameterError(f"project is True or False, got {self.project!r}")
        if isinstance(self.n_init, bool) or not isinstance(self.n_init, int) or self.n_init < 1:
            raise InvalidParameterError(f"n_init is an integer of at least 1, got {self.n_init!r}")

    def arrangement(
        self, votes: VoteMatrix, shape: tuple[int, int], rng: np.random.Generator
    ) -> tuple[IntArray, IntArray]:
        """Return the spectral arrangement, with the k-means seeded by ``rng``.

        Each side keeps as many singular vectors as it has clusters, or all of them
        when there are fewer.
        """
        n_groups, n_blocks = shape
        voter_values, item_values = n_singular_values(votes, self.encoding)
        embedding = spectral_embedding(
            votes,
            n_voter_components=min(n_groups, voter_values),
            n_item_components=min(n_blocks, item_values),
            encoding=self.encoding,
            regularization=self.regularization,
        )
        return embedding.labels(
            n_groups,
            n_blocks,
            rng,
            project=self.project,
            leverage_threshold=self.leverage_threshold,
            n_init=self.n_init,
        )


@dataclass(frozen=True)
class FixedStart:
    """An arrangement given by the caller, the same for every start.

    Use it to start a model where another computation stopped, for example a
    mixed-membership model from the groups and blocks of a single-membership one::

        single = LatentBlockModel(n_groups=3, n_blocks=2).estimate(votes).value
        mixed = MixedMembershipBlockModel(
            n_groups=3, n_blocks=2, initialization=FixedStart.from_coclustering(single)
        )

    The labels are settings: they enter the provenance, so a result says where it
    started from. Several starts from a fixed arrangement only differ by the starting
    parameters of the response model; one start is then usually enough.

    Attributes:
        group_index: The group of each voter, in the order of the matrix, from 0.
        block_index: The block of each item, in the order of the matrix, from 0.
        source: The fingerprint of the computation the arrangement comes from, if any.

    Raises:
        InvalidParameterError: If a label is not a non-negative integer.
    """

    group_index: tuple[int, ...]
    block_index: tuple[int, ...]
    source: str | None = None

    def __post_init__(self) -> None:
        for name in ("group_index", "block_index"):
            labels = tuple(getattr(self, name))
            if not all(_is_label(label) for label in labels):
                raise InvalidParameterError(f"{name} holds non-negative integers, got {labels!r}")
            object.__setattr__(self, name, tuple(int(label) for label in labels))

    @classmethod
    def from_coclustering(cls, coclustering: Any) -> "FixedStart":
        """Start from the main group of each voter and the main block of each item of a value.

        Args:
            coclustering: A :class:`~civiqhub_selene.blocks.results.Coclustering`, or
                any object with ``group_index``, ``block_index`` and ``computation``.
        """
        return cls(
            tuple(int(label) for label in coclustering.group_index),
            tuple(int(label) for label in coclustering.block_index),
            source=getattr(coclustering, "computation", None),
        )

    def arrangement(
        self, votes: VoteMatrix, shape: tuple[int, int], rng: np.random.Generator
    ) -> tuple[IntArray, IntArray]:
        """Return the arrangement given, after checking it fits the votes and the shape.

        Raises:
            InvalidInputError: If the labels do not follow the voters and the items of
                the matrix, are out of the shape, or leave a group or a block empty.
        """
        for name, labels, size, kind in (
            ("group_index", self.group_index, shape[0], "voters"),
            ("block_index", self.block_index, shape[1], "items"),
        ):
            expected = len(votes.voters) if kind == "voters" else len(votes.items)
            if len(labels) != expected:
                raise InvalidInputError(
                    f"{name} has {len(labels)} labels for the {expected} {kind} of the matrix"
                )
            counts = np.bincount(np.array(labels, dtype=np.int64), minlength=size)
            if counts.shape[0] > size or (counts == 0).any():
                raise InvalidInputError(
                    f"{name} must use every label from 0 to {size - 1} and no other, "
                    f"got the sizes {counts.tolist()}"
                )
        groups: IntArray = np.array(self.group_index, dtype=np.int64)
        blocks: IntArray = np.array(self.block_index, dtype=np.int64)
        return groups, blocks


def _is_label(value: object) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and int(value) >= 0
    )
