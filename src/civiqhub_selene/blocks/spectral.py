# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Regularised spectral co-clustering of a vote matrix: a fast arrangement to start from.

A block model only reaches a local optimum of its objective, and the arrangement it
starts from largely decides the one it ends with. This module computes an arrangement
from the singular vectors of a regularised graph Laplacian of the votes, in the way of
regularised spectral clustering (RSC, Qin and Rohe 2013) and of its co-clustering
version for directed and bipartite graphs (DI-SIM, Rohe, Qin and Yu 2016):

1. encode the votes as a matrix with one row per voter (or per item);
2. divide each entry by the square roots of the regularised degrees of its row and its
   column: the regularised graph Laplacian;
3. keep the leading singular vectors: one row per voter (or per item);
4. project each row onto the unit sphere, which removes the effect of participation;
5. run k-means on the rows, leaving out the rows of low leverage if asked, then give
   every row the centroid nearest to it.

:func:`spectral_embedding` computes steps 1 to 3 and returns a
:class:`SpectralEmbedding`; :meth:`SpectralEmbedding.labels` runs steps 4 and 5. The
start object :class:`~civiqhub_selene.blocks.starts.SpectralStart` hands the result to
a block model.
"""

import functools
import math
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np

from civiqhub_selene.blocks._arrays import FloatArray, IntArray
from civiqhub_selene.blocks._numerics import iterate
from civiqhub_selene.blocks.data import VoteMatrix
from civiqhub_selene.errors import InvalidInputError, InvalidParameterError
from civiqhub_selene.types import Displayable

__all__ = ["ENCODINGS", "Encoding", "SpectralEmbedding", "n_singular_values", "spectral_embedding"]

Encoding: TypeAlias = Literal["states", "scores"]
"""How the votes are written as a matrix before the decomposition."""

ENCODINGS: tuple[Encoding, ...] = ("states", "scores")
"""The encodings :func:`spectral_embedding` knows, the default first."""

_MAX_KMEANS_ITERATIONS = 100
_CACHE_SIZE = 4


@dataclass(frozen=True, kw_only=True, eq=False, repr=False)
class SpectralEmbedding(Displayable):
    """The leading singular vectors of the regularised Laplacian of a vote matrix.

    Row ``i`` of ``voter_vectors`` places the voter ``i`` in a space of
    ``n_voter_components`` dimensions; voters who answer alike point in the same
    direction. Under a degree-corrected block model, the rows of one group share a
    direction and differ in length only, and the rows of two groups are orthogonal
    (Qin and Rohe 2013, Lemma 3.3): projecting the rows onto the unit sphere gathers
    each group on one point. The same holds for the items.

    The arrays are copied and made read-only.

    See the mathematical model: :doc:`/model/initialization`, equation
    :eq:`eq-starts-laplacian`.

    Attributes:
        voters: The voters, in the order of the matrix.
        items: The items, in the order of the matrix.
        encoding: The encoding of the votes, ``"states"`` or ``"scores"``.
        regularization: The regulariser, as a multiple of the average degree of each
            side.
        voter_vectors: Shape (M, d): the leading left singular vectors ``U``, one row
            per voter.
        item_vectors: Shape (P, d'): the leading singular vectors ``V`` on the side of
            the items, one row per item.
        voter_singular_values: Every singular value ``delta_k`` of the Laplacian read
            on the side of the voters, in decreasing order. A gap after the ``k``-th one suggests
            ``k`` groups.
        item_singular_values: Every singular value on the side of the items, likewise.

    Raises:
        InvalidInputError: If the shapes of the arrays do not follow the voters and the
            items.
    """

    voters: tuple[str, ...]
    items: tuple[str, ...]
    encoding: str
    regularization: float
    voter_vectors: FloatArray
    item_vectors: FloatArray
    voter_singular_values: FloatArray
    item_singular_values: FloatArray

    def __post_init__(self) -> None:
        voters, items = tuple(self.voters), tuple(self.items)
        object.__setattr__(self, "voters", voters)
        object.__setattr__(self, "items", items)
        for name, rows in (("voter_vectors", len(voters)), ("item_vectors", len(items))):
            array = np.array(getattr(self, name), dtype=np.float64)
            if array.ndim != 2 or array.shape[0] != rows:
                raise InvalidInputError(f"{name} has one row per identifier, got {array.shape}")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for name in ("voter_singular_values", "item_singular_values"):
            values = np.array(getattr(self, name), dtype=np.float64)
            if values.ndim != 1:
                raise InvalidInputError(f"{name} is a vector, got the shape {values.shape}")
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def n_voter_components(self) -> int:
        """The number of dimensions of the space of the voters."""
        return int(self.voter_vectors.shape[1])

    @property
    def n_item_components(self) -> int:
        """The number of dimensions of the space of the items."""
        return int(self.item_vectors.shape[1])

    @property
    def voter_leverage(self) -> FloatArray:
        """Shape (M,): the length of the row of each voter, its leverage score.

        A voter with few shown cells has a short row: its direction is noisy.
        """
        return _row_norms(self.voter_vectors)

    @property
    def item_leverage(self) -> FloatArray:
        """Shape (P,): the length of the row of each item, its leverage score."""
        return _row_norms(self.item_vectors)

    def voter_directions(self) -> FloatArray:
        """Return the rows of the voters projected onto the unit sphere; a zero row stays 0."""
        return _project(self.voter_vectors)

    def item_directions(self) -> FloatArray:
        """Return the rows of the items projected onto the unit sphere; a zero row stays 0."""
        return _project(self.item_vectors)

    def labels(
        self,
        n_groups: int,
        n_blocks: int,
        rng: np.random.Generator,
        *,
        project: bool = True,
        leverage_threshold: float = 0.0,
        n_init: int = 10,
    ) -> tuple[IntArray, IntArray]:
        """Return a group for each voter and a block for each item, by k-means.

        On each side, with ``k`` clusters (``n_groups`` or ``n_blocks``):

        1. the first ``k`` singular vectors are kept (all of them when the embedding has
           fewer: DI-SIM also clusters ``k`` columns in fewer dimensions), and each row is
           projected onto the unit sphere when ``project`` is true;
        2. k-means runs on the rows whose leverage, before projection, is at least
           ``gamma / sqrt(M)`` for the voters and ``gamma / sqrt(P)`` for the items,
           with ``gamma`` the ``leverage_threshold``: the
           thresholded spectral clustering t-RSC of Qin and Rohe (2013) when the
           threshold is positive. When fewer than ``k`` rows pass, every row is kept;
        3. k-means is seeded by k-means++ and refined by Lloyd's algorithm, ``n_init``
           times; the run with the smallest within-cluster sum of squares is kept;
        4. every row takes the cluster of the nearest centroid. If a cluster is then
           empty, the row farthest from its centroid, in a cluster of more than one
           row, moves to it: no cluster is ever empty.

        Args:
            n_groups: The number of groups, from 1 to the number of voters.
            n_blocks: The number of blocks, from 1 to the number of items.
            rng: The generator of the seeding of k-means.
            project: Whether the rows are projected onto the unit sphere.
            leverage_threshold: The threshold ``gamma`` of t-RSC, at least 0. 0 keeps
                every row.
            n_init: The number of runs of k-means, at least 1.

        Returns:
            The group of each voter, shape (M,), and the block of each item, shape
            (P,), both integers.

        Raises:
            InvalidParameterError: If a setting is out of range.
        """
        _check_count("n_groups", n_groups, len(self.voters))
        _check_count("n_blocks", n_blocks, len(self.items))
        if not isinstance(project, bool):
            raise InvalidParameterError(f"project is True or False, got {project!r}")
        _check_number("leverage_threshold", leverage_threshold)
        if isinstance(n_init, bool) or not isinstance(n_init, int) or n_init < 1:
            raise InvalidParameterError(f"n_init is an integer of at least 1, got {n_init!r}")
        groups = _cluster_rows(
            self.voter_vectors, n_groups, rng, project, leverage_threshold, n_init
        )
        blocks = _cluster_rows(
            self.item_vectors, n_blocks, rng, project, leverage_threshold, n_init
        )
        return groups, blocks

    def _describe(self) -> list[str]:
        n_voters, n_items = len(self.voters), len(self.items)

        def leading(values: FloatArray) -> str:
            return ", ".join(f"{value:.3f}" for value in values[:5])

        return [
            f"SpectralEmbedding({n_voters} voters in {self.n_voter_components} dimensions x "
            f"{n_items} items in {self.n_item_components} dimensions, "
            f"encoding {self.encoding!r})",
            f"leading singular values of the voters: {leading(self.voter_singular_values)}",
            f"leading singular values of the items: {leading(self.item_singular_values)}",
        ]


def spectral_embedding(
    votes: VoteMatrix,
    *,
    n_voter_components: int = 2,
    n_item_components: int = 2,
    encoding: Encoding = "states",
    regularization: float = 1.0,
) -> SpectralEmbedding:
    """Return the leading singular vectors of the regularised Laplacian of the votes.

    Definition. For a matrix ``X`` with nonnegative entries, with ``D_r`` the diagonal
    matrix of its row degrees (the sums of its rows) and ``D_c`` that of its column
    degrees (the sums of its columns), the regularised graph Laplacian of DI-SIM (Rohe,
    Qin and Yu 2016, equation 1) is::

        X_tau = (D_r + tau_r I)^(-1/2) X (D_c + tau_c I)^(-1/2)

    where each regulariser is ``regularization`` times the average degree of its side,
    as Qin and Rohe (2013) recommend (``tau = 0`` is the standard Laplacian of Dhillon
    2001). A degree of 0 with ``tau = 0`` gives a row or a column of zeros.

    The encoding says what ``X`` is:

    ``"states"`` (default)
        Each shown cell is one of K + 1 states: shown without a vote, or a vote at one
        of the K levels. For the voters, ``X`` has one row per voter and one column per
        pair (item, state), with a 1 where the voter is in this state on this item; a
        cell that is not shown is a row of zeros. ``X Xᵀ`` then counts the items on
        which two voters gave the same answer. Under the latent block model, the
        expectation of ``X`` is the indicator matrix of the groups times a matrix of
        the co-clusters: its rank can reach the number of groups even when there are
        fewer blocks, and a group that splits where another agrees (same mean, larger
        spread) changes that matrix. The same is done on the transposed matrix for the
        items. The degree of a voter is its number of shown cells.
    ``"scores"``
        One matrix of centred scores: ``level - (K + 1) / 2`` on a voted cell, 0 on any
        other cell, so that a neutral vote, an abstention and a cell that is not shown
        all read 0. Its entries can be negative: the degrees are the numbers of votes
        of each row and column. The left singular vectors place the voters and the
        right ones the items, as in DI-SIM. Cheaper, but blind to a difference of
        spread at equal mean.

    Items on several scales. A level means another answer from one scale to the next,
    so both encodings first place the votes on one axis, the unit axis of
    :class:`~civiqhub_selene.blocks.responses.Thresholds`, where every scale covers the
    axis from 0 to 1 in equal shares, and K is the levels of the longest scale. Under
    ``"states"``, the vote at the level ``k`` of a scale of ``K_c`` levels covers the
    interval ``((k - 1) / K_c, k / K_c]`` of the axis, which is cut into K equal bins:
    the vote is spread over the bins it covers, in proportion to the length it covers
    in each, so that it still weighs 1; an abstention keeps its own state. Under
    ``"scores"``, the centred level of a vote is stretched by ``K / K_c``, the centre of
    its interval on the axis, measured in levels of the longest scale. With one scale,
    both are the encodings above, unchanged.

    See the mathematical model: :doc:`/model/initialization`, equation
    :eq:`eq-starts-laplacian`.

    Args:
        votes: The vote matrix.
        n_voter_components: The number of singular vectors kept for the voters, at
            least 1 and at most the number of singular values on that side.
        n_item_components: The same for the items.
        encoding: ``"states"`` or ``"scores"``.
        regularization: The regulariser as a multiple of the average degree, a finite
            number of at least 0. Defaults to 1, the average degree.

    Returns:
        The embedding. Every singular value of each side is kept, for a scree plot.

    Raises:
        InvalidInputError: If ``votes`` is not a ``VoteMatrix``, or has no voter or no
            item.
        InvalidParameterError: If a setting is out of range.
    """
    if not isinstance(votes, VoteMatrix):
        raise InvalidInputError(f"votes are a VoteMatrix, got {type(votes).__name__}")
    if encoding not in ENCODINGS:
        raise InvalidParameterError(f"encoding is one of {ENCODINGS}, got {encoding!r}")
    _check_number("regularization", regularization)
    n_voters, n_items = votes.shape
    if n_voters == 0 or n_items == 0:
        raise InvalidInputError(f"a spectral embedding needs voters and items, got {votes!r}")
    voter_basis, voter_values, item_basis, item_values = _decomposition(
        votes, encoding, float(regularization)
    )
    _check_count("n_voter_components", n_voter_components, voter_values.shape[0])
    _check_count("n_item_components", n_item_components, item_values.shape[0])
    return SpectralEmbedding(
        voters=votes.voters,
        items=votes.items,
        encoding=encoding,
        regularization=float(regularization),
        voter_vectors=voter_basis[:, :n_voter_components],
        item_vectors=item_basis[:, :n_item_components],
        voter_singular_values=voter_values,
        item_singular_values=item_values,
    )


def n_singular_values(votes: VoteMatrix, encoding: Encoding = "states") -> tuple[int, int]:
    """Return how many singular values each side of the embedding of ``votes`` has.

    With M voters, P items and K levels: ``min(M, P * (K + 1))`` for the voters and
    ``min(P, M * (K + 1))`` for the items under ``"states"``; ``min(M, P)`` on both
    sides under ``"scores"``. These are the largest numbers of components
    :func:`spectral_embedding` accepts. Nothing is decomposed.
    """
    n_voters, n_items = votes.shape
    if encoding == "scores":
        return min(n_voters, n_items), min(n_voters, n_items)
    n_states = votes.n_levels + 1
    return min(n_voters, n_items * n_states), min(n_items, n_voters * n_states)


# Decomposition --------------------------------------------------------------------------


@functools.lru_cache(maxsize=_CACHE_SIZE)
def _decomposition(
    votes: VoteMatrix, encoding: str, regularization: float
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return the singular vectors and values of both sides, computed once per input.

    A vote matrix is hashed and compared by its fingerprint, so two matrices of the
    same content share an entry. The arrays returned are read-only.
    """
    if encoding == "states":
        states = _aligned_states(votes)
        n_voters, n_items, n_states = states.shape
        voter_basis, voter_values = _left_vectors(
            _laplacian(states.reshape(n_voters, n_items * n_states), regularization)
        )
        item_basis, item_values = _left_vectors(
            _laplacian(
                states.transpose(1, 0, 2).reshape(n_items, n_voters * n_states), regularization
            )
        )
    else:
        voted = votes.voted
        scales = votes.item_scales
        if scales.is_uniform:
            middle = (votes.n_levels + 1) / 2
            centred = np.where(voted, votes.levels - middle, 0.0)
        else:
            levels = scales.levels_by_item
            stretch = votes.n_levels / levels
            centred = np.where(voted, (votes.levels - (levels + 1) / 2) * stretch, 0.0)
        laplacian = _laplacian(centred, regularization, degrees=voted.astype(np.float64))
        left, values, right_t = np.linalg.svd(laplacian, full_matrices=False)
        voter_basis, item_basis = left, right_t.T
        voter_values = item_values = values
    arrays = (voter_basis, voter_values, item_basis, item_values)
    for array in arrays:
        array.setflags(write=False)
    return (
        np.asarray(voter_basis),
        np.asarray(voter_values),
        np.asarray(item_basis),
        np.asarray(item_values),
    )


def _aligned_states(votes: VoteMatrix) -> FloatArray:
    """Return the states of each cell, read on the unit axis: shape (M, P, K + 1).

    With one scale, the states of the matrix. With several, state 0 stays the
    abstention, and the vote at the level ``k`` of a scale of ``K_c`` levels is spread
    over the K bins of the axis that its interval ``((k - 1) / K_c, k / K_c]`` covers,
    each in proportion to the length it covers, times ``K_c``: the weights of a vote
    sum to 1.
    """
    scales = votes.item_scales
    states = votes._states()
    if scales.is_uniform:
        return states
    n_levels = votes.n_levels
    weights = np.zeros((scales.n_scales, n_levels + 1, n_levels + 1))
    for position, scale in enumerate(scales.scales):
        weights[position, 0, 0] = 1.0
        own = scale.n_levels
        for level in range(1, own + 1):
            for bin_ in range(1, n_levels + 1):
                low = max((level - 1) / own, (bin_ - 1) / n_levels)
                high = min(level / own, bin_ / n_levels)
                weights[position, level, bin_] = max(high - low, 0.0) * own
    aligned: FloatArray = np.einsum("ijs,jst->ijt", states, weights[scales.scale_index])
    return aligned


def _laplacian(
    matrix: FloatArray, regularization: float, degrees: FloatArray | None = None
) -> FloatArray:
    """Return ``(O + tau_r I)^(-1/2) A (P + tau_c I)^(-1/2)``.

    The degrees are the sums of ``degrees`` (``matrix`` itself by default) along each
    axis; each regulariser is ``regularization`` times the mean degree of its side.
    """
    weights = matrix if degrees is None else degrees
    row_degrees = weights.sum(axis=1)
    column_degrees = weights.sum(axis=0)
    rows = _inverse_root(row_degrees + regularization * row_degrees.mean())
    columns = _inverse_root(column_degrees + regularization * column_degrees.mean())
    laplacian: FloatArray = rows[:, np.newaxis] * matrix * columns[np.newaxis, :]
    return laplacian


def _inverse_root(degrees: FloatArray) -> FloatArray:
    """Return ``1 / sqrt(d)`` where ``d > 0`` and 0 elsewhere."""
    safe = np.where(degrees > 0, degrees, 1.0)
    inverse: FloatArray = np.where(degrees > 0, 1.0 / np.sqrt(safe), 0.0)
    return inverse


def _left_vectors(matrix: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return the left singular vectors and the singular values, in decreasing order."""
    left, values, _ = np.linalg.svd(matrix, full_matrices=False)
    return left, values


# Clustering of the rows -------------------------------------------------------------------


def _row_norms(vectors: FloatArray) -> FloatArray:
    norms: FloatArray = np.sqrt(np.einsum("ij,ij->i", vectors, vectors))
    return norms


def _project(vectors: FloatArray) -> FloatArray:
    norms = _row_norms(vectors)
    safe = np.where(norms > 0, norms, 1.0)
    directions: FloatArray = vectors / safe[:, np.newaxis]
    return directions


def _squared_distances(points: FloatArray, centroids: FloatArray) -> FloatArray:
    """Return the squared Euclidean distance of each point to each centroid, (n, k)."""
    distances: FloatArray = np.maximum(
        np.einsum("ij,ij->i", points, points)[:, np.newaxis]
        - 2.0 * points @ centroids.T
        + np.einsum("kj,kj->k", centroids, centroids)[np.newaxis, :],
        0.0,
    )
    return distances


def _seed_centroids(points: FloatArray, n_clusters: int, rng: np.random.Generator) -> FloatArray:
    """Pick ``n_clusters`` starting centroids among the points by k-means++.

    The first is drawn uniformly; each next one with a probability proportional to the
    squared distance to the nearest centroid already picked, or uniformly when every
    point lies on a centroid.
    """
    n_points = points.shape[0]
    chosen = [int(rng.integers(n_points))]
    nearest = _squared_distances(points, points[chosen])[:, 0]
    for _ in range(1, n_clusters):
        total = float(nearest.sum())
        if total > 0:
            picked = int(rng.choice(n_points, p=nearest / total))
        else:
            picked = int(rng.integers(n_points))
        chosen.append(picked)
        nearest = np.minimum(nearest, _squared_distances(points, points[[picked]])[:, 0])
    centroids: FloatArray = points[chosen].copy()
    return centroids


def _lloyd(points: FloatArray, centroids: FloatArray) -> tuple[FloatArray, IntArray, float]:
    """Refine centroids by Lloyd's algorithm; return them, the labels and the inertia.

    An empty cluster keeps its centroid. The loop stops when no label changes.
    """
    n_clusters = centroids.shape[0]

    def assign(current: FloatArray) -> IntArray:
        labels: IntArray = _squared_distances(points, current).argmin(axis=1).astype(np.int64)
        return labels

    def step(state: tuple[FloatArray, IntArray]) -> tuple[FloatArray, IntArray]:
        current, labels = state
        sums = np.zeros_like(current)
        np.add.at(sums, labels, points)
        sizes = np.bincount(labels, minlength=n_clusters)
        updated: FloatArray = np.where(
            sizes[:, np.newaxis] > 0, sums / np.maximum(sizes, 1)[:, np.newaxis], current
        )
        return updated, assign(updated)

    (fitted, labels), _ = iterate(
        step,
        (centroids, assign(centroids)),
        gap=lambda old, new: float((old[1] != new[1]).sum()),
        value=lambda state: 0.0,
        tolerance=0.5,
        max_iterations=_MAX_KMEANS_ITERATIONS,
    )
    inertia = float(_squared_distances(points, fitted)[np.arange(points.shape[0]), labels].sum())
    return fitted, labels, inertia


def _fill_empty(labels: IntArray, distances: FloatArray) -> IntArray:
    """Give each empty cluster the row farthest from its centroid, in a larger cluster."""
    n_clusters = distances.shape[1]
    filled: IntArray = labels.copy()
    own = distances[np.arange(labels.shape[0]), labels]
    for cluster in range(n_clusters):
        sizes = np.bincount(filled, minlength=n_clusters)
        if sizes[cluster] > 0:
            continue
        movable = sizes[filled] > 1
        candidates = np.flatnonzero(movable)
        # The farthest row, the first one on a tie.
        row = int(candidates[np.argmax(own[candidates])])
        filled[row] = cluster
        own[row] = distances[row, cluster]
    return filled


def _cluster_rows(
    vectors: FloatArray,
    n_clusters: int,
    rng: np.random.Generator,
    project: bool,
    leverage_threshold: float,
    n_init: int,
) -> IntArray:
    """Steps 4 and 5 of the module docstring, on one side."""
    vectors = vectors[:, :n_clusters]
    n_rows = vectors.shape[0]
    if n_clusters == 1:
        return np.zeros(n_rows, dtype=np.int64)
    points = _project(vectors) if project else np.asarray(vectors, dtype=np.float64)
    kept = _row_norms(vectors) >= leverage_threshold / math.sqrt(n_rows)
    if int(kept.sum()) < n_clusters:
        kept = np.ones(n_rows, dtype=np.bool_)
    selected = points[kept]
    best: tuple[float, FloatArray] | None = None
    for _ in range(n_init):
        centroids, _, inertia = _lloyd(selected, _seed_centroids(selected, n_clusters, rng))
        if best is None or inertia < best[0]:
            best = (inertia, centroids)
    assert best is not None, "n_init is at least 1"
    distances = _squared_distances(points, best[1])
    labels: IntArray = distances.argmin(axis=1).astype(np.int64)
    return _fill_empty(labels, distances)


# Checks ---------------------------------------------------------------------------------


def _check_number(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < math.inf:
        raise InvalidParameterError(f"{name} is a finite number of at least 0, got {value!r}")


def _check_count(name: str, value: Any, *limits: int) -> None:
    largest = min(limits)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= largest:
        raise InvalidParameterError(f"{name} is an integer from 1 to {largest}, got {value!r}")
