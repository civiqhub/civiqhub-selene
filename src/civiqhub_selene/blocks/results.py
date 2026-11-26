# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""What a co-clustering returns.

:class:`Coclustering` carries what every model returns: the groups of voters, the
blocks of items, the table of the co-clusters, and what it takes to judge the
computation. :class:`SingleCoclustering` and :class:`MixedCoclustering` add what is
proper to a single and to a mixed membership; their names take the two words of the
``membership`` choice.

Two protocols, checkable at run time, say what a value can do: a criterion relies on
them, not on the name of a class.
"""

import hashlib
import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from civiqhub_selene.blocks._arrays import FloatArray, IntArray
from civiqhub_selene.blocks._numerics import History
from civiqhub_selene.blocks.coclusters import Cocluster, CoclusterTable
from civiqhub_selene.blocks.data import OrdinalScale, VoteMatrix
from civiqhub_selene.blocks.partition import ItemBlock, Partition, VoterGroup
from civiqhub_selene.errors import InvalidInputError
from civiqhub_selene.types import Displayable

__all__ = [
    "Coclustering",
    "HasIntegratedLikelihood",
    "MixedCoclustering",
    "PredictsVotes",
    "Selection",
    "SingleCoclustering",
]

_MIN_PROBABILITY = 1e-300


@runtime_checkable
class HasIntegratedLikelihood(Protocol):
    """A value that carries an integrated likelihood, for the criterion of that name.

    See the mathematical model: :doc:`/model/choosing-the-shape`, equation
    :eq:`eq-shape-icl`.
    """

    @property
    def integrated_likelihood(self) -> float:
        """The integrated completed likelihood; larger is better."""
        ...


@runtime_checkable
class PredictsVotes(Protocol):
    """A value that gives the probability of the state of each cell of a vote matrix.

    See the mathematical model: :doc:`/model/choosing-the-shape`, equations
    :eq:`eq-shape-held-out-single` and :eq:`eq-shape-held-out-mixed`.
    """

    def log_probability(self, votes: VoteMatrix) -> FloatArray:
        """Return, for each cell, the log-probability of its state, of shape (M, P)."""
        ...


def _array(received: ArrayLike, dtype: type[Any], name: str, shape: tuple[int, ...]) -> Any:
    """Copy an array, check its shape and make it read-only."""
    try:
        array: NDArray[Any] = np.array(received, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise InvalidInputError(f"{name} is an array of numbers: {error}") from error
    if array.shape != shape:
        raise InvalidInputError(f"{name} has the shape {shape}, got {array.shape}")
    array.setflags(write=False)
    return array


def _plain(value: object) -> Any:
    """Turn arrays and floats into plain JSON-ready data; a non-finite float becomes None."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _float_bytes(array: FloatArray) -> bytes:
    return np.ascontiguousarray(array, dtype="<f8").tobytes()


def _plural(count: int) -> str:
    return "" if count == 1 else "s"


@dataclass(frozen=True, kw_only=True, eq=False, repr=False)
class Selection(Displayable):
    """The trace of a choice of shape.

    Attributes:
        criterion: Key of the criterion that compared the shapes.
        n_groups: The numbers of groups compared.
        n_blocks: The numbers of blocks compared.
        scores: Shape (len(n_groups), len(n_blocks)): the score of each shape; larger
            is better.
        standard_errors: The standard error of each score, when the criterion gives one.
        best: The shape kept, ``(n_groups, n_blocks)``.
        alternatives: The shapes within one standard error of the best one.

    Raises:
        InvalidInputError: If the scores do not have the shape of the grid, or if the
            best shape is not on the grid.
    """

    criterion: str
    n_groups: tuple[int, ...]
    n_blocks: tuple[int, ...]
    scores: FloatArray
    standard_errors: FloatArray | None = None
    best: tuple[int, int]
    alternatives: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        n_groups = tuple(int(size) for size in self.n_groups)
        n_blocks = tuple(int(size) for size in self.n_blocks)
        grid = (len(n_groups), len(n_blocks))
        object.__setattr__(self, "n_groups", n_groups)
        object.__setattr__(self, "n_blocks", n_blocks)
        object.__setattr__(self, "scores", _array(self.scores, np.float64, "scores", grid))
        if self.standard_errors is not None:
            object.__setattr__(
                self,
                "standard_errors",
                _array(self.standard_errors, np.float64, "standard_errors", grid),
            )
        best = (int(self.best[0]), int(self.best[1]))
        if best[0] not in n_groups or best[1] not in n_blocks:
            raise InvalidInputError(
                f"the best shape {best} is not among the shapes compared: "
                f"groups {n_groups}, blocks {n_blocks}"
            )
        object.__setattr__(self, "best", best)
        object.__setattr__(
            self, "alternatives", tuple((int(q), int(b)) for q, b in self.alternatives)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return plain lists; a score that is not finite becomes ``None``."""
        return {
            "criterion": self.criterion,
            "n_groups": list(self.n_groups),
            "n_blocks": list(self.n_blocks),
            "scores": _plain(self.scores),
            "standard_errors": _plain(self.standard_errors),
            "best": list(self.best),
            "alternatives": [list(shape) for shape in self.alternatives],
        }

    def _describe(self) -> list[str]:
        lines = [
            f"Selection({self.criterion}: {self.best[0]} group{_plural(self.best[0])} x "
            f"{self.best[1]} block{_plural(self.best[1])} among "
            f"{len(self.n_groups) * len(self.n_blocks)} shapes)"
        ]
        lines.append("groups \\ blocks" + "".join(f"{blocks:>14d}" for blocks in self.n_blocks))
        for row, groups in enumerate(self.n_groups):
            cells = ""
            for column, blocks in enumerate(self.n_blocks):
                mark = "*" if (groups, blocks) == self.best else " "
                cells += f"{float(self.scores[row, column]):>13.2f}{mark}"
            lines.append(f"{groups:<15d}{cells}")
        return lines


@dataclass(frozen=True, kw_only=True, eq=False, repr=False)
class Coclustering(Displayable):
    """Groups of voters, blocks of items and their co-clusters, as a model found them.

    ``coclustering[q, l]`` is the co-cluster of the group ``q`` and the block ``l``.
    Arrays are copied and made read-only.

    See the mathematical model: :doc:`/model/reading-results`.

    Attributes:
        computation: Fingerprint of the computation that produced the value.
        voters: The voters, in the order of the matrix.
        items: The items, in the order of the matrix.
        scale: The scale of scores.
        group_index: Shape (M,): for each voter, the index of its main group.
        block_index: Shape (P,): for each item, the index of its main block.
        group_memberships: Shape (M, Q): degree of membership of each voter to each
            group, rows summing to 1.
        block_memberships: Shape (P, L): the same for items and blocks.
        coclusters: The table of the Q x L co-clusters.
        objective: The quantity the algorithm maximised. It only compares computations
            of the same method on the same data.
        history: The objective after each iteration of the start that was kept, and
            whether it converged.
        selection: The trace of the choice of shape, when the shape was chosen by
            comparison.

    Raises:
        InvalidInputError: If the shapes of the arrays do not agree with the voters,
            the items and the table.
    """

    computation: str
    voters: tuple[str, ...]
    items: tuple[str, ...]
    scale: OrdinalScale
    group_index: IntArray
    block_index: IntArray
    group_memberships: FloatArray
    block_memberships: FloatArray
    coclusters: CoclusterTable
    objective: float
    history: History
    selection: Selection | None = None

    def __post_init__(self) -> None:
        voters, items = tuple(self.voters), tuple(self.items)
        n_voters, n_items = len(voters), len(items)
        if not isinstance(self.coclusters, CoclusterTable):
            raise InvalidInputError(f"coclusters are a CoclusterTable, got {self.coclusters!r}")
        n_groups, n_blocks = self.coclusters.shape
        if self.coclusters.scale != self.scale:
            raise InvalidInputError(
                f"the table of co-clusters is on the scale {self.coclusters.scale.scores!r}, "
                f"the value on the scale {self.scale.scores!r}"
            )
        object.__setattr__(self, "voters", voters)
        object.__setattr__(self, "items", items)
        for name, dtype, shape in (
            ("group_index", np.int64, (n_voters,)),
            ("block_index", np.int64, (n_items,)),
            ("group_memberships", np.float64, (n_voters, n_groups)),
            ("block_memberships", np.float64, (n_items, n_blocks)),
        ):
            object.__setattr__(self, name, _array(getattr(self, name), dtype, name, shape))
        for name, index, size in (
            ("group_index", self.group_index, n_groups),
            ("block_index", self.block_index, n_blocks),
        ):
            if ((index < 0) | (index >= size)).any():
                raise InvalidInputError(f"{name} holds indices from 0 to {size - 1}")
        object.__setattr__(self, "objective", float(self.objective))

    # Shape and clusters ---------------------------------------------------------------

    @property
    def n_groups(self) -> int:
        """The number of groups of voters."""
        return int(self.group_memberships.shape[1])

    @property
    def n_blocks(self) -> int:
        """The number of blocks of items."""
        return int(self.block_memberships.shape[1])

    def _shares(self, index: IntArray, memberships: FloatArray) -> FloatArray:
        """Return the share of each cluster: its size over the number of identifiers."""
        n_clusters = memberships.shape[1]
        if index.shape[0] == 0:
            return np.zeros(n_clusters)
        shares: FloatArray = np.bincount(index, minlength=n_clusters) / index.shape[0]
        return shares

    @cached_property
    def groups(self) -> Partition[VoterGroup]:
        """The groups of voters, as a partition that cites this computation."""
        return Partition.from_voter_labels(
            self.voters,
            self.group_index,
            n_clusters=self.n_groups,
            memberships=self.group_memberships,
            shares=self._shares(self.group_index, self.group_memberships),
            computation=self.computation,
        )

    @cached_property
    def blocks(self) -> Partition[ItemBlock]:
        """The blocks of items, as a partition that cites this computation."""
        return Partition.from_item_labels(
            self.items,
            self.block_index,
            n_clusters=self.n_blocks,
            memberships=self.block_memberships,
            shares=self._shares(self.block_index, self.block_memberships),
            computation=self.computation,
        )

    def group_of(self, voter: str) -> VoterGroup:
        """Return the main group of a voter.

        Raises:
            InvalidInputError: If the voter is not in the co-clustering.
        """
        return self.groups.cluster_of(voter)

    def block_of(self, item: str) -> ItemBlock:
        """Return the main block of an item.

        Raises:
            InvalidInputError: If the item is not in the co-clustering.
        """
        return self.blocks.cluster_of(item)

    def __getitem__(self, key: tuple[int, int]) -> Cocluster:
        """Return the co-cluster of the group ``q`` and the block ``l``."""
        return self.coclusters[key]

    def display_order(self) -> tuple[IntArray, IntArray]:
        """Return the order of voters and of items that sorts the matrix by groups and blocks.

        The sort is stable: the order inside a group, or inside a block, is kept.
        """
        voters: IntArray = np.argsort(self.group_index, kind="stable").astype(np.int64)
        items: IntArray = np.argsort(self.block_index, kind="stable").astype(np.int64)
        return voters, items

    # Identity and export --------------------------------------------------------------

    def fingerprint(self) -> str:
        """Return a stable fingerprint of what was found: 16 hexadecimal characters.

        SHA-256 of the voters, the items and the scores, then the bytes of the indices,
        the memberships, and the vote rates and vote shares of the table. The history,
        the objective and the trace of the choice of shape are left out.
        """
        return self._fingerprint

    @cached_property
    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(repr((self.voters, self.items, self.scale.scores)).encode())
        digest.update(np.ascontiguousarray(self.group_index, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(self.block_index, dtype="<i8").tobytes())
        for array in (
            self.group_memberships,
            self.block_memberships,
            self.coclusters.vote_rate,
            self.coclusters.vote_shares,
        ):
            digest.update(_float_bytes(array))
        return digest.hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Return plain lists and numbers, ready for JSON.

        A float that is not finite becomes ``None``. The fields proper to a subclass
        are added by it.
        """
        return {
            "kind": type(self).__name__,
            "voters": list(self.voters),
            "items": list(self.items),
            "scores": list(self.scale.scores),
            "group_index": self.group_index.tolist(),
            "block_index": self.block_index.tolist(),
            "group_memberships": _plain(self.group_memberships),
            "block_memberships": _plain(self.block_memberships),
            "coclusters": _plain_dict(self.coclusters.to_dict()),
            "objective": _plain(self.objective),
            "history": {
                "objectives": _plain(self.history.objectives),
                "converged": self.history.converged,
            },
            "selection": None if self.selection is None else self.selection.to_dict(),
            "computation": self.computation,
            "fingerprint": self.fingerprint(),
        }

    def _describe(self) -> list[str]:
        n_voters, n_items = len(self.voters), len(self.items)
        n_groups, n_blocks = self.n_groups, self.n_blocks
        group_sizes = ", ".join(str(size) for size in self.groups.sizes)
        block_sizes = ", ".join(str(size) for size in self.blocks.sizes)
        return [
            f"{type(self).__name__}({n_voters} voter{_plural(n_voters)} in {n_groups} "
            f"group{_plural(n_groups)} x {n_items} item{_plural(n_items)} in {n_blocks} "
            f"block{_plural(n_blocks)})",
            f"groups of {group_sizes} voters; blocks of {block_sizes} items",
            *self.coclusters._describe()[1:],
        ]

    def _check_votes(self, votes: VoteMatrix) -> None:
        if not isinstance(votes, VoteMatrix):
            raise InvalidInputError(f"votes are a VoteMatrix, got {type(votes).__name__}")
        if votes.voters != self.voters or votes.items != self.items or votes.scale != self.scale:
            raise InvalidInputError(
                "log_probability reads a matrix of the same voters, items and scale, in the "
                f"same order, as the one the co-clustering was computed on: got {votes!r}"
            )


def _plain_dict(data: dict[str, Any]) -> dict[str, Any]:
    return {key: _plain(value) for key, value in data.items()}


@dataclass(frozen=True, kw_only=True, eq=False, repr=False)
class SingleCoclustering(Coclustering):
    """A co-clustering where each voter belongs to one group and each item to one block.

    Its memberships are the soft reading ``theta_hat[i, q]`` and ``phi_hat[j, l]`` of
    the final arrangement, and its objective the classified log-likelihood.

    See the mathematical model: :doc:`/model/latent-block-model`, equations
    :eq:`eq-lbm-objective` and :eq:`eq-lbm-memberships`; :doc:`/model/choosing-the-shape`,
    equation :eq:`eq-shape-icl`.

    Attributes:
        integrated_likelihood: The integrated completed likelihood criterion (ICL,
            Biernacki, Celeux and Govaert 2000); larger is better.
    """

    integrated_likelihood: float

    def log_probability(self, votes: VoteMatrix) -> FloatArray:
        """Return, for each cell, the log-probability of its state under the model.

        Shape (M, P). The state of a cell is its vote or its absence of vote, read
        without ``shown``; its probability is the one of the co-cluster of its voter
        and of its item, from the table of co-clusters::

            log F[R[i], C[j], s[i, j]]

        See the mathematical model: :doc:`/model/choosing-the-shape`, equation
        :eq:`eq-shape-held-out-single`.

        Raises:
            InvalidInputError: If ``votes`` does not have the voters, the items and the
                scale of the co-clustering.
        """
        self._check_votes(votes)
        logs = self.coclusters.log_probabilities()
        cells: FloatArray = logs[
            self.group_index[:, np.newaxis], self.block_index[np.newaxis, :], votes.levels
        ]
        return cells

    def to_dict(self) -> dict[str, Any]:
        """Return the common fields and the integrated likelihood."""
        data = super().to_dict()
        data["integrated_likelihood"] = _plain(float(self.integrated_likelihood))
        return data


@dataclass(frozen=True, kw_only=True, eq=False, repr=False)
class MixedCoclustering(Coclustering):
    """A co-clustering where a voter can follow different groups on different blocks.

    A profile is one choice of a group per block; with Q groups and L blocks there are
    Q**L of them. Its memberships are ``theta[i, q]`` and ``phi[j, l]``, and its
    objective the log-likelihood plus the log-densities of the priors.

    See the mathematical model: :doc:`/model/mixed-membership`, equations
    :eq:`eq-mm-likelihood` to :eq:`eq-mm-profiles`.

    Attributes:
        role_by_block: Shape (M, L, Q): for each voter and each block, the probability
            ``kappa[i, l, q]`` that the voter follows each group on this block.
        profiles: Shape (Q**L, L), integers: every profile ``c``, the group followed on
            each block, in the order of ``itertools.product(range(Q), repeat=L)``.
        profile_probabilities: Shape (M, Q**L): for each voter, the probability
            ``eta[i, c]`` of each profile.
        log_likelihood: The observed log-likelihood.
    """

    role_by_block: FloatArray
    profiles: IntArray
    profile_probabilities: FloatArray
    log_likelihood: float

    def __post_init__(self) -> None:
        super().__post_init__()
        n_voters = len(self.voters)
        n_groups, n_blocks = self.n_groups, self.n_blocks
        n_profiles = n_groups**n_blocks
        for name, dtype, shape in (
            ("role_by_block", np.float64, (n_voters, n_blocks, n_groups)),
            ("profiles", np.int64, (n_profiles, n_blocks)),
            ("profile_probabilities", np.float64, (n_voters, n_profiles)),
        ):
            object.__setattr__(self, name, _array(getattr(self, name), dtype, name, shape))
        if ((self.profiles < 0) | (self.profiles >= n_groups)).any():
            raise InvalidInputError(f"profiles hold groups from 0 to {n_groups - 1}")

    def _shares(self, index: IntArray, memberships: FloatArray) -> FloatArray:
        """Return the share of each cluster: the mean of the memberships."""
        if memberships.shape[0] == 0:
            return np.zeros(memberships.shape[1])
        shares: FloatArray = memberships.mean(axis=0)
        return shares

    @property
    def mixedness(self) -> FloatArray:
        """Shape (M,): one minus the largest membership of each voter.

        ``1 - max_q theta[i, q]``: 0 for a voter who belongs to a single group. It is
        published as a histogram over the voters, never as a list by voter.

        See the mathematical model: :doc:`/model/reading-results`, equation
        :eq:`eq-results-mixedness`.
        """
        if self.group_memberships.shape[1] == 0:
            return np.zeros(self.group_memberships.shape[0])
        mixedness: FloatArray = 1.0 - self.group_memberships.max(axis=1)
        return mixedness

    def role_changes(self, block_a: int = 0, block_b: int = 1) -> IntArray:
        """Count the voters by the group they follow on two blocks.

        Returns:
            Integers of shape (Q, Q): entry ``[q, r]`` is the number of voters whose
            most probable group is ``q`` on ``block_a`` and ``r`` on ``block_b``, by
            ``role_by_block``. Off the diagonal, a voter changes role.

        Raises:
            InvalidInputError: If a block does not exist.
        """
        n_blocks = self.n_blocks
        for block in (block_a, block_b):
            if isinstance(block, bool) or not isinstance(block, int) or not 0 <= block < n_blocks:
                raise InvalidInputError(f"there are {n_blocks} blocks, got the block {block!r}")
        roles = self.role_by_block.argmax(axis=2)
        changes: IntArray = np.zeros((self.n_groups, self.n_groups), dtype=np.int64)
        np.add.at(changes, (roles[:, block_a], roles[:, block_b]), 1)
        return changes

    def log_probability(self, votes: VoteMatrix) -> FloatArray:
        """Return, for each cell, the log-probability of its state under the model.

        Shape (M, P). For the cell of the voter ``i`` and the item ``j``, in the state
        ``s`` (its vote or its absence of vote, read without ``shown``)::

            log( sum_c eta[i, c] * sum_l phi[j, l] * F[c_l, l, s] )

        where ``c`` runs over the profiles, ``c_l`` is the group of the profile on the
        block ``l``, ``eta`` is ``profile_probabilities``, ``phi`` is
        ``block_memberships`` and ``F`` the probabilities of the table of co-clusters.
        The probability is floored at 1e-300 before the logarithm.

        See the mathematical model: :doc:`/model/choosing-the-shape`, equation
        :eq:`eq-shape-held-out-mixed`.

        Raises:
            InvalidInputError: If ``votes`` does not have the voters, the items and the
                scale of the co-clustering.
        """
        self._check_votes(votes)
        probabilities = np.exp(self.coclusters.log_probabilities())
        blocks = np.arange(self.n_blocks)
        by_profile = probabilities[self.profiles, blocks[np.newaxis, :], :]
        by_item = np.einsum("jl,cls->cjs", self.block_memberships, by_profile)
        by_cell = np.einsum("ic,cjs->ijs", self.profile_probabilities, by_item)
        cells = np.take_along_axis(by_cell, votes.levels[..., np.newaxis], axis=2)[..., 0]
        logs: FloatArray = np.log(np.maximum(cells, _MIN_PROBABILITY))
        return logs

    def to_dict(self) -> dict[str, Any]:
        """Return the common fields, the roles, the profiles and the log-likelihood."""
        data = super().to_dict()
        data["role_by_block"] = _plain(self.role_by_block)
        data["profiles"] = self.profiles.tolist()
        data["profile_probabilities"] = _plain(self.profile_probabilities)
        data["log_likelihood"] = _plain(float(self.log_likelihood))
        return data
