# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Co-clustering results as NetworkX graphs.

Each function reads a value of :mod:`civiqhub_selene.blocks` and returns a graph whose
nodes and edges carry the numbers of the value as attributes. The graphs are data: draw
them with :mod:`civiqhub_selene.viz.draw`, export them (GraphML, GEXF, JSON) for another
tool, or compute on them with NetworkX. Nothing here draws or computes a new estimate.

A node is a tuple whose first element says its kind:

========================  ==============================================================
Node                      Meaning
========================  ==============================================================
``("group", q)``          The group ``q`` of voters
``("block", l)``          The block ``l`` of items
``("voter", voter)``      A voter, by its identifier
``("item", item)``        An item, by its identifier
``("role", l, q)``        The voters who follow the group ``q`` on the block ``l``
========================  ==============================================================

Every node has the attributes ``kind`` (the first element of the tuple) and ``label``
(a short text to print).
"""

from collections.abc import Hashable
from typing import Any, Literal, TypeAlias

import networkx
import numpy as np

from civiqhub_selene.blocks.data import VoteMatrix
from civiqhub_selene.blocks.results import Coclustering, MixedCoclustering
from civiqhub_selene.errors import InvalidInputError, InvalidParameterError

__all__ = [
    "Node",
    "coclustering_graph",
    "group_graph",
    "membership_graph",
    "role_graph",
    "vote_graph",
]

Node: TypeAlias = tuple[Hashable, ...]
"""A node: a tuple whose first element is its kind."""


def _check_value(value: object, kind: type = Coclustering) -> None:
    if not isinstance(value, kind):
        raise InvalidInputError(
            f"a graph of co-clustering reads a {kind.__name__}, got {type(value).__name__}"
        )


def _mean_scores(value: Coclustering) -> Any:
    """Return the mean score of the votes of each co-cluster, shape (Q, L)."""
    table = value.coclusters
    return np.einsum("qlk,k->ql", table.vote_shares, np.array(table.scale.scores, dtype=float))


def _group_nodes(value: Coclustering) -> list[tuple[Node, dict[str, Any]]]:
    return [
        (
            ("group", group.index),
            {
                "kind": "group",
                "index": group.index,
                "label": f"G{group.index}",
                "share": float(group.share),
                "size": len(group),
            },
        )
        for group in value.groups
    ]


def _block_nodes(value: Coclustering) -> list[tuple[Node, dict[str, Any]]]:
    return [
        (
            ("block", block.index),
            {
                "kind": "block",
                "index": block.index,
                "label": f"B{block.index}",
                "share": float(block.share),
                "size": len(block),
            },
        )
        for block in value.blocks
    ]


def coclustering_graph(value: Coclustering) -> "networkx.Graph[Node]":
    """Return the bipartite graph of the groups and the blocks, one edge per co-cluster.

    Nodes ``("group", q)`` and ``("block", l)`` carry ``index``, ``label``, ``share``
    (the share of the voters or items in the cluster) and ``size`` (the number of
    voters or items whose main cluster it is). Groups have the attribute
    ``bipartite=0`` and blocks ``bipartite=1``, as NetworkX expects.

    The edge between ``("group", q)`` and ``("block", l)`` carries what the table of
    co-clusters says of ``(q, l)``: ``vote_rate``, ``n_votes``, ``typical_score``,
    ``mean_score`` (the mean of the scores weighted by ``vote_shares``),
    ``vote_shares`` (a tuple), and ``latent_mean`` and ``latent_spread`` under the
    ordinal probit response model (``None`` otherwise).

    Raises:
        InvalidInputError: If ``value`` is not a ``Coclustering``.
    """
    _check_value(value)
    graph: networkx.Graph[Node] = networkx.Graph(kind="coclustering", computation=value.computation)
    graph.add_nodes_from((node, {**data, "bipartite": 0}) for node, data in _group_nodes(value))
    graph.add_nodes_from((node, {**data, "bipartite": 1}) for node, data in _block_nodes(value))
    means = _mean_scores(value)
    for cocluster in value.coclusters:
        graph.add_edge(
            ("group", cocluster.group),
            ("block", cocluster.block),
            vote_rate=float(cocluster.vote_rate),
            n_votes=float(cocluster.n_votes),
            typical_score=int(cocluster.typical_score),
            mean_score=float(means[cocluster.group, cocluster.block]),
            vote_shares=tuple(float(share) for share in cocluster.vote_shares),
            latent_mean=cocluster.latent_mean,
            latent_spread=cocluster.latent_spread,
        )
    return graph


def membership_graph(
    value: Coclustering,
    *,
    side: Literal["voters", "items"] = "voters",
    threshold: float = 0.05,
) -> "networkx.Graph[Node]":
    """Return the graph that links each voter (or item) to the clusters it belongs to.

    With ``side="voters"``, the nodes are the groups and the voters; an edge links a
    voter to a group when its degree of membership is at least ``threshold``, with the
    attribute ``weight`` (the membership). A voter node carries ``main`` (its main
    group), ``mixedness`` (one minus its largest membership) and ``memberships`` (a
    tuple). Under a mixed membership, a voter tied to two groups bridges them. With
    ``side="items"``, the same for the blocks and the items.

    Args:
        value: A co-clustering.
        side: ``"voters"`` or ``"items"``.
        threshold: The smallest membership drawn as an edge, from 0 to 1. A voter
            always keeps the edge to its main group.

    Raises:
        InvalidInputError: If ``value`` is not a ``Coclustering``.
        InvalidParameterError: If ``side`` or ``threshold`` is out of range.
    """
    _check_value(value)
    if side not in ("voters", "items"):
        raise InvalidParameterError(f"side is 'voters' or 'items', got {side!r}")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise InvalidParameterError(f"threshold is a number from 0 to 1, got {threshold!r}")
    if not 0 <= threshold <= 1:
        raise InvalidParameterError(f"threshold is a number from 0 to 1, got {threshold!r}")
    if side == "voters":
        clusters, kind, member = _group_nodes(value), "group", "voter"
        ids, index, memberships = value.voters, value.group_index, value.group_memberships
    else:
        clusters, kind, member = _block_nodes(value), "block", "item"
        ids, index, memberships = value.items, value.block_index, value.block_memberships
    graph: networkx.Graph[Node] = networkx.Graph(
        kind="membership", side=side, computation=value.computation
    )
    graph.add_nodes_from(clusters)
    for position, identifier in enumerate(ids):
        row = memberships[position]
        main = int(index[position])
        graph.add_node(
            (member, identifier),
            kind=member,
            label=identifier,
            main=main,
            mixedness=float(1.0 - row.max()) if row.size else 0.0,
            memberships=tuple(float(weight) for weight in row),
        )
        for cluster, weight in enumerate(row):
            if weight >= threshold or cluster == main:
                graph.add_edge((member, identifier), (kind, cluster), weight=float(weight))
    return graph


def role_graph(value: MixedCoclustering) -> "networkx.DiGraph[Node]":
    """Return how the voters change role from one block to the next, as a layered graph.

    The node ``("role", l, q)`` stands for the voters whose most probable role on the
    block ``l`` is the group ``q``; it carries ``block``, ``group``, ``count`` (their
    number), ``label`` and ``layer`` (the block, for ``multipartite_layout``). An edge
    goes from ``("role", l, q)`` to ``("role", l + 1, r)`` when some voters follow
    ``q`` on ``l`` and ``r`` on ``l + 1``, with the attribute ``count``; it is
    ``MixedCoclustering.role_changes(l, l + 1)[q, r]``. An edge between two different
    groups is a change of role; ``changes_role`` says so.

    Raises:
        InvalidInputError: If ``value`` is not a ``MixedCoclustering``.
    """
    _check_value(value, MixedCoclustering)
    roles = value.role_by_block.argmax(axis=2)
    graph: networkx.DiGraph[Node] = networkx.DiGraph(kind="roles", computation=value.computation)
    for block in range(value.n_blocks):
        counts = np.bincount(roles[:, block], minlength=value.n_groups)
        for group in range(value.n_groups):
            graph.add_node(
                ("role", block, group),
                kind="role",
                block=block,
                group=group,
                layer=block,
                count=int(counts[group]),
                label=f"G{group}",
            )
    for block in range(value.n_blocks - 1):
        changes = value.role_changes(block, block + 1)
        for group in range(value.n_groups):
            for other in range(value.n_groups):
                count = int(changes[group, other])
                if count > 0:
                    graph.add_edge(
                        ("role", block, group),
                        ("role", block + 1, other),
                        count=count,
                        changes_role=group != other,
                    )
    return graph


def group_graph(value: Coclustering) -> "networkx.Graph[Node]":
    """Return how much each pair of groups agrees, as a complete graph on the groups.

    Definition. For two groups ``q`` and ``r``, the agreement is one minus the total
    variation distance between their distributions of votes on a block, averaged over
    the blocks with the share of each block as its weight::

        agreement(q, r) = 1 - sum_l share_l * 0.5 * sum_k |shares[q, l, k] - shares[r, l, k]|

    It reads the distributions of the votes only, not the vote rates: two groups that
    vote alike but not as often agree fully. It is 1 for identical opinions and 0 for
    opinions without any common score.

    Each edge carries ``agreement`` and ``by_block`` (the agreement on each block, a
    tuple); each node carries the attributes of the groups of
    :func:`coclustering_graph`.

    Raises:
        InvalidInputError: If ``value`` is not a ``Coclustering``.
    """
    _check_value(value)
    graph: networkx.Graph[Node] = networkx.Graph(kind="groups", computation=value.computation)
    graph.add_nodes_from(_group_nodes(value))
    shares = value.coclusters.vote_shares
    weights = np.array([block.share for block in value.blocks], dtype=float)
    total = weights.sum()
    weights = weights / total if total > 0 else np.full(value.n_blocks, 1.0 / value.n_blocks)
    for group in range(value.n_groups):
        for other in range(group + 1, value.n_groups):
            by_block = 1.0 - 0.5 * np.abs(shares[group] - shares[other]).sum(axis=-1)
            graph.add_edge(
                ("group", group),
                ("group", other),
                agreement=float((weights * by_block).sum()),
                by_block=tuple(float(agreement) for agreement in by_block),
            )
    return graph


def vote_graph(votes: VoteMatrix, value: Coclustering | None = None) -> "networkx.Graph[Node]":
    """Return the bipartite graph of the voters and the items, one edge per vote.

    The edge between ``("voter", voter)`` and ``("item", item)`` carries ``score``
    (the score of the vote) and ``level`` (its rank on the scale). A cell shown without
    a vote, or not shown, has no edge. Voters have ``bipartite=0`` and items
    ``bipartite=1``. With a co-clustering of the same matrix, each voter node also
    carries ``group`` (its main group) and each item node ``block``.

    Raises:
        InvalidInputError: If ``votes`` is not a ``VoteMatrix``, or if ``value`` was
            not computed on the voters and items of ``votes``.
    """
    if not isinstance(votes, VoteMatrix):
        raise InvalidInputError(f"votes are a VoteMatrix, got {type(votes).__name__}")
    if value is not None:
        _check_value(value)
        if value.voters != votes.voters or value.items != votes.items:
            raise InvalidInputError(
                "the co-clustering was computed on other voters or items than these votes"
            )
    graph: networkx.Graph[Node] = networkx.Graph(kind="votes", fingerprint=votes.fingerprint())
    for position, voter in enumerate(votes.voters):
        data: dict[str, Any] = {"kind": "voter", "label": voter, "bipartite": 0}
        if value is not None:
            data["group"] = int(value.group_index[position])
        graph.add_node(("voter", voter), **data)
    for position, item in enumerate(votes.items):
        data = {"kind": "item", "label": item, "bipartite": 1}
        if value is not None:
            data["block"] = int(value.block_index[position])
        graph.add_node(("item", item), **data)
    for row, column in np.argwhere(votes.voted):
        level = int(votes.levels[row, column])
        graph.add_edge(
            ("voter", votes.voters[row]),
            ("item", votes.items[column]),
            level=level,
            score=votes.scale.score_of(level),
        )
    return graph
