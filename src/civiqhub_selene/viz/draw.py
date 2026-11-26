# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Figures of co-clustering results, drawn with Matplotlib and NetworkX.

Each function draws on the axes it receives, or on new axes of a new figure, and
returns them, so that a figure can be composed, titled again and saved as any
Matplotlib figure::

    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    draw_coclustering(value, ax=left)
    plot_vote_matrix(votes, value, ax=right)
    figure.savefig("coclustering.png")

The graphs come from :mod:`civiqhub_selene.viz.graphs`; the colours from
:mod:`civiqhub_selene.viz.style`. A function never changes the global settings of
Matplotlib and never shows a figure.
"""

import math
from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
from numpy.typing import ArrayLike

from civiqhub_selene.blocks.data import VoteMatrix
from civiqhub_selene.blocks.results import Coclustering, MixedCoclustering
from civiqhub_selene.blocks.spectral import SpectralEmbedding
from civiqhub_selene.errors import InvalidInputError, InvalidParameterError
from civiqhub_selene.viz.graphs import (
    Node,
    coclustering_graph,
    group_graph,
    membership_graph,
    role_graph,
)
from civiqhub_selene.viz.style import (
    BLOCK_COLOR,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    NOT_SHOWN,
    SURFACE,
    group_color,
    score_colors,
)

__all__ = [
    "draw_coclustering",
    "draw_groups",
    "draw_memberships",
    "draw_roles",
    "plot_embedding",
    "plot_history",
    "plot_mixedness",
    "plot_spectrum",
    "plot_vote_matrix",
]

_TITLE_SIZE = 11.0
_LABEL_SIZE = 9.0
_BLOCK_MARKERS = ("s", "D", "^", "v", "P", "X", "h", "*")


# Shared pieces ------------------------------------------------------------------------


def _axes(ax: Axes | None, size: tuple[float, float]) -> Axes:
    """Return ``ax``, or the axes of a new figure of ``size`` inches."""
    if ax is not None:
        return ax
    figure, created = plt.subplots(figsize=size, layout="constrained")
    figure.patch.set_facecolor(SURFACE)
    return created


def _tidy(ax: Axes, *, grid: bool = True) -> None:
    """Recessive chrome: no top or right spine, muted ticks, a hairline grid."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(NOT_SHOWN)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=_LABEL_SIZE)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    if grid:
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def _bare(ax: Axes) -> None:
    """Axes for a graph: no frame, no ticks."""
    ax.set_facecolor(SURFACE)
    ax.set_axis_off()


def _title(ax: Axes, text: str, pad: float = 6.0) -> None:
    ax.set_title(text, loc="left", color=INK, fontsize=_TITLE_SIZE, pad=pad)


def _text_on(color: str) -> str:
    """Ink for a label written on a fill: black on light fills, white elsewhere."""
    red, green, blue, _ = to_rgba(color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return INK if luminance > 0.5 else "#ffffff"


def _score_map(scores: Sequence[int]) -> tuple[LinearSegmentedColormap, TwoSlopeNorm]:
    """The diverging colour map of the scores and its norm, centred on the middle."""
    colors = score_colors(len(scores))
    cmap = LinearSegmentedColormap.from_list("civiqhub_scores", list(colors))
    low, high = float(scores[0]), float(scores[-1])
    return cmap, TwoSlopeNorm(vmin=low, vcenter=(low + high) / 2, vmax=high)


def _check(value: object, kind: type = Coclustering) -> None:
    if not isinstance(value, kind):
        raise InvalidInputError(f"this figure reads a {kind.__name__}, got {type(value).__name__}")


def _node_sizes(shares: Sequence[float], smallest: float, largest: float) -> list[float]:
    return [smallest + (largest - smallest) * float(share) for share in shares]


# Graphs -----------------------------------------------------------------------------


def draw_coclustering(value: Coclustering, *, ax: Axes | None = None) -> Axes:
    """Draw the groups, the blocks and one edge per co-cluster.

    Groups are circles on the left, in their colour; blocks are squares on the right,
    in ink. The area of a node follows the share of its cluster. The edge of a
    co-cluster is coloured by the mean score of its votes, on the diverging Likert
    ramp (orange for disagreement, blue for agreement), and its width follows its vote
    rate. Reads :func:`~civiqhub_selene.viz.graphs.coclustering_graph`.

    Args:
        value: A co-clustering.
        ax: The axes to draw on. Defaults to new axes.

    Returns:
        The axes.
    """
    _check(value)
    graph = coclustering_graph(value)
    ax = _axes(ax, (7.0, 4.6))
    _bare(ax)
    groups = [node for node, data in graph.nodes(data=True) if data["kind"] == "group"]
    blocks = [node for node, data in graph.nodes(data=True) if data["kind"] == "block"]

    def column(nodes: list[Node], x: float) -> dict[Node, tuple[float, float]]:
        count = len(nodes)
        return {node: (x, (count - 1) / 2 - rank) for rank, node in enumerate(nodes)}

    pos = {**column(groups, 0.0), **column(blocks, 1.0)}
    cmap, norm = _score_map(value.scale.scores)
    edges = list(graph.edges(data=True))
    widths = [1.0 + 9.0 * float(data["vote_rate"]) for _, _, data in edges]
    edge_list = [(u, v) for u, v, _ in edges]
    # A grey under-stroke keeps an edge of neutral score visible on the surface.
    nx.draw_networkx_edges(
        graph, pos, edgelist=edge_list, width=[w + 1.4 for w in widths], edge_color=NOT_SHOWN, ax=ax
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=edge_list,
        width=widths,
        edge_color=[cmap(norm(float(data["mean_score"]))) for _, _, data in edges],
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=groups,
        node_size=_node_sizes([graph.nodes[n]["share"] for n in groups], 250, 2200),
        node_color=[group_color(int(graph.nodes[n]["index"])) for n in groups],
        edgecolors=SURFACE,
        linewidths=2.0,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=blocks,
        node_shape="s",
        node_size=_node_sizes([graph.nodes[n]["share"] for n in blocks], 250, 2200),
        node_color=BLOCK_COLOR,
        edgecolors=SURFACE,
        linewidths=2.0,
        ax=ax,
    )
    for node in groups + blocks:
        data = graph.nodes[node]
        x, y = pos[node]
        left = data["kind"] == "group"
        ax.annotate(
            f"{data['label']}  {data['share']:.0%}",
            (x, y),
            xytext=(-26 if left else 26, 0),
            textcoords="offset points",
            ha="right" if left else "left",
            va="center",
            color=INK,
            fontsize=_LABEL_SIZE + 1,
        )
    ax.set_xlim(-0.45, 1.45)
    span = max(len(groups), len(blocks))
    ax.set_ylim(-span / 2 - 0.1, span / 2 + 0.1)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    bar = ax.figure.colorbar(mappable, ax=ax, shrink=0.6, pad=0.02, aspect=25)
    bar.set_label("mean score of the votes", color=INK_SECONDARY, fontsize=_LABEL_SIZE)
    bar.set_ticks([float(score) for score in value.scale.scores])
    bar.ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=_LABEL_SIZE)
    bar.outline.set_visible(False)
    ax.legend(
        handles=[
            Line2D([], [], color=INK_MUTED, linewidth=1.0 + 9.0 * rate, label=f"{rate:.0%}")
            for rate in (0.2, 0.5, 0.9)
        ],
        title="vote rate",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncols=3,
        frameon=False,
        fontsize=_LABEL_SIZE,
        title_fontsize=_LABEL_SIZE,
        labelcolor=INK_SECONDARY,
    )
    _title(ax, "Groups (circles), blocks (squares), co-clusters (edges)")
    return ax


def _polygon(count: int) -> list[tuple[float, float]]:
    """The vertices of a regular polygon of radius 1, the first one at the top."""
    if count == 1:
        return [(0.0, 0.0)]
    if count == 2:
        return [(-1.0, 0.0), (1.0, 0.0)]
    return [
        (math.sin(2 * math.pi * k / count), math.cos(2 * math.pi * k / count)) for k in range(count)
    ]


def draw_memberships(
    value: Coclustering,
    *,
    side: Literal["voters", "items"] = "voters",
    threshold: float = 0.2,
    ax: Axes | None = None,
) -> Axes:
    """Place each voter (or item) between the clusters it belongs to.

    The clusters sit at the vertices of a regular polygon; each voter sits at the mean
    of the vertices weighted by its memberships, and is linked to each cluster it
    belongs to by at least ``threshold``. A voter of a single group sits on its vertex;
    a voter shared between two groups sits on the side between them. Voters are
    coloured by their main group; a small spiral keeps voters with the same
    memberships apart. Each cluster is a translucent disk at its vertex, named outside
    the polygon. Reads :func:`~civiqhub_selene.viz.graphs.membership_graph`.

    Args:
        value: A co-clustering.
        side: ``"voters"`` (groups) or ``"items"`` (blocks).
        threshold: The smallest membership drawn as an edge.
        ax: The axes to draw on. Defaults to new axes.

    Returns:
        The axes.
    """
    _check(value)
    graph = membership_graph(value, side=side, threshold=threshold)
    kind = "group" if side == "voters" else "block"
    member = "voter" if side == "voters" else "item"
    clusters = sorted(
        (node for node, data in graph.nodes(data=True) if data["kind"] == kind),
        key=lambda node: int(graph.nodes[node]["index"]),
    )
    members = [node for node, data in graph.nodes(data=True) if data["kind"] == member]
    vertices = np.array(_polygon(len(clusters)))
    pos: dict[Node, tuple[float, float]] = {
        node: (float(vertices[rank, 0]), float(vertices[rank, 1]))
        for rank, node in enumerate(clusters)
    }
    golden = math.pi * (3 - math.sqrt(5))
    for rank, node in enumerate(members):
        weights = np.array(graph.nodes[node]["memberships"])
        x, y = weights @ vertices
        radius = 0.11 * math.sqrt((rank % 60) / 60)
        pos[node] = (
            float(x + radius * math.cos(rank * golden)),
            float(y + radius * math.sin(rank * golden)),
        )
    ax = _axes(ax, (5.6, 5.0))
    _bare(ax)
    edges = list(graph.edges(data=True))
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=[(u, v) for u, v, _ in edges],
        width=0.6,
        edge_color=[
            to_rgba(INK_MUTED, 0.08 + 0.35 * float(data["weight"])) for _, _, data in edges
        ],
        ax=ax,
    )

    def tint(index: int) -> str:
        return group_color(index) if side == "voters" else BLOCK_COLOR

    fills = [tint(int(graph.nodes[node]["index"])) for node in clusters]
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=clusters,
        node_size=1500,
        node_color=[to_rgba(fill, 0.16) for fill in fills],
        linewidths=0,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=members,
        node_size=22,
        node_color=[tint(int(graph.nodes[node]["main"])) for node in members],
        edgecolors=SURFACE,
        linewidths=0.5,
        ax=ax,
    )
    for node in clusters:
        x, y = pos[node]
        norm = math.hypot(x, y) or 1.0
        away = (x / norm, y / norm) if len(clusters) > 1 else (0.0, 1.0)
        ax.annotate(
            graph.nodes[node]["label"],
            (x, y),
            xytext=(26 * away[0], 26 * away[1]),
            textcoords="offset points",
            ha="center",
            va="center",
            color=INK,
            fontsize=_LABEL_SIZE + 2,
            fontweight="bold",
        )
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4 if len(clusters) > 2 else -0.6, 1.4 if len(clusters) > 2 else 0.6)
    ax.set_aspect("equal")
    _title(ax, f"{len(members)} {member}s placed by their memberships")
    return ax


def draw_roles(value: MixedCoclustering, *, ax: Axes | None = None) -> Axes:
    """Draw the group each voter follows on each block, and the changes of role.

    One column per block; in each column, one node per group, of area following the
    number of voters whose most probable role on that block is that group. A band goes
    from a group on a block to a group on the next block, as wide as the number of
    voters who make that move, in the colour of the group they leave. A band between
    two different groups is a change of role. Reads
    :func:`~civiqhub_selene.viz.graphs.role_graph`.

    Args:
        value: A mixed co-clustering.
        ax: The axes to draw on. Defaults to new axes.

    Returns:
        The axes.
    """
    _check(value, MixedCoclustering)
    graph = role_graph(value)
    n_groups, n_blocks = value.n_groups, value.n_blocks
    pos: dict[Node, tuple[float, float]] = {
        node: (float(data["block"]), (n_groups - 1) / 2 - float(data["group"]))
        for node, data in graph.nodes(data=True)
    }
    ax = _axes(ax, (2.2 + 2.2 * n_blocks, 1.4 + 0.9 * n_groups))
    _bare(ax)
    n_voters = max(len(value.voters), 1)
    edges = list(graph.edges(data=True))
    if edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=[(u, v) for u, v, _ in edges],
            width=[1.0 + 22.0 * int(data["count"]) / n_voters for _, _, data in edges],
            edge_color=[
                to_rgba(group_color(int(graph.nodes[u]["group"])), 0.45) for u, _, _ in edges
            ],
            arrows=False,
            ax=ax,
        )
    nodes: list[tuple[Node, dict[str, Any]]] = [(node, graph.nodes[node]) for node in graph.nodes]
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=[node for node, _ in nodes],
        node_size=[60 + 1600 * int(data["count"]) / n_voters for _, data in nodes],
        node_color=[group_color(int(data["group"])) for _, data in nodes],
        edgecolors=SURFACE,
        linewidths=2.0,
        ax=ax,
    )
    for node, data in nodes:
        x, y = pos[node]
        ax.annotate(
            f"G{data['group']}  {data['count']}",
            (x, y),
            xytext=(0, -17 - 9 * math.sqrt(int(data["count"]) / n_voters)),
            textcoords="offset points",
            ha="center",
            va="top",
            color=INK,
            fontsize=_LABEL_SIZE,
        )
    for block in range(n_blocks):
        ax.text(
            block,
            n_groups / 2 + 0.15,
            f"block {block}",
            ha="center",
            va="bottom",
            color=INK_SECONDARY,
            fontsize=_LABEL_SIZE + 1,
        )
    changed = sum(int(data["count"]) for _, _, data in edges if bool(data["changes_role"]))
    ax.set_xlim(-0.5, n_blocks - 0.5)
    ax.set_ylim(-n_groups / 2 - 0.35, n_groups / 2 + 0.45)
    _title(ax, f"Roles from block to block: {changed} changes of role")
    return ax


def draw_groups(value: Coclustering, *, ax: Axes | None = None) -> Axes:
    """Draw how much each pair of groups agrees.

    Groups sit on a circle; the edge between two groups is as thick and as dark as
    their agreement (one minus the total variation distance between their votes,
    averaged over the blocks), and carries its value. Reads
    :func:`~civiqhub_selene.viz.graphs.group_graph`.

    See the mathematical model: :doc:`/model/reading-results`, equation
    :eq:`eq-results-agreement`.

    Args:
        value: A co-clustering.
        ax: The axes to draw on. Defaults to new axes.

    Returns:
        The axes.
    """
    _check(value)
    graph = group_graph(value)
    nodes = sorted(graph.nodes, key=lambda node: int(graph.nodes[node]["index"]))
    vertices = _polygon(len(nodes))
    pos: dict[Node, tuple[float, float]] = dict(zip(nodes, vertices, strict=True))
    ax = _axes(ax, (5.0, 4.4))
    _bare(ax)
    edges = list(graph.edges(data=True))
    if edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=[(u, v) for u, v, _ in edges],
            width=[0.5 + 9.0 * float(data["agreement"]) for _, _, data in edges],
            edge_color=[
                to_rgba(INK_SECONDARY, 0.15 + 0.75 * float(data["agreement"]))
                for _, _, data in edges
            ],
            ax=ax,
        )
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels={(u, v): f"{float(data['agreement']):.2f}" for u, v, data in edges},
            font_size=_LABEL_SIZE,
            font_color=INK,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": SURFACE, "edgecolor": "none"},
            ax=ax,
        )
    fills = [group_color(int(graph.nodes[node]["index"])) for node in nodes]
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=nodes,
        node_size=_node_sizes([graph.nodes[n]["share"] for n in nodes], 500, 2400),
        node_color=fills,
        edgecolors=SURFACE,
        linewidths=2.0,
        ax=ax,
    )
    for node, fill in zip(nodes, fills, strict=True):
        x, y = pos[node]
        ax.text(
            x,
            y,
            graph.nodes[node]["label"],
            ha="center",
            va="center",
            color=_text_on(fill),
            fontsize=_LABEL_SIZE + 1,
            fontweight="bold",
        )
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45 if len(nodes) > 2 else -0.7, 1.45 if len(nodes) > 2 else 0.7)
    ax.set_aspect("equal")
    _title(ax, "Agreement between groups")
    return ax


# Matrices and curves ------------------------------------------------------------------


def plot_vote_matrix(
    votes: VoteMatrix, value: Coclustering | None = None, *, ax: Axes | None = None
) -> Axes:
    """Draw the vote matrix, sorted by groups and blocks when a co-clustering is given.

    One row per voter, one column per item. A vote takes the colour of its score on
    the Likert ramp; a cell shown without a vote is a grey dot on the surface; a cell
    that is not shown is hatched. With a co-clustering of the same matrix, rows and
    columns follow :meth:`~civiqhub_selene.blocks.results.Coclustering.display_order`,
    and lines separate the groups and the blocks, named in the margins.

    Args:
        votes: The vote matrix.
        value: A co-clustering computed on ``votes``, or ``None`` for the order of the
            matrix.
        ax: The axes to draw on. Defaults to new axes.

    Returns:
        The axes.

    Raises:
        InvalidInputError: If ``value`` was not computed on the voters and items of
            ``votes``.
    """
    if not isinstance(votes, VoteMatrix):
        raise InvalidInputError(f"votes are a VoteMatrix, got {type(votes).__name__}")
    n_voters, n_items = votes.shape
    if value is not None:
        _check(value)
        if value.voters != votes.voters or value.items != votes.items:
            raise InvalidInputError(
                "the co-clustering was computed on other voters or items than these votes"
            )
        rows, columns = value.display_order()
    else:
        rows, columns = np.arange(n_voters), np.arange(n_items)
    levels = votes.levels[np.ix_(rows, columns)]
    shown = votes.shown[np.ix_(rows, columns)]
    palette = np.array([to_rgba(color) for color in score_colors(votes.n_levels)])
    image = np.zeros((n_voters, n_items, 4))
    voted = shown & (levels > 0)
    image[voted] = palette[levels[voted] - 1]
    image[shown & ~voted] = to_rgba(SURFACE)
    ax = _axes(ax, (6.4, 6.0))
    ax.set_facecolor(SURFACE)
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            n_items,
            n_voters,
            facecolor=SURFACE,
            edgecolor=NOT_SHOWN,
            hatch="////",
            linewidth=0,
            zorder=0,
        )
    )
    ax.imshow(image, aspect="auto", interpolation="nearest", zorder=1)
    silent_rows, silent_columns = np.nonzero(shown & ~voted)
    ax.scatter(silent_columns, silent_rows, s=2.5, color=INK_MUTED, linewidths=0, zorder=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(NOT_SHOWN)
    if value is not None:
        group_sizes = np.bincount(value.group_index, minlength=value.n_groups)
        block_sizes = np.bincount(value.block_index, minlength=value.n_blocks)
        for edge in np.cumsum(group_sizes)[:-1]:
            ax.axhline(edge - 0.5, color=INK, linewidth=1.2, zorder=3)
        for edge in np.cumsum(block_sizes)[:-1]:
            ax.axvline(edge - 0.5, color=INK, linewidth=1.2, zorder=3)
        for group, (start, size) in enumerate(
            zip(np.cumsum(group_sizes) - group_sizes, group_sizes, strict=True)
        ):
            if size:
                ax.text(
                    -0.8,
                    start + size / 2 - 0.5,
                    f"G{group}",
                    ha="right",
                    va="center",
                    color=INK,
                    fontsize=_LABEL_SIZE,
                )
        for block, (start, size) in enumerate(
            zip(np.cumsum(block_sizes) - block_sizes, block_sizes, strict=True)
        ):
            if size:
                ax.text(
                    start + size / 2 - 0.5,
                    -0.9,
                    f"B{block}",
                    ha="center",
                    va="bottom",
                    color=INK,
                    fontsize=_LABEL_SIZE,
                )
    ax.set_xlabel(
        "items" + (", by block" if value is not None else ""),
        color=INK_SECONDARY,
        fontsize=_LABEL_SIZE,
    )
    ax.set_ylabel(
        "voters" + (", by group" if value is not None else ""),
        color=INK_SECONDARY,
        fontsize=_LABEL_SIZE,
        labelpad=24 if value is not None else 4,
    )
    handles: list[Any] = [
        Patch(facecolor=color, label=votes.scale.format_score(score))
        for color, score in zip(palette, votes.scale.scores, strict=True)
    ]
    handles.append(
        Line2D([], [], linestyle="none", marker="o", markersize=3, color=INK_MUTED, label="no vote")
    )
    handles.append(Patch(facecolor=SURFACE, edgecolor=NOT_SHOWN, hatch="////", label="not shown"))
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncols=len(handles),
        frameon=False,
        fontsize=_LABEL_SIZE,
        labelcolor=INK_SECONDARY,
        handlelength=1.2,
        columnspacing=1.0,
    )
    _title(ax, f"Votes of {n_voters} voters on {n_items} items", pad=20 if value is not None else 6)
    return ax


def plot_spectrum(
    embedding: SpectralEmbedding, *, n_values: int = 10, ax: Axes | None = None
) -> Axes:
    """Draw the leading singular values of both sides of a spectral embedding.

    A gap after the ``k``-th value suggests ``k`` clusters on that side; it is a hint
    for the shape, not a choice of it (the integrated likelihood or the held-out
    likelihood choose).

    See the mathematical model: :doc:`/model/initialization`;
    :doc:`/model/choosing-the-shape`.

    Args:
        embedding: The spectral embedding.
        n_values: The number of values drawn on each side, at least 1.
        ax: The axes to draw on. Defaults to new axes.

    Returns:
        The axes.
    """
    if not isinstance(embedding, SpectralEmbedding):
        raise InvalidInputError(
            f"this figure reads a SpectralEmbedding, got {type(embedding).__name__}"
        )
    if isinstance(n_values, bool) or not isinstance(n_values, int) or n_values < 1:
        raise InvalidParameterError(f"n_values is an integer of at least 1, got {n_values!r}")
    ax = _axes(ax, (6.0, 3.6))
    _tidy(ax)
    for name, values, color, marker, style in (
        ("voters", embedding.voter_singular_values, INK, "o", "-"),
        ("items", embedding.item_singular_values, INK_MUTED, "s", "--"),
    ):
        shown = values[:n_values]
        ranks = np.arange(1, shown.shape[0] + 1)
        ax.plot(
            ranks,
            shown,
            color=color,
            linestyle=style,
            linewidth=2,
            marker=marker,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=name,
        )
        if shown.size:
            ax.annotate(
                name,
                (ranks[-1], shown[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                color=INK_SECONDARY,
                fontsize=_LABEL_SIZE,
            )
    ax.set_xlabel("rank of the singular value")
    ax.set_ylabel("singular value")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=_LABEL_SIZE, labelcolor=INK_SECONDARY)
    _title(ax, f"Singular values, encoding {embedding.encoding!r}")
    return ax


def plot_embedding(
    embedding: SpectralEmbedding,
    labels: ArrayLike | None = None,
    *,
    side: Literal["voters", "items"] = "voters",
    project: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Draw the rows of a spectral embedding on its first two dimensions.

    Before projection (``project=False``), the rows of one cluster share a direction
    and spread along it with their participation: the "star" of Qin and Rohe (2013,
    figure 1). After projection onto the unit sphere (``project=True``), each cluster
    gathers on one point. The hollow circle is the origin.

    See the mathematical model: :doc:`/model/initialization`, equation
    :eq:`eq-starts-laplacian`.

    Args:
        embedding: The spectral embedding, with at least two components on that side.
        labels: The cluster of each row (for example the groups found), to colour the
            points; ``None`` draws them in ink.
        side: ``"voters"`` or ``"items"``.
        project: Whether the rows are projected onto the unit sphere first.
        ax: The axes to draw on. Defaults to new axes.

    Returns:
        The axes.

    Raises:
        InvalidParameterError: If the side has fewer than two components, or if the
            labels do not have one entry per row.
    """
    if not isinstance(embedding, SpectralEmbedding):
        raise InvalidInputError(
            f"this figure reads a SpectralEmbedding, got {type(embedding).__name__}"
        )
    if side not in ("voters", "items"):
        raise InvalidParameterError(f"side is 'voters' or 'items', got {side!r}")
    if side == "voters":
        points = embedding.voter_directions() if project else embedding.voter_vectors
    else:
        points = embedding.item_directions() if project else embedding.item_vectors
    if points.shape[1] < 2:
        raise InvalidParameterError(
            f"the {side} have {points.shape[1]} component; this figure needs two"
        )
    ax = _axes(ax, (4.8, 4.4))
    _tidy(ax)
    ax.axhline(0, color=NOT_SHOWN, linewidth=0.8, zorder=1)
    ax.axvline(0, color=NOT_SHOWN, linewidth=0.8, zorder=1)
    if labels is None:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=26,
            color=INK,
            edgecolors=SURFACE,
            linewidths=0.6,
            zorder=2,
        )
    else:
        label_array = np.asarray(labels, dtype=np.int64)
        if label_array.shape != (points.shape[0],):
            raise InvalidParameterError(
                f"labels have one entry per {side[:-1]}: expected {points.shape[0]}, "
                f"got the shape {label_array.shape}"
            )
        for cluster in np.unique(label_array):
            chosen = label_array == cluster
            if side == "voters":
                color, marker, name = group_color(int(cluster)), "o", f"G{cluster}"
            else:
                # Blocks are drawn in ink: their shape tells them apart.
                color, marker, name = BLOCK_COLOR, _BLOCK_MARKERS[int(cluster) % 8], f"B{cluster}"
            ax.scatter(
                points[chosen, 0],
                points[chosen, 1],
                s=26,
                color=color,
                marker=marker,
                edgecolors=SURFACE,
                linewidths=0.6,
                zorder=2,
                label=name,
            )
        ax.legend(frameon=False, fontsize=_LABEL_SIZE, labelcolor=INK_SECONDARY, loc="best")
    ax.scatter([0], [0], s=60, facecolors="none", edgecolors=INK, linewidths=1.2, zorder=3)
    ax.set_xlabel("first singular vector")
    ax.set_ylabel("second singular vector")
    ax.set_aspect("equal", adjustable="datalim")
    state = "projected" if project else "before projection"
    _title(ax, f"Embedding of the {side}, {state}")
    return ax


def plot_history(value: Coclustering, *, ax: Axes | None = None) -> Axes:
    """Draw the objective after each iteration of the start that was kept.

    See the mathematical model: :doc:`/model/estimation`.

    Args:
        value: A co-clustering.
        ax: The axes to draw on. Defaults to new axes.

    Returns:
        The axes.
    """
    _check(value)
    objectives = np.array(value.history.objectives, dtype=float)
    ax = _axes(ax, (6.0, 3.4))
    _tidy(ax)
    ax.plot(np.arange(1, objectives.shape[0] + 1), objectives, color=INK, linewidth=2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("objective")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    state = "converged" if value.history.converged else "stopped before converging"
    _title(ax, f"Objective: {objectives.shape[0]} iterations, {state}")
    return ax


def plot_mixedness(value: MixedCoclustering, *, bins: int = 10, ax: Axes | None = None) -> Axes:
    """Draw how mixed the voters are: the histogram of their mixedness.

    The mixedness of a voter is one minus its largest membership: 0 for a voter of a
    single group, up to ``1 - 1 / Q``. It is shown over the voters, never as a list by
    voter.

    See the mathematical model: :doc:`/model/reading-results`, equation
    :eq:`eq-results-mixedness`.

    Args:
        value: A mixed co-clustering.
        bins: The number of bars, at least 1.
        ax: The axes to draw on. Defaults to new axes.

    Returns:
        The axes.
    """
    _check(value, MixedCoclustering)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise InvalidParameterError(f"bins is an integer of at least 1, got {bins!r}")
    mixedness = value.mixedness
    ax = _axes(ax, (6.0, 3.4))
    _tidy(ax)
    upper = 1.0 - 1.0 / max(value.n_groups, 1)
    ax.hist(
        mixedness,
        bins=bins,
        range=(0.0, max(upper, 1e-9)),
        color=INK_SECONDARY,
        edgecolor=SURFACE,
        linewidth=2,
    )
    ax.set_xlabel("mixedness: one minus the largest membership")
    ax.set_ylabel("voters")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _title(ax, f"Mixedness of {len(value.voters)} voters, median {float(np.median(mixedness)):.2f}")
    return ax
