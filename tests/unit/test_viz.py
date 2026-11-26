# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from collections.abc import Callable, Iterator
from typing import Any

import pytest

pytest.importorskip("numpy")
pytest.importorskip("networkx")
pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes

from civiqhub_selene import viz
from civiqhub_selene.blocks import LatentBlockModel, MixedMembershipBlockModel, SpectralStart
from civiqhub_selene.blocks.spectral import spectral_embedding
from civiqhub_selene.blocks.synthetic import make_votes, single_memberships
from civiqhub_selene.errors import InvalidInputError, InvalidParameterError
from civiqhub_selene.viz.style import (
    GROUP_COLORS,
    INK_MUTED,
    SCORE_RAMP,
    group_color,
    score_colors,
)

pytestmark = [pytest.mark.blocks, pytest.mark.viz]

VOTES, _ = make_votes(single_memberships((12, 10, 8)), single_memberships((5, 4)), seed=0)
SINGLE = (
    LatentBlockModel(n_groups=3, n_blocks=2, initialization=SpectralStart()).estimate(VOTES).value
)
MIXED_VOTES, _ = make_votes(
    np.vstack([single_memberships((10, 10, 10)), [[0.5, 0.5, 0.0]] * 6]),
    np.vstack([single_memberships((5, 5)), [[0.5, 0.5]]]),
    seed=1,
)
MIXED = (
    MixedMembershipBlockModel(n_groups=3, n_blocks=2, n_starts=2, initialization=SpectralStart())
    .estimate(MIXED_VOTES)
    .value
)
EMBEDDING = spectral_embedding(VOTES, n_voter_components=3, n_item_components=2)


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    yield
    plt.close("all")


# Graphs -------------------------------------------------------------------------------


def test_the_coclustering_graph_has_one_edge_per_cocluster() -> None:
    graph = viz.coclustering_graph(SINGLE)
    assert graph.number_of_nodes() == 5
    assert graph.number_of_edges() == 6
    assert nx.is_bipartite(graph)
    group = graph.nodes[("group", 0)]
    assert group["kind"] == "group"
    assert group["label"] == "G0"
    assert group["bipartite"] == 0
    assert sum(graph.nodes[("group", q)]["size"] for q in range(3)) == 30
    edge = graph.edges[("group", 1), ("block", 0)]
    cocluster = SINGLE[1, 0]
    assert edge["vote_rate"] == pytest.approx(cocluster.vote_rate)
    assert edge["typical_score"] == cocluster.typical_score
    assert edge["mean_score"] == pytest.approx(np.dot(cocluster.vote_shares, (-2, -1, 0, 1, 2)))
    assert edge["latent_mean"] == cocluster.latent_mean
    assert graph.graph["computation"] == SINGLE.computation


def test_the_membership_graph_links_each_voter_to_its_groups() -> None:
    graph = viz.membership_graph(MIXED, threshold=0.2)
    voters = [node for node in graph if node[0] == "voter"]
    assert len(voters) == 36
    for node in voters:
        data = graph.nodes[node]
        assert ("group", data["main"]) in graph[node]
        for neighbour in graph[node]:
            weight = float(graph.edges[node, neighbour]["weight"])
            assert weight >= 0.2 or neighbour == ("group", data["main"])
    shared = [node for node in voters if graph.degree(node) >= 2]
    assert len(shared) >= 4
    items = viz.membership_graph(MIXED, side="items", threshold=0.0)
    assert items.number_of_edges() == 11 * 2


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"side": "groups"}, "side"),
        ({"threshold": 1.5}, "threshold"),
        ({"threshold": "a"}, "threshold"),
    ],
)
def test_membership_settings_out_of_range_are_refused(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(InvalidParameterError, match=message):
        viz.membership_graph(SINGLE, **changes)


def test_the_role_graph_follows_the_role_changes() -> None:
    graph = viz.role_graph(MIXED)
    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() == 3 * 2
    for block in range(2):
        assert sum(graph.nodes[("role", block, q)]["count"] for q in range(3)) == 36
    changes = MIXED.role_changes(0, 1)
    total = 0
    for source, target, data in graph.edges(data=True):
        count = int(data["count"])
        groups = (int(graph.nodes[source]["group"]), int(graph.nodes[target]["group"]))
        assert count == changes[groups]
        total += count
    assert total == 36
    assert all(data["changes_role"] == (u[2] != v[2]) for u, v, data in graph.edges(data=True))


def test_the_group_graph_measures_agreement_between_zero_and_one() -> None:
    graph = viz.group_graph(SINGLE)
    assert graph.number_of_edges() == 3
    for _, _, data in graph.edges(data=True):
        assert 0.0 <= data["agreement"] <= 1.0
        assert len(data["by_block"]) == 2
    # Groups 0 and 1 hold opposite opinions on both blocks of the default collective.
    approving = SINGLE.group_of("v1").index
    rejecting = SINGLE.group_of("v13").index
    assert graph.edges[("group", approving), ("group", rejecting)]["agreement"] < 0.3


def test_the_vote_graph_has_one_edge_per_vote() -> None:
    graph = viz.vote_graph(VOTES, SINGLE)
    assert graph.number_of_edges() == VOTES.n_votes
    assert graph.number_of_nodes() == 30 + 9
    voter, item = next(iter(VOTES.to_records()))[:2]
    assert graph.edges[("voter", voter), ("item", item)]["score"] in (-2, -1, 0, 1, 2)
    assert graph.nodes[("voter", "v1")]["group"] == SINGLE.group_of("v1").index
    assert "group" not in viz.vote_graph(VOTES).nodes[("voter", "v1")]


def test_graphs_refuse_what_is_not_a_coclustering() -> None:
    with pytest.raises(InvalidInputError, match="Coclustering"):
        viz.coclustering_graph(VOTES)  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError, match="MixedCoclustering"):
        viz.role_graph(SINGLE)  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError, match="VoteMatrix"):
        viz.vote_graph(SINGLE)  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError, match="other voters"):
        viz.vote_graph(MIXED_VOTES, SINGLE)


# Figures ------------------------------------------------------------------------------

FIGURES: list[tuple[str, Callable[[], Axes]]] = [
    ("coclustering", lambda: viz.draw_coclustering(SINGLE)),
    ("memberships", lambda: viz.draw_memberships(MIXED)),
    ("memberships of items", lambda: viz.draw_memberships(MIXED, side="items")),
    (
        "memberships, one group",
        lambda: viz.draw_memberships(
            LatentBlockModel(n_groups=1, n_blocks=2).estimate(VOTES).value
        ),
    ),
    ("roles", lambda: viz.draw_roles(MIXED)),
    ("groups", lambda: viz.draw_groups(SINGLE)),
    ("matrix", lambda: viz.plot_vote_matrix(VOTES, SINGLE)),
    ("raw matrix", lambda: viz.plot_vote_matrix(VOTES)),
    ("spectrum", lambda: viz.plot_spectrum(EMBEDDING)),
    ("embedding", lambda: viz.plot_embedding(EMBEDDING, SINGLE.group_index)),
    (
        "projected items",
        lambda: viz.plot_embedding(EMBEDDING, SINGLE.block_index, side="items", project=True),
    ),
    ("history", lambda: viz.plot_history(MIXED)),
    ("mixedness", lambda: viz.plot_mixedness(MIXED)),
]


@pytest.mark.parametrize(("name", "draw"), FIGURES, ids=[name for name, _ in FIGURES])
def test_every_figure_draws_on_new_axes(name: str, draw: Callable[[], Axes]) -> None:
    ax = draw()
    assert isinstance(ax, Axes)
    assert ax.get_title(loc="left")
    ax.figure.canvas.draw()


def test_a_figure_draws_on_the_axes_it_receives() -> None:
    _, (left, right) = plt.subplots(1, 2)
    assert viz.draw_coclustering(SINGLE, ax=left) is left
    assert viz.plot_vote_matrix(VOTES, SINGLE, ax=right) is right


def test_the_matrix_shows_the_three_states_of_a_cell() -> None:
    hidden = np.zeros(VOTES.shape, dtype=bool)
    hidden[0, 0] = True
    ax = viz.plot_vote_matrix(VOTES.hold_out(hidden))
    image = ax.get_images()[0].get_array()
    assert image is not None
    assert image[0, 0, 3] == 0.0  # not shown: transparent over the hatching
    legend = ax.get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["-2", "-1", "0", "+1", "+2", "no vote", "not shown"]


def test_figures_refuse_bad_inputs() -> None:
    with pytest.raises(InvalidInputError):
        viz.draw_roles(SINGLE)  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError, match="other voters"):
        viz.plot_vote_matrix(MIXED_VOTES, SINGLE)
    with pytest.raises(InvalidParameterError, match="two"):
        viz.plot_embedding(spectral_embedding(VOTES, n_voter_components=1))
    with pytest.raises(InvalidParameterError, match="labels"):
        viz.plot_embedding(EMBEDDING, [0, 1])
    with pytest.raises(InvalidParameterError, match="n_values"):
        viz.plot_spectrum(EMBEDDING, n_values=0)
    with pytest.raises(InvalidParameterError, match="bins"):
        viz.plot_mixedness(MIXED, bins=0)


# Style --------------------------------------------------------------------------------


def test_group_colours_follow_a_fixed_order_then_fold_to_grey() -> None:
    assert [group_color(q) for q in range(8)] == list(GROUP_COLORS)
    assert group_color(8) == INK_MUTED
    with pytest.raises(ValueError, match="at least 0"):
        group_color(-1)


def test_score_colours_keep_the_middle_neutral() -> None:
    assert score_colors(5) == SCORE_RAMP
    seven = score_colors(7)
    assert len(seven) == 7
    assert seven[0] == SCORE_RAMP[0]
    assert seven[3] == SCORE_RAMP[2]
    assert seven[-1] == SCORE_RAMP[-1]
    assert score_colors(3) == (SCORE_RAMP[0], SCORE_RAMP[2], SCORE_RAMP[4])
    with pytest.raises(ValueError, match="two levels"):
        score_colors(1)


def test_without_networkx_viz_says_what_to_install() -> None:
    script = (
        "import sys\n"
        "sys.modules['networkx'] = None\n"
        "import civiqhub_selene.blocks\n"
        "from civiqhub_selene.errors import MissingExtraError\n"
        "try:\n"
        "    import civiqhub_selene.viz\n"
        "except MissingExtraError as error:\n"
        "    assert 'civiqhub-selene[viz]' in error.hint, error.hint\n"
        "    assert 'networkx' in str(error), str(error)\n"
        "else:\n"
        "    raise AssertionError('civiqhub_selene.viz was imported without NetworkX')\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_the_engine_never_imports_viz() -> None:
    script = (
        "import sys\n"
        "import civiqhub_selene.blocks\n"
        "from civiqhub_selene.blocks import LatentBlockModel, MixedMembershipBlockModel\n"
        "assert 'civiqhub_selene.viz' not in sys.modules\n"
        "assert 'networkx' not in sys.modules\n"
        "assert 'matplotlib' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_viz_is_not_a_public_name_of_the_root() -> None:
    import civiqhub_selene

    assert "viz" not in civiqhub_selene.__all__
    assert np.isfinite(SINGLE.objective)


def test_every_figure_reads_items_on_several_scales() -> None:
    from civiqhub_selene.blocks import OrdinalScale

    # TEST ONLY. Items mixing scales of two, four, five and six answers.
    scales = [OrdinalScale(tuple(range(1, (2, 4, 5, 6)[item % 4] + 1))) for item in range(12)]
    votes, _ = make_votes(
        single_memberships((15, 15)),
        single_memberships((6, 6)),
        vote_rate=[[0.8, 0.6], [0.6, 0.8]],
        latent_mean=[[0.8, 0.25], [0.2, 0.75]],
        latent_spread=[[0.12, 0.14], [0.14, 0.12]],
        item_scales=scales,
        seed=4,
    )
    single = LatentBlockModel(n_groups=2, n_blocks=2).estimate(votes, seed=0).value
    mixed = (
        MixedMembershipBlockModel(n_groups=2, n_blocks=2, max_iterations=20)
        .estimate(votes, seed=0)
        .value
    )
    edge = next(iter(viz.graphs.coclustering_graph(single).edges(data=True)))[2]
    assert "mean_score" not in edge
    assert 0.0 <= edge["mean_position"] <= 1.0
    assert len(edge["typical_score_by_scale"]) == 4
    agreement = viz.graphs.group_graph(single).edges[("group", 0), ("group", 1)]["agreement"]
    assert 0.0 <= agreement <= 1.0
    scores = {data["score"] for _, _, data in viz.graphs.vote_graph(votes).edges(data=True)}
    assert scores <= set(range(1, 7))
    for figure in (
        viz.draw.draw_coclustering(single),
        viz.draw.draw_groups(single),
        viz.draw.draw_memberships(mixed),
        viz.draw.draw_roles(mixed),
        viz.draw.plot_vote_matrix(votes, single),
        viz.draw.plot_vote_matrix(votes),
        viz.draw.plot_history(mixed),
    ):
        assert isinstance(figure, Axes)
