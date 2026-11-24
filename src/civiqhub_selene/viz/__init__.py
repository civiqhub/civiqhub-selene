# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Graphs and figures of co-clustering results. Needs NetworkX and Matplotlib (``viz`` extra).

Two layers, one per module:

- :mod:`civiqhub_selene.viz.graphs` turns a result into a NetworkX graph whose nodes
  and edges carry its numbers: the co-clusters, the memberships, the roles of the
  voters from block to block, the agreement between groups, the votes themselves;
- :mod:`civiqhub_selene.viz.draw` draws these graphs, and the figures that go with
  them (the sorted vote matrix, the spectrum and the embedding of the spectral start,
  the convergence, the mixedness), with Matplotlib.

Install it with ``pip install "civiqhub-selene[viz]"``, which also installs the
``blocks`` extra. Without NetworkX or Matplotlib, importing this subpackage raises
:class:`~civiqhub_selene.errors.MissingExtraError`. No other part of the engine imports
it: a figure is a reading of a result, never an input of a computation.
"""

from civiqhub_selene.errors import MissingExtraError

try:
    import matplotlib  # noqa: F401
    import networkx  # noqa: F401
    import numpy  # noqa: F401
except ImportError as error:
    raise MissingExtraError(
        f"civiqhub_selene.viz needs NumPy, NetworkX and Matplotlib; {error.name} is not installed",
        hint='Install the extra: pip install "civiqhub-selene[viz]".',
    ) from error

from civiqhub_selene.viz.draw import (
    draw_coclustering,
    draw_groups,
    draw_memberships,
    draw_roles,
    plot_embedding,
    plot_history,
    plot_mixedness,
    plot_spectrum,
    plot_vote_matrix,
)
from civiqhub_selene.viz.graphs import (
    coclustering_graph,
    group_graph,
    membership_graph,
    role_graph,
    vote_graph,
)

__all__ = [
    "coclustering_graph",
    "draw_coclustering",
    "draw_groups",
    "draw_memberships",
    "draw_roles",
    "group_graph",
    "membership_graph",
    "plot_embedding",
    "plot_history",
    "plot_mixedness",
    "plot_spectrum",
    "plot_vote_matrix",
    "role_graph",
    "vote_graph",
]
