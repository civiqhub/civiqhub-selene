# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Co-clustering of voters and items. Needs NumPy, behind the ``blocks`` extra.

A group gathers voters, a block gathers items, and a co-cluster is the pair of a group
and a block. This subpackage holds the data the block models work on, the models, the
choice of their shape and the consensus between groups.

Install it with ``pip install "civiqhub-selene[blocks]"``. Without NumPy, importing
this subpackage raises :class:`~civiqhub_selene.errors.MissingExtraError`; the package
root and ``civiqhub_selene.ordinal`` never import it.
"""

from civiqhub_selene.errors import MissingExtraError

try:
    import numpy  # noqa: F401
except ImportError as error:
    raise MissingExtraError("civiqhub_selene.blocks needs NumPy, which is not installed") from error

from civiqhub_selene.blocks.coclusters import Cocluster, CoclusterTable
from civiqhub_selene.blocks.data import GroupedVotes, OrdinalScale, Vote, VoteMatrix
from civiqhub_selene.blocks.partition import ItemBlock, Partition, VoterGroup
from civiqhub_selene.blocks.single import LatentBlockModel

__all__ = [
    "Cocluster",
    "CoclusterTable",
    "GroupedVotes",
    "ItemBlock",
    "LatentBlockModel",
    "OrdinalScale",
    "Partition",
    "Vote",
    "VoteMatrix",
    "VoterGroup",
]
