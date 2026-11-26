# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Co-clustering of voters and items. Needs NumPy, behind the ``blocks`` extra.

A group gathers voters, a block gathers items, and a co-cluster is the pair of a group
and a block. This subpackage holds the data the block models work on (a matrix of votes
whose items may each use their own scale, :class:`ItemScales`), the models (one
group per voter with :class:`LatentBlockModel`, mixed memberships with
:class:`MixedMembershipBlockModel`) and the arrangements they start from
(:class:`RandomStart`, :class:`SpectralStart`, :class:`FixedStart`).

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
from civiqhub_selene.blocks.covariates import VoterCovariates
from civiqhub_selene.blocks.data import GroupedVotes, ItemScales, OrdinalScale, Vote, VoteMatrix
from civiqhub_selene.blocks.mixed import MixedMembershipBlockModel
from civiqhub_selene.blocks.partition import ItemBlock, Partition, VoterGroup
from civiqhub_selene.blocks.single import LatentBlockModel
from civiqhub_selene.blocks.starts import FixedStart, RandomStart, SpectralStart
from civiqhub_selene.blocks.survey import read_survey_csv

__all__ = [
    "Cocluster",
    "CoclusterTable",
    "FixedStart",
    "GroupedVotes",
    "ItemBlock",
    "ItemScales",
    "LatentBlockModel",
    "MixedMembershipBlockModel",
    "OrdinalScale",
    "Partition",
    "RandomStart",
    "SpectralStart",
    "Vote",
    "VoteMatrix",
    "VoterCovariates",
    "VoterGroup",
    "read_survey_csv",
]
