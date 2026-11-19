# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import json
import pathlib
from typing import Any

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene.blocks import LatentBlockModel, OrdinalScale, Partition, VoteMatrix

pytestmark = pytest.mark.blocks

VECTOR: dict[str, Any] = json.loads(
    pathlib.Path(__file__).with_name("latent_block_model.json").read_text(encoding="utf-8")
)


def test_reference_run_on_the_stored_votes() -> None:
    settings, expected, tolerance = VECTOR["settings"], VECTOR["expected"], VECTOR["tolerance"]
    votes = VoteMatrix.from_arrays(
        np.array(VECTOR["votes"]["levels"]), scale=OrdinalScale(tuple(VECTOR["votes"]["scores"]))
    )
    model = LatentBlockModel(
        n_groups=settings["n_groups"],
        n_blocks=settings["n_blocks"],
        n_starts=settings["n_starts"],
        max_iterations=settings["max_iterations"],
        tolerance=settings["tolerance"],
    )
    assert type(model).method().name == VECTOR["method"]["name"]
    assert type(model).method().version == VECTOR["method"]["version"]
    value = model.estimate(votes, seed=settings["seed"]).value

    # The arrangement is compared up to the numbering of groups and blocks.
    groups = Partition.from_voter_labels(votes.voters, expected["group_index"])
    blocks = Partition.from_item_labels(votes.items, expected["block_index"])
    assert value.groups.agreement_with(groups) == 1.0
    assert value.blocks.agreement_with(blocks) == 1.0
    group_of = [int(value.group_of(group.voters[0]).index) for group in groups]
    block_of = [int(value.block_of(block.items[0]).index) for block in blocks]

    assert value.objective == pytest.approx(expected["objective"], abs=tolerance)
    assert value.integrated_likelihood == pytest.approx(
        expected["integrated_likelihood"], abs=tolerance
    )
    table = value.coclusters
    assert table.latent_mean is not None
    assert table.latent_spread is not None
    for name, computed in (
        ("vote_rate", table.vote_rate),
        ("latent_mean", table.latent_mean),
        ("latent_spread", table.latent_spread),
        ("n_votes", table.n_votes),
    ):
        reordered = computed[np.ix_(group_of, block_of)]
        assert reordered == pytest.approx(np.array(expected[name]), abs=tolerance), name

    planted = Partition.from_voter_labels(votes.voters, VECTOR["planted"]["group_index"])
    assert value.groups.agreement_with(planted) == pytest.approx(
        expected["agreement_with_planted_groups"]
    )
    planted_blocks = Partition.from_item_labels(votes.items, VECTOR["planted"]["block_index"])
    assert value.blocks.agreement_with(planted_blocks) == 1.0
