# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene.blocks import (
    FixedStart,
    LatentBlockModel,
    MixedMembershipBlockModel,
    Partition,
    RandomStart,
    SpectralStart,
    VoteMatrix,
)
from civiqhub_selene.blocks.synthetic import make_votes, single_memberships
from civiqhub_selene.errors import InvalidInputError, InvalidParameterError
from civiqhub_selene.types import settings_of

pytestmark = pytest.mark.blocks

VOTES, TRUTH = make_votes(single_memberships((20, 16, 12)), single_memberships((7, 7)), seed=0)
PLANTED_GROUPS = Partition.from_voter_labels(VOTES.voters, TRUTH.roles[:, 0])
PLANTED_BLOCKS = Partition.from_item_labels(VOTES.items, np.repeat([0, 1], 7))
STARTS = [RandomStart(), SpectralStart(), SpectralStart(encoding="scores", leverage_threshold=1.0)]


@pytest.mark.parametrize("start", STARTS)
@pytest.mark.parametrize("shape", [(1, 1), (3, 2), (5, 4)])
def test_an_arrangement_has_one_label_per_voter_and_item_and_no_empty_cluster(
    start: Any, shape: tuple[int, int]
) -> None:
    groups, blocks = start.arrangement(VOTES, shape, np.random.default_rng(0))
    assert groups.shape == (48,)
    assert blocks.shape == (14,)
    assert groups.dtype == np.int64
    assert blocks.dtype == np.int64
    assert (np.bincount(groups, minlength=shape[0]) >= 1).all()
    assert (np.bincount(blocks, minlength=shape[1]) >= 1).all()
    assert groups.max() < shape[0]
    assert blocks.max() < shape[1]


def test_the_random_start_deals_shuffled_voters_in_turn() -> None:
    rng = np.random.default_rng(4)
    groups, blocks = RandomStart().arrangement(VOTES, (3, 2), rng)
    replay = np.random.default_rng(4)
    assert groups.tolist() == (replay.permutation(48) % 3).tolist()
    assert blocks.tolist() == (replay.permutation(14) % 2).tolist()
    assert np.bincount(groups).tolist() == [16, 16, 16]


def test_the_spectral_start_finds_the_planted_arrangement() -> None:
    groups, blocks = SpectralStart().arrangement(VOTES, (3, 2), np.random.default_rng(0))
    assert Partition.from_voter_labels(VOTES.voters, groups).agreement_with(PLANTED_GROUPS) >= 0.9
    assert Partition.from_item_labels(VOTES.items, blocks).agreement_with(PLANTED_BLOCKS) == 1.0


def test_start_objects_are_settings_of_the_model() -> None:
    model = LatentBlockModel(initialization=SpectralStart(regularization=0.5))
    settings = dict(settings_of(model))
    assert settings["initialization"] == (
        ("type", "SpectralStart"),
        ("encoding", "states"),
        ("leverage_threshold", 0.0),
        ("n_init", 10),
        ("project", True),
        ("regularization", 0.5),
    )
    assert LatentBlockModel() == LatentBlockModel(initialization=RandomStart())
    assert LatentBlockModel() != LatentBlockModel(initialization=SpectralStart())
    result = model.estimate(VOTES, seed=0)
    assert dict(result.provenance.parameters)["initialization"] == settings["initialization"]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"encoding": "signs"}, "encoding"),
        ({"regularization": -0.1}, "regularization"),
        ({"regularization": "1"}, "regularization"),
        ({"leverage_threshold": float("nan")}, "leverage_threshold"),
        ({"project": "yes"}, "project"),
        ({"n_init": 0}, "n_init"),
        ({"n_init": True}, "n_init"),
    ],
)
def test_spectral_settings_out_of_range_are_refused(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(InvalidParameterError, match=message):
        SpectralStart(**changes)


def test_a_model_refuses_an_initialization_that_gives_no_arrangement() -> None:
    with pytest.raises(InvalidParameterError, match="initialization"):
        LatentBlockModel(initialization="spectral")  # type: ignore[arg-type]
    with pytest.raises(InvalidParameterError, match="initialization"):
        MixedMembershipBlockModel(initialization=object())  # type: ignore[arg-type]


def test_the_latent_block_model_finds_the_planted_truth_from_a_spectral_start() -> None:
    model = LatentBlockModel(n_groups=3, n_blocks=2, n_starts=1, initialization=SpectralStart())
    value = model.estimate(VOTES, seed=0).value
    assert value.groups.agreement_with(PLANTED_GROUPS) >= 0.9
    assert value.blocks.agreement_with(PLANTED_BLOCKS) == 1.0
    reference = LatentBlockModel(n_groups=3, n_blocks=2).estimate(VOTES, seed=0).value
    assert value.objective >= reference.objective - 1e-6


def test_the_default_start_leaves_the_latent_block_model_unchanged() -> None:
    explicit = LatentBlockModel(n_groups=3, n_blocks=2, initialization=RandomStart())
    default = LatentBlockModel(n_groups=3, n_blocks=2)
    assert explicit.estimate(VOTES, seed=3).value.fingerprint() == (
        default.estimate(VOTES, seed=3).value.fingerprint()
    )


# FixedStart -----------------------------------------------------------------------------


def test_a_fixed_start_gives_its_labels_back() -> None:
    start = FixedStart((0, 1, 1), (1, 0))
    votes = VoteMatrix.from_arrays([[5, 1], [1, 5], [1, 4]])
    groups, blocks = start.arrangement(votes, (2, 2), np.random.default_rng(0))
    assert groups.tolist() == [0, 1, 1]
    assert blocks.tolist() == [1, 0]
    assert start.source is None


def test_a_fixed_start_from_a_coclustering_cites_it() -> None:
    value = LatentBlockModel(n_groups=3, n_blocks=2).estimate(VOTES, seed=0).value
    start = FixedStart.from_coclustering(value)
    assert start.group_index == tuple(value.group_index.tolist())
    assert start.block_index == tuple(value.block_index.tolist())
    assert start.source == value.computation
    again = LatentBlockModel(n_groups=3, n_blocks=2, n_starts=1, initialization=start)
    assert again.estimate(VOTES, seed=0).value.groups.agreement_with(value.groups) == 1.0


def test_fixed_labels_are_integers() -> None:
    assert FixedStart([np.int64(0), 1], [0]).group_index == (0, 1)  # type: ignore[arg-type]
    for labels in ((0, -1), (0, 1.0), (True, 0)):
        with pytest.raises(InvalidParameterError, match="group_index"):
            FixedStart(labels, (0,))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("groups", "blocks", "message"),
    [
        ((0, 1), (0, 1), "3 voters"),
        ((0, 1, 1), (0,), "2 items"),
        ((0, 0, 0), (0, 1), "every label"),
        ((0, 1, 2), (0, 1), "every label"),
    ],
)
def test_a_fixed_start_that_does_not_fit_is_refused(
    groups: tuple[int, ...], blocks: tuple[int, ...], message: str
) -> None:
    votes = VoteMatrix.from_arrays([[5, 1], [1, 5], [1, 4]])
    with pytest.raises(InvalidInputError, match=message):
        FixedStart(groups, blocks).arrangement(votes, (2, 2), np.random.default_rng(0))
