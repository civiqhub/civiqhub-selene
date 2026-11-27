# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import re
import subprocess
import sys

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene.blocks import ItemScales, OrdinalScale, Partition, VoteMatrix
from civiqhub_selene.blocks import synthetic as synthetic_module
from civiqhub_selene.blocks.synthetic import (
    DEFAULT_LATENT_MEAN,
    DEFAULT_LATENT_SPREAD,
    DEFAULT_VOTE_RATE,
    PlantedTruth,
    make_votes,
    single_memberships,
)
from civiqhub_selene.errors import InvalidParameterError

pytestmark = pytest.mark.blocks

GROUPS = single_memberships((20, 16, 12))
BLOCKS = single_memberships((7, 7))


def test_single_memberships_are_indicator_rows_in_order() -> None:
    assert single_memberships((2, 1)).tolist() == [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    assert GROUPS.shape == (48, 3)
    assert GROUPS.sum(axis=0).tolist() == [20.0, 16.0, 12.0]
    assert single_memberships((0, 2)).tolist() == [[0.0, 1.0], [0.0, 1.0]]
    for sizes in [(2, -1), (1.5,), (True,)]:
        with pytest.raises(InvalidParameterError, match="size of a cluster"):
            single_memberships(sizes)  # type: ignore[arg-type]


def test_defaults_are_for_three_groups_and_two_blocks() -> None:
    for default in (DEFAULT_VOTE_RATE, DEFAULT_LATENT_MEAN, DEFAULT_LATENT_SPREAD):
        assert np.array(default).shape == (3, 2)
    assert DEFAULT_VOTE_RATE[0] == (0.85, 0.45)
    assert DEFAULT_LATENT_MEAN[1] == (1.3, 4.6)
    assert DEFAULT_LATENT_SPREAD[2] == (1.0, 1.0)


def test_plain_memberships_give_one_role_per_voter() -> None:
    matrix, truth = make_votes(GROUPS, BLOCKS, seed=0)
    assert isinstance(matrix, VoteMatrix)
    assert isinstance(truth, PlantedTruth)
    assert matrix.shape == (48, 14)
    assert matrix.n_levels == 5
    assert matrix.scale.scores == (-2, -1, 0, 1, 2)
    assert matrix.n_shown == 48 * 14
    assert truth.roles.shape == (48, 2)
    group_of = np.repeat([0, 1, 2], [20, 16, 12])
    assert truth.roles[:, 0].tolist() == group_of.tolist()
    assert truth.roles[:, 1].tolist() == group_of.tolist()
    assert np.array_equal(truth.voter_memberships, GROUPS)
    assert np.array_equal(truth.item_memberships, BLOCKS)
    for array in (truth.voter_memberships, truth.item_memberships, truth.roles):
        with pytest.raises(ValueError, match="read-only"):
            array[0, 0] = 1


def test_identifiers_do_not_resemble_real_data() -> None:
    matrix, _ = make_votes(GROUPS, BLOCKS)
    assert matrix.voters[:2] == ("v1", "v2")
    assert matrix.voters[-1] == "v48"
    assert matrix.items == tuple(f"i{number}" for number in range(1, 15))


def test_the_same_seed_gives_the_same_matrix_in_two_processes() -> None:
    matrix, _ = make_votes(GROUPS, BLOCKS, seed=0)
    again, _ = make_votes(GROUPS, BLOCKS, seed=0)
    other, _ = make_votes(GROUPS, BLOCKS, seed=1)
    assert matrix == again
    assert matrix.fingerprint() != other.fingerprint()
    script = (
        "from civiqhub_selene.blocks.synthetic import make_votes, single_memberships\n"
        "matrix, _ = make_votes(single_memberships((20, 16, 12)), single_memberships((7, 7)))\n"
        "print(matrix.fingerprint())\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == matrix.fingerprint()


def test_observed_vote_rates_and_typical_scores_follow_the_planted_ones() -> None:
    matrix, truth = make_votes(single_memberships((200, 200, 200)), BLOCKS, seed=0)
    groups = Partition.from_voter_labels(matrix.voters, truth.roles[:, 0])
    blocks = Partition.from_item_labels(matrix.items, np.repeat([0, 1], 7))
    table = matrix.group_by(groups, blocks).to_table()
    assert np.abs(table.vote_rate - np.array(DEFAULT_VOTE_RATE)).max() < 0.05
    assert table[0, 0].typical_score == 2
    assert table[0, 1].typical_score == -2
    assert table[1, 0].typical_score == -2
    assert table[1, 1].typical_score == 2
    assert table[2, 0].typical_score == 0


def test_spread_memberships_plant_voters_who_change_role() -> None:
    mixed = np.vstack([single_memberships((30, 30)), np.full((40, 2), 0.5)])
    matrix, truth = make_votes(
        mixed,
        BLOCKS,
        vote_rate=np.full((2, 2), 0.8),
        latent_mean=[[4.7, 1.4], [1.3, 4.6]],
        latent_spread=np.full((2, 2), 0.6),
        seed=3,
    )
    assert matrix.shape == (100, 14)
    plain, shared = truth.roles[:60], truth.roles[60:]
    assert (plain[:, 0] == plain[:, 1]).all()
    assert 0 < (shared[:, 0] != shared[:, 1]).sum() < 40


def test_spread_item_memberships_spread_the_cells_of_an_item() -> None:
    items = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    matrix, _ = make_votes(single_memberships((300, 300, 0)), items, seed=5)
    table = matrix.group_by(
        Partition.from_voter_labels(matrix.voters, np.repeat([0, 1], 300)),
        Partition.from_item_labels(matrix.items, [0, 1, 2]),
    ).to_table()
    # Group 0 approves block 0 and rejects block 1: the shared item splits its votes.
    shared = table[0, 2].vote_shares
    assert shared[0] > 0.15
    assert shared[4] > 0.15
    assert table[0, 0].vote_shares[4] > 0.6


def test_another_number_of_levels_uses_scores_from_one() -> None:
    matrix, _ = make_votes(
        single_memberships((30,)),
        single_memberships((4,)),
        vote_rate=[[1.0]],
        latent_mean=[[2.0]],
        latent_spread=[[1.0]],
        n_levels=3,
        seed=2,
    )
    assert matrix.scale.scores == (1, 2, 3)
    assert matrix.n_votes == 120
    assert set(np.unique(matrix.levels).tolist()) == {1, 2, 3}


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"voter_memberships": [[0.5, 0.4, 0.0]]}, "row 0 sums to 0.9"),
        ({"voter_memberships": [[1.5, -0.5, 0.0]]}, "not negative"),
        ({"voter_memberships": [1.0, 0.0, 0.0]}, "one row per identifier"),
        ({"voter_memberships": "groups"}, "table of numbers"),
        ({"item_memberships": [[0.2, 0.2]]}, "each row of item_memberships sums to 1"),
        ({"vote_rate": [[0.5, 0.5]]}, r"vote_rate has the shape (groups, blocks) = (3, 2)"),
        ({"latent_mean": np.zeros((2, 3))}, "latent_mean has the shape"),
        ({"latent_spread": np.zeros((3, 2))}, "strictly positive"),
        ({"vote_rate": np.full((3, 2), 1.2)}, "between 0 and 1"),
        ({"latent_mean": np.full((3, 2), np.nan)}, "finite"),
        ({"n_levels": 1}, "at least 2"),
        ({"n_levels": 5.0}, "at least 2"),
        ({"seed": -1}, "non-negative integer"),
        ({"seed": 1.5}, "non-negative integer"),
    ],
)
def test_invalid_parameters_are_refused(changes: dict[str, object], named: str) -> None:
    arguments: dict[str, object] = {"voter_memberships": GROUPS, "item_memberships": BLOCKS}
    arguments.update(changes)
    with pytest.raises(InvalidParameterError, match=re.escape(named)):
        make_votes(**arguments)  # type: ignore[arg-type]


def test_the_docstring_says_the_data_are_synthetic_and_avoids_greek_letters() -> None:
    docstring = make_votes.__doc__ or ""
    assert "synthetic" in docstring
    assert "not real data" in docstring
    for parameter in ("voter_memberships", "item_memberships", "vote_rate", "latent_mean"):
        assert f"{parameter}:" in docstring
    text = (synthetic_module.__doc__ or "") + docstring
    assert not re.search(r"[Ͱ-Ͽ]", text)
    for word in ("theta", "phi", "sigma", "alpha", "beta", "gamma", "omega", "lambda"):
        assert not re.search(rf"\b{word}\b", text)


# Several scales -----------------------------------------------------------------------

YES_NO = OrdinalScale((0, 1))
FOUR = OrdinalScale((1, 2, 3, 4))


def test_a_draw_on_one_scale_is_unchanged() -> None:
    # Stored on develop at d80da06, before items could have their own scales.
    votes, truth = make_votes(single_memberships((30, 25, 20)), single_memberships((7, 6)), seed=3)
    assert votes.fingerprint() == "56a338260ef3ad30"
    assert truth.item_scales == votes.item_scales
    assert [grid.tolist() for grid in truth.thresholds] == [[1.5, 2.5, 3.5, 4.5]]


def test_each_item_is_drawn_on_its_scale_with_the_same_draws() -> None:
    scales = [YES_NO, FOUR] * 7
    one, _ = make_votes(GROUPS, BLOCKS, seed=4)
    several, truth = make_votes(GROUPS, BLOCKS, item_scales=scales, seed=4)
    assert several.item_scales == ItemScales.from_scales(scales)
    assert (several.levels[:, 0::2] <= 2).all()
    assert (several.levels[:, 1::2] <= 4).all()
    # The draws come in the same order: the cells that carry a vote are the same.
    assert np.array_equal(several.levels > 0, one.levels > 0)
    assert np.array_equal(truth.roles, make_votes(GROUPS, BLOCKS, seed=4)[1].roles)
    assert [grid.tolist() for grid in truth.thresholds] == [[0.5], [0.25, 0.5, 0.75]]


def test_planted_grids_cut_the_opinions_of_their_scale() -> None:
    low = {FOUR: (0.05, 0.1, 0.15)}
    votes, truth = make_votes(
        GROUPS, BLOCKS, item_scales=[FOUR] * 7 + [YES_NO] * 7, thresholds=low, seed=1
    )
    # With every threshold near 0, almost every vote on four answers is at the top.
    four = votes.levels[:, :7][votes.voted[:, :7]]
    assert (four == 4).mean() > 0.8
    assert truth.thresholds[0].tolist() == [0.05, 0.1, 0.15]


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"item_scales": [FOUR] * 13}, "13 items"),
        ({"item_scales": [FOUR] * 14, "n_levels": 4}, "not both"),
        ({"thresholds": {FOUR: (0.2, 0.5, 0.8)}}, "with item_scales only"),
        ({"item_scales": [FOUR] * 14, "thresholds": {FOUR: (0.5, 0.2, 0.8)}}, "increasing"),
        ({"item_scales": [FOUR] * 13 + [(1, 2)]}, "OrdinalScale"),
    ],
)
def test_scales_that_do_not_follow_the_items_are_refused(
    changes: dict[str, object], named: str
) -> None:
    with pytest.raises(InvalidParameterError, match=named):
        make_votes(GROUPS, BLOCKS, **changes)  # type: ignore[arg-type]
