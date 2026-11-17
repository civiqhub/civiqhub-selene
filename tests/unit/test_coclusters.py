# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene.blocks import Cocluster, CoclusterTable, OrdinalScale
from civiqhub_selene.blocks._numerics import count_states, logsumexp, one_hot
from civiqhub_selene.errors import InvalidInputError

pytestmark = pytest.mark.blocks

SCORES = (-2, -1, 0, 1, 2)


def make_table(**changes: object) -> CoclusterTable:
    fields: dict[str, object] = {
        "scale": OrdinalScale(),
        "vote_rate": [[0.8, 0.0], [1.0, 0.5]],
        "vote_shares": [
            [[0.0, 0.0, 0.0, 0.25, 0.75], [0.2, 0.2, 0.2, 0.2, 0.2]],
            [[0.5, 0.5, 0.0, 0.0, 0.0], [0.1, 0.2, 0.4, 0.2, 0.1]],
        ],
        "n_votes": [[8, 0], [10, 2.5]],
    }
    fields.update(changes)
    return CoclusterTable(**fields)  # type: ignore[arg-type]


# Numerics -----------------------------------------------------------------------------


def test_one_hot() -> None:
    indicator = one_hot(np.array([0, 2, 1]), 3)
    assert indicator.tolist() == [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    assert indicator.dtype == np.float64
    assert one_hot(np.array([], dtype=np.int64), 2).shape == (0, 2)


def test_count_states_matches_a_direct_count() -> None:
    rng = np.random.default_rng(0)
    states = one_hot(rng.integers(0, 4, size=30), 4).reshape(6, 5, 4)
    rows = np.array([0, 0, 1, 1, 2, 2])
    cols = np.array([0, 1, 1, 0, 1])
    counts = count_states(states, rows, cols, (3, 2))
    assert counts.shape == (3, 2, 4)
    expected = np.zeros((3, 2, 4))
    for i in range(6):
        for j in range(5):
            expected[rows[i], cols[j]] += states[i, j]
    assert np.array_equal(counts, expected)
    assert counts.sum() == 30


def test_logsumexp_is_stable() -> None:
    values = np.array([[1000.0, 1000.0], [-1000.0, -1000.0], [0.0, -np.inf]])
    result = logsumexp(values, axis=1)
    assert result.tolist() == pytest.approx([1000.0 + np.log(2), -1000.0 + np.log(2), 0.0])
    assert logsumexp(np.array([-np.inf, -np.inf]), axis=0) == -np.inf
    small = np.log(np.array([[0.1, 0.2, 0.7]]))
    assert logsumexp(small, axis=1).tolist() == pytest.approx([0.0], abs=1e-12)
    assert logsumexp(np.zeros((2, 3, 4)), axis=-1).shape == (2, 3)


# Cocluster ----------------------------------------------------------------------------


def make_cocluster(vote_shares: tuple[float, ...], scores: tuple[int, ...] = SCORES) -> Cocluster:
    return Cocluster(
        group=0,
        block=1,
        vote_rate=0.5,
        vote_shares=vote_shares,
        n_votes=12.0,
        latent_mean=None,
        latent_spread=None,
        scores=scores,
    )


@pytest.mark.parametrize(
    ("vote_shares", "typical"),
    [
        ((0.0, 0.0, 0.0, 0.25, 0.75), 2),
        ((0.6, 0.1, 0.1, 0.1, 0.1), -2),
        ((0.4, 0.1, 0.0, 0.1, 0.4), -2),  # tie at both ends, equally far: the lower score
        ((0.4, 0.0, 0.2, 0.4, 0.0), 1),  # tie: +1 is closer to the middle than -2
        ((0.2, 0.2, 0.2, 0.2, 0.2), 0),  # uniform: the middle of the scale
        ((0.0, 0.5, 0.0, 0.5, 0.0), -1),  # tie, equally close to the middle: the lower score
    ],
)
def test_typical_score(vote_shares: tuple[float, ...], typical: int) -> None:
    assert make_cocluster(vote_shares).typical_score == typical


def test_typical_score_on_an_even_scale() -> None:
    assert make_cocluster((0.25, 0.25, 0.25, 0.25), scores=(1, 2, 3, 4)).typical_score == 2


def test_a_cocluster_is_frozen_and_prints_on_one_line() -> None:
    cocluster = make_cocluster((0.0, 0.0, 0.0, 0.25, 0.75))
    assert repr(cocluster) == (
        "Cocluster(group=0, block=1, vote_rate=0.500, typical_score=+2, n_votes=12)"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        cocluster.group = 2  # type: ignore[misc]
    assert cocluster == make_cocluster((0.0, 0.0, 0.0, 0.25, 0.75))


# CoclusterTable -----------------------------------------------------------------------


def test_a_table_gives_its_coclusters() -> None:
    table = make_table()
    assert table.shape == (2, 2)
    assert len(table) == 4
    cocluster = table[0, 0]
    assert isinstance(cocluster, Cocluster)
    assert (cocluster.group, cocluster.block) == (0, 0)
    assert cocluster.vote_rate == 0.8
    assert cocluster.vote_shares == (0.0, 0.0, 0.0, 0.25, 0.75)
    assert cocluster.n_votes == 8.0
    assert cocluster.scores == SCORES
    assert cocluster.typical_score == 2
    assert [(c.group, c.block) for c in table] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert table[1, 1].n_votes == 2.5
    for key in [(2, 0), (0, 2), (-1, 0), 0, (0, 0, 0)]:
        with pytest.raises(IndexError, match=r"table\[group, block\]"):
            table[key]  # type: ignore[index]


def test_latent_parameters_are_read_when_the_table_has_them() -> None:
    table = make_table(latent_mean=[[4.7, 1.4], [1.3, 4.6]], latent_spread=[[0.6, 0.7], [0.7, 0.6]])
    assert table[0, 1].latent_mean == 1.4
    assert table[0, 1].latent_spread == 0.7
    assert make_table()[0, 1].latent_mean is None


def test_log_probabilities_sum_to_one_over_the_states() -> None:
    table = make_table()
    logs = table.log_probabilities()
    assert logs.shape == (2, 2, 6)
    assert np.exp(logs).sum(axis=-1) == pytest.approx(np.ones((2, 2)), abs=1e-9)
    assert logs[0, 0, 0] == pytest.approx(np.log(0.2))
    assert logs[0, 0, 5] == pytest.approx(np.log(0.8) + np.log(0.75))
    # Null probabilities are floored at 1e-300 before the logarithm.
    assert logs[0, 0, 1] == pytest.approx(np.log(1e-300))
    assert logs[1, 0, 0] == pytest.approx(np.log(1e-300))
    assert np.isfinite(logs).all()


def test_to_dict_is_plain_and_json_ready() -> None:
    data = make_table().to_dict()
    assert json.loads(json.dumps(data)) == data
    assert data["scores"] == [-2, -1, 0, 1, 2]
    assert data["vote_rate"] == [[0.8, 0.0], [1.0, 0.5]]
    assert data["n_votes"] == [[8.0, 0.0], [10.0, 2.5]]
    assert set(data) == {"scores", "vote_rate", "vote_shares", "n_votes"}
    latent = make_table(latent_mean=np.ones((2, 2)), latent_spread=np.ones((2, 2))).to_dict()
    assert latent["latent_mean"] == [[1.0, 1.0], [1.0, 1.0]]
    assert latent["latent_spread"] == [[1.0, 1.0], [1.0, 1.0]]


def test_arrays_are_copied_and_read_only() -> None:
    rate = np.array([[0.8, 0.0], [1.0, 0.5]])
    table = make_table(vote_rate=rate)
    rate[0, 0] = 0.1
    assert table.vote_rate[0, 0] == 0.8
    for array in (table.vote_rate, table.vote_shares, table.n_votes):
        with pytest.raises(ValueError, match="read-only"):
            array[0, 0] = 0.0


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"vote_rate": [0.5, 0.5]}, "vote_rate has the shape"),
        ({"vote_rate": [[1.2, 0.0], [1.0, 0.5]]}, "between 0 and 1"),
        ({"vote_rate": [[float("nan"), 0.0], [1.0, 0.5]]}, "not finite"),
        ({"vote_shares": [[[1.0], [1.0]], [[1.0], [1.0]]]}, "vote_shares has the shape"),
        ({"n_votes": [8, 0]}, "n_votes has the shape"),
        ({"n_votes": [[-1, 0], [10, 2]]}, "not negative"),
        ({"latent_mean": [1.0, 2.0]}, "latent_mean has the shape"),
        ({"latent_spread": "wide"}, "latent_spread is an array of numbers"),
        ({"scale": (1, 2, 3)}, "OrdinalScale"),
    ],
)
def test_an_inconsistent_table_is_refused(changes: dict[str, object], named: str) -> None:
    with pytest.raises(InvalidInputError, match=named):
        make_table(**changes)


def test_vote_shares_must_be_a_distribution() -> None:
    shares = np.full((2, 2, 5), 0.2)
    shares[0, 0, 0] = 0.5
    with pytest.raises(InvalidInputError, match="sum to 1"):
        make_table(vote_shares=shares)


def test_description_has_one_line_per_cocluster() -> None:
    lines = str(make_table()).splitlines()
    assert lines[0] == "CoclusterTable(2 groups x 2 blocks, 5 levels)"
    assert lines[1] == "group 0 x block 0: vote rate 0.80, typical score +2, 8 votes"
    assert lines[2] == "group 0 x block 1: vote rate 0.00, typical score 0, 0 votes"
    assert lines[4] == "group 1 x block 1: vote rate 0.50, typical score 0, 2.5 votes"
    assert len(lines) == 5
    assert repr(make_table()) == lines[0]
