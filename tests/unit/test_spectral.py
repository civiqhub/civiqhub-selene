# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene.blocks import Partition, VoteMatrix
from civiqhub_selene.blocks.spectral import (
    ENCODINGS,
    SpectralEmbedding,
    _cluster_rows,
    _fill_empty,
    _laplacian,
    n_singular_values,
    spectral_embedding,
)
from civiqhub_selene.blocks.synthetic import make_votes, single_memberships
from civiqhub_selene.errors import InvalidInputError, InvalidParameterError

pytestmark = pytest.mark.blocks

VOTES, TRUTH = make_votes(single_memberships((20, 16, 12)), single_memberships((7, 7)), seed=0)
PLANTED_GROUPS = Partition.from_voter_labels(VOTES.voters, TRUTH.roles[:, 0])
PLANTED_BLOCKS = Partition.from_item_labels(VOTES.items, np.repeat([0, 1], 7))


def agreement(labels: Any, planted: Partition[Any], items: bool = False) -> float:
    build = Partition.from_item_labels if items else Partition.from_voter_labels
    return build(planted.ids, labels).agreement_with(planted)


# The Laplacian --------------------------------------------------------------------------


def test_the_laplacian_follows_its_definition_on_a_small_matrix() -> None:
    matrix = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    rows, columns = matrix.sum(axis=1), matrix.sum(axis=0)
    tau_rows, tau_columns = rows.mean(), columns.mean()
    expected = (
        np.diag(1 / np.sqrt(rows + tau_rows)) @ matrix @ np.diag(1 / np.sqrt(columns + tau_columns))
    )
    assert _laplacian(matrix, 1.0) == pytest.approx(expected)
    assert _laplacian(matrix, 0.0) == pytest.approx(
        np.diag(1 / np.sqrt(rows)) @ matrix @ np.diag(1 / np.sqrt(columns))
    )


def test_a_zero_degree_without_regulariser_gives_zeros_not_infinities() -> None:
    matrix = np.array([[0.0, 0.0], [1.0, 1.0]])
    laplacian = _laplacian(matrix, 0.0)
    assert np.isfinite(laplacian).all()
    assert laplacian[0].tolist() == [0.0, 0.0]


def test_the_states_encoding_decomposes_the_one_hot_states_of_each_side() -> None:
    embedding = spectral_embedding(VOTES, n_voter_components=3, n_item_components=2)
    states = VOTES._states()
    voters = _laplacian(states.reshape(VOTES.shape[0], -1), 1.0)
    items = _laplacian(states.transpose(1, 0, 2).reshape(VOTES.shape[1], -1), 1.0)
    assert embedding.voter_singular_values == pytest.approx(np.linalg.svd(voters)[1])
    assert embedding.item_singular_values == pytest.approx(np.linalg.svd(items)[1])
    # The leading singular vector of a degree-normalised matrix has one sign.
    first = embedding.voter_vectors[:, 0]
    assert (first > 0).all() or (first < 0).all()


def test_the_scores_encoding_is_one_decomposition_for_both_sides() -> None:
    embedding = spectral_embedding(VOTES, encoding="scores")
    assert embedding.voter_singular_values.tolist() == embedding.item_singular_values.tolist()
    centred = np.where(VOTES.voted, VOTES.levels - 3.0, 0.0)
    expected = _laplacian(centred, 1.0, degrees=VOTES.voted.astype(float))
    assert embedding.voter_singular_values == pytest.approx(np.linalg.svd(expected)[1])


def test_singular_values_decrease_and_vectors_are_orthonormal() -> None:
    for encoding in ENCODINGS:
        embedding = spectral_embedding(
            VOTES, n_voter_components=3, n_item_components=2, encoding=encoding
        )
        for values in (embedding.voter_singular_values, embedding.item_singular_values):
            assert (np.diff(values) <= 1e-12).all()
        gram = embedding.voter_vectors.T @ embedding.voter_vectors
        assert gram == pytest.approx(np.eye(3), abs=1e-9)


def test_a_cell_that_is_not_shown_enters_nowhere() -> None:
    hidden = np.random.default_rng(3).random(VOTES.shape) < 0.3
    held = VOTES.hold_out(hidden)
    erased = VoteMatrix.from_arrays(np.where(hidden, 0, VOTES.levels), shown=~hidden)
    for encoding in ENCODINGS:
        first = spectral_embedding(held, encoding=encoding)
        second = spectral_embedding(erased, encoding=encoding)
        assert first.voter_singular_values == pytest.approx(second.voter_singular_values)
        assert first.item_singular_values == pytest.approx(second.item_singular_values)


def test_the_embedding_is_a_read_only_value() -> None:
    embedding = spectral_embedding(VOTES)
    assert isinstance(embedding, SpectralEmbedding)
    assert embedding.voters == VOTES.voters
    assert embedding.items == VOTES.items
    assert (embedding.n_voter_components, embedding.n_item_components) == (2, 2)
    with pytest.raises(ValueError, match="read-only"):
        embedding.voter_vectors[0, 0] = 1.0
    assert repr(embedding).startswith("SpectralEmbedding(48 voters in 2 dimensions")
    assert "leading singular values of the items" in str(embedding)
    assert embedding.voter_leverage.shape == (48,)
    assert np.linalg.norm(embedding.voter_directions(), axis=1) == pytest.approx(np.ones(48))


def test_the_decomposition_is_cached_by_content() -> None:
    copy = VoteMatrix.from_arrays(VOTES.levels.copy(), shown=VOTES.shown.copy())
    first = spectral_embedding(VOTES)
    second = spectral_embedding(copy)
    assert first.voter_singular_values.tolist() == second.voter_singular_values.tolist()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"encoding": "signs"}, "encoding"),
        ({"regularization": -1.0}, "regularization"),
        ({"regularization": float("inf")}, "regularization"),
        ({"regularization": True}, "regularization"),
        ({"n_voter_components": 0}, "n_voter_components"),
        ({"n_voter_components": 1000}, "n_voter_components"),
        ({"n_item_components": 2.0}, "n_item_components"),
    ],
)
def test_settings_out_of_range_are_refused(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(InvalidParameterError, match=message):
        spectral_embedding(VOTES, **changes)


def test_an_input_that_is_not_a_matrix_or_is_empty_is_refused() -> None:
    with pytest.raises(InvalidInputError, match="VoteMatrix"):
        spectral_embedding([[1, 2], [3, 4]])  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError, match="voters and items"):
        spectral_embedding(VoteMatrix.from_arrays(np.zeros((0, 3), dtype=np.int64)))


# The clustering of the rows ------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_planted_groups_and_blocks_are_found_before_any_model(seed: int) -> None:
    embedding = spectral_embedding(VOTES, n_voter_components=3, n_item_components=2)
    groups, blocks = embedding.labels(3, 2, np.random.default_rng(seed))
    assert agreement(groups, PLANTED_GROUPS) >= 0.9
    assert agreement(blocks, PLANTED_BLOCKS, items=True) == 1.0


def test_the_states_encoding_sees_a_difference_of_spread_that_the_scores_miss() -> None:
    # Two groups with the same mean opinion on each block: one agrees, one is split.
    votes, truth = make_votes(
        single_memberships((40, 40)),
        single_memberships((10, 10)),
        vote_rate=[[0.8, 0.8], [0.8, 0.8]],
        latent_mean=[[3.0, 4.5], [3.0, 4.5]],
        latent_spread=[[0.4, 0.6], [2.0, 0.6]],
        seed=2,
    )
    planted = Partition.from_voter_labels(votes.voters, truth.roles[:, 0])
    found = {}
    for encoding in ENCODINGS:
        embedding = spectral_embedding(votes, encoding=encoding)
        groups, _ = embedding.labels(2, 2, np.random.default_rng(0))
        found[encoding] = agreement(groups, planted)
    assert found["states"] >= 0.9
    assert found["scores"] < 0.7


def test_labels_leave_no_cluster_empty_even_with_identical_rows() -> None:
    vectors = np.array([[1.0, 0.0]] * 5 + [[0.0, 1.0]])
    labels = _cluster_rows(vectors, 3, np.random.default_rng(0), True, 0.0, 3)
    assert sorted(np.bincount(labels, minlength=3).tolist()) == [1, 1, 4]


def test_an_empty_cluster_takes_the_farthest_row_of_a_larger_cluster() -> None:
    labels = np.array([0, 0, 0, 1])
    distances = np.array([[0.1, 9.0, 9.0], [0.5, 9.0, 9.0], [0.2, 9.0, 9.0], [9.0, 0.0, 9.0]])
    assert _fill_empty(labels, distances).tolist() == [0, 2, 0, 1]


def test_low_leverage_rows_take_the_nearest_centroid() -> None:
    vectors = np.array([[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4 + [[0.001, 0.0005]])
    labels = _cluster_rows(vectors, 2, np.random.default_rng(0), True, 1.0, 5)
    assert labels[-1] == labels[0]
    assert len(set(labels[:4].tolist())) == 1
    assert labels[0] != labels[4]


def test_more_clusters_than_components_is_allowed_as_in_di_sim() -> None:
    embedding = spectral_embedding(VOTES, n_voter_components=1, n_item_components=1)
    groups, blocks = embedding.labels(3, 2, np.random.default_rng(0))
    assert sorted(set(groups.tolist())) == [0, 1, 2]
    assert sorted(set(blocks.tolist())) == [0, 1]


def test_the_number_of_singular_values_of_each_side() -> None:
    assert n_singular_values(VOTES) == (48, 14)
    assert n_singular_values(VOTES, "scores") == (14, 14)
    for encoding in ENCODINGS:
        embedding = spectral_embedding(VOTES, encoding=encoding)
        assert (
            embedding.voter_singular_values.shape[0],
            embedding.item_singular_values.shape[0],
        ) == n_singular_values(VOTES, encoding)


def test_a_spectral_start_with_more_groups_than_items_still_answers() -> None:
    from civiqhub_selene.blocks import SpectralStart

    votes = VoteMatrix.from_arrays([[5, 1], [5, 2], [1, 5], [2, 5], [3, 3], [3, 4]])
    groups, blocks = SpectralStart(encoding="scores").arrangement(
        votes, (3, 2), np.random.default_rng(0)
    )
    assert sorted(set(groups.tolist())) == [0, 1, 2]
    assert sorted(set(blocks.tolist())) == [0, 1]


def test_one_cluster_puts_every_row_in_it() -> None:
    embedding = spectral_embedding(VOTES, n_voter_components=1, n_item_components=1)
    groups, blocks = embedding.labels(1, 1, np.random.default_rng(0))
    assert set(groups.tolist()) == {0}
    assert set(blocks.tolist()) == {0}


def test_the_same_generator_state_gives_the_same_labels() -> None:
    embedding = spectral_embedding(VOTES, n_voter_components=3, n_item_components=2)
    first = embedding.labels(3, 2, np.random.default_rng(5))
    second = embedding.labels(3, 2, np.random.default_rng(5))
    assert first[0].tolist() == second[0].tolist()
    assert first[1].tolist() == second[1].tolist()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"n_groups": 49}, "n_groups"),
        ({"n_blocks": 0}, "n_blocks"),
        ({"project": 1}, "project"),
        ({"leverage_threshold": -0.5}, "leverage_threshold"),
        ({"n_init": 0}, "n_init"),
    ],
)
def test_label_settings_out_of_range_are_refused(changes: dict[str, Any], message: str) -> None:
    embedding = spectral_embedding(VOTES)
    arguments: dict[str, Any] = {"n_groups": 2, "n_blocks": 2, **changes}
    n_groups, n_blocks = arguments.pop("n_groups"), arguments.pop("n_blocks")
    with pytest.raises(InvalidParameterError, match=message):
        embedding.labels(n_groups, n_blocks, np.random.default_rng(0), **arguments)


def test_an_embedding_with_inconsistent_arrays_is_refused() -> None:
    with pytest.raises(InvalidInputError, match="voter_vectors"):
        SpectralEmbedding(
            voters=("a", "b"),
            items=("x",),
            encoding="states",
            regularization=1.0,
            voter_vectors=np.zeros((3, 1)),
            item_vectors=np.zeros((1, 1)),
            voter_singular_values=np.ones(1),
            item_singular_values=np.ones(1),
        )
