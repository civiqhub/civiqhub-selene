# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("numpy")

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from civiqhub_selene.blocks import MixedMembershipBlockModel, SpectralStart, VoteMatrix
from civiqhub_selene.blocks.mixed import _expect, _profiles
from civiqhub_selene.blocks.responses import Categorical, OrdinalProbit
from civiqhub_selene.blocks.spectral import ENCODINGS, spectral_embedding

pytestmark = pytest.mark.blocks


@st.composite
def matrices(draw: st.DrawFn) -> VoteMatrix:
    """Small vote matrices of any kind: structure is neither planted nor expected."""
    n_voters = draw(st.integers(min_value=2, max_value=9))
    n_items = draw(st.integers(min_value=2, max_value=6))
    cells = n_voters * n_items
    levels = draw(st.lists(st.integers(0, 5), min_size=cells, max_size=cells))
    shown = draw(st.lists(st.booleans(), min_size=cells, max_size=cells))
    return VoteMatrix.from_arrays(
        np.array(levels).reshape(n_voters, n_items),
        shown=np.array(shown).reshape(n_voters, n_items),
    )


@settings(max_examples=25)
@given(
    votes=matrices(),
    n_groups=st.integers(1, 3),
    n_blocks=st.integers(1, 2),
    probit=st.booleans(),
    spectral=st.booleans(),
    seed=st.integers(0, 50),
)
def test_any_matrix_gives_a_sound_value_or_an_explained_absence(
    votes: VoteMatrix, n_groups: int, n_blocks: int, probit: bool, spectral: bool, seed: int
) -> None:
    model = MixedMembershipBlockModel(
        n_groups=n_groups,
        n_blocks=n_blocks,
        response_model=OrdinalProbit() if probit else Categorical(smoothing=0.0),
        initialization=SpectralStart() if spectral else MixedMembershipBlockModel().initialization,
        n_starts=2,
        max_iterations=30,
    )
    result = model.estimate(votes, seed=seed)
    if not result.is_available:
        assert result.unavailable is not None
        assert result.unavailable.detail
        return
    value = result.value
    n_voters, n_items = votes.shape
    assert (value.n_groups, value.n_blocks) == (n_groups, n_blocks)
    assert value.group_memberships.sum(axis=1) == pytest.approx(np.ones(n_voters))
    assert value.block_memberships.sum(axis=1) == pytest.approx(np.ones(n_items))
    assert value.profile_probabilities.sum(axis=1) == pytest.approx(np.ones(n_voters))
    assert value.role_by_block.sum(axis=2) == pytest.approx(np.ones((n_voters, n_blocks)))
    assert np.isfinite(value.objective)
    assert np.isfinite(value.log_likelihood)
    assert (np.diff(value.history.objectives) >= -1e-7).all()
    assert value.coclusters.n_votes.sum() == pytest.approx(votes.n_votes)
    assert np.isfinite(value.log_probability(votes)).all()
    assert sum(value.groups.sizes) == n_voters
    assert model.estimate(votes, seed=seed).value.fingerprint() == value.fingerprint()


@settings(max_examples=30)
@given(votes=matrices(), seed=st.integers(0, 1000))
def test_the_expectation_follows_a_reordering_of_the_voters(votes: VoteMatrix, seed: int) -> None:
    rng = np.random.default_rng(seed)
    theta = rng.dirichlet(np.ones(2), size=votes.shape[0])
    phi = rng.dirichlet(np.ones(2), size=votes.shape[1])
    logs = np.log(rng.dirichlet(np.ones(votes.n_levels + 1), size=(2, 2)))
    order = rng.permutation(votes.shape[0])
    shuffled = VoteMatrix.from_arrays(votes.levels[order], shown=votes.shown[order])
    first = _expect(votes._states(), votes.shown, theta, phi, logs, _profiles(2, 2))
    second = _expect(shuffled._states(), shuffled.shown, theta[order], phi, logs, _profiles(2, 2))
    assert second.log_likelihood == pytest.approx(first.log_likelihood, rel=1e-10, abs=1e-10)
    assert second.role_by_block == pytest.approx(first.role_by_block[order])
    assert second.counts == pytest.approx(first.counts)
    assert second.item_mass == pytest.approx(first.item_mass)


@settings(max_examples=30)
@given(votes=matrices(), encoding=st.sampled_from(ENCODINGS), seed=st.integers(0, 1000))
def test_the_spectral_labels_cover_every_cluster(
    votes: VoteMatrix, encoding: str, seed: int
) -> None:
    n_groups = min(2, votes.shape[0])
    n_blocks = min(2, votes.shape[1])
    embedding = spectral_embedding(
        votes,
        n_voter_components=n_groups,
        n_item_components=n_blocks,
        encoding=encoding,  # type: ignore[arg-type]
    )
    groups, blocks = embedding.labels(n_groups, n_blocks, np.random.default_rng(seed))
    assert np.isfinite(embedding.voter_vectors).all()
    assert np.isfinite(embedding.item_vectors).all()
    assert sorted(set(groups.tolist())) == list(range(n_groups))
    assert sorted(set(blocks.tolist())) == list(range(n_blocks))
