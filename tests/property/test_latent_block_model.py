# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("numpy")

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from civiqhub_selene.blocks import LatentBlockModel, VoteMatrix
from civiqhub_selene.blocks.responses import Categorical, OrdinalProbit

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


@settings(max_examples=30)
@given(
    votes=matrices(),
    n_groups=st.integers(1, 3),
    n_blocks=st.integers(1, 2),
    probit=st.booleans(),
    seed=st.integers(0, 50),
)
def test_any_matrix_gives_a_sound_value_or_an_explained_absence(
    votes: VoteMatrix, n_groups: int, n_blocks: int, probit: bool, seed: int
) -> None:
    model = LatentBlockModel(
        n_groups=n_groups,
        n_blocks=n_blocks,
        response_model=OrdinalProbit() if probit else Categorical(),
        n_starts=2,
        max_iterations=40,
    )
    result = model.estimate(votes, seed=seed)
    if not result.is_available:
        assert result.unavailable is not None
        assert result.unavailable.detail
        return
    value = result.value
    n_voters, n_items = votes.shape
    assert (value.n_groups, value.n_blocks) == (n_groups, n_blocks)
    assert min(value.groups.sizes) >= 1
    assert min(value.blocks.sizes) >= 1
    assert sum(value.groups.sizes) == n_voters
    assert sum(value.blocks.sizes) == n_items
    assert value.group_memberships.sum(axis=1) == pytest.approx(np.ones(n_voters))
    assert value.block_memberships.sum(axis=1) == pytest.approx(np.ones(n_items))
    assert np.isfinite(value.objective)
    assert np.isfinite(value.integrated_likelihood)
    assert (np.diff(value.history.objectives) >= -1e-9).all()
    assert value.coclusters.n_votes.sum() == pytest.approx(votes.n_votes)
    assert np.isfinite(value.log_probability(votes)).all()
    assert model.estimate(votes, seed=seed).value.fingerprint() == value.fingerprint()
