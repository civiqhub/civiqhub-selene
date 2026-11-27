# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("numpy")

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from civiqhub_selene.blocks.responses import _shared_vote_rate, _vote_rate

pytestmark = pytest.mark.blocks

MIN_PROBABILITY = 1e-6

# Counts of Q x L co-clusters over K + 1 = 6 states, integer or fractional.
counts = st.tuples(st.integers(1, 4), st.integers(1, 4)).flatmap(
    lambda shape: arrays(
        np.float64,
        (*shape, 6),
        elements=st.floats(0.0, 50.0, allow_nan=False, allow_infinity=False),
    )
)


@given(counts=counts)
def test_a_shared_rate_is_one_number_between_the_bounds(counts: np.ndarray) -> None:
    previous = np.full(counts.shape[:2], 0.5)
    rate = _shared_vote_rate(counts, previous, MIN_PROBABILITY)
    assert rate.shape == previous.shape
    assert np.unique(rate).size == 1
    assert MIN_PROBABILITY <= rate[0, 0] <= 1 - MIN_PROBABILITY


@given(counts=counts, seed=st.integers(0, 2**32 - 1))
def test_a_shared_rate_does_not_depend_on_the_arrangement(counts: np.ndarray, seed: int) -> None:
    # Moving cells between co-clusters, as a new arrangement does, keeps the shared rate.
    previous = np.full(counts.shape[:2], 0.5)
    flat = counts.reshape(-1, counts.shape[-1])
    moved = np.random.default_rng(seed).permutation(flat).reshape(counts.shape)
    pooled = np.zeros_like(counts)
    pooled[0, 0] = flat.sum(axis=0)
    rate = _shared_vote_rate(counts, previous, MIN_PROBABILITY)
    assert _shared_vote_rate(moved, previous, MIN_PROBABILITY) == pytest.approx(rate)
    assert _shared_vote_rate(pooled, previous, MIN_PROBABILITY) == pytest.approx(rate)


@given(counts=counts)
def test_a_shared_rate_is_the_rate_of_all_cells_in_one_cocluster(counts: np.ndarray) -> None:
    previous = np.full(counts.shape[:2], 0.5)
    one = counts.sum(axis=(0, 1))[np.newaxis, np.newaxis, :]
    by_cocluster = _vote_rate(one, np.full((1, 1), 0.5), MIN_PROBABILITY)[0, 0]
    assert _shared_vote_rate(counts, previous, MIN_PROBABILITY)[0, 0] == pytest.approx(by_cocluster)
