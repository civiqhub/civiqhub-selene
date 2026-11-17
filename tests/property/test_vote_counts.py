# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

from fractions import Fraction
from itertools import pairwise

from hypothesis import given
from hypothesis import strategies as st

from civiqhub_selene.ordinal import SCORES, VoteCounts

counts = st.tuples(*[st.integers(min_value=0, max_value=10_000)] * 5)
scores = st.sampled_from(SCORES)


@given(counts=counts, score=scores)
def test_adding_then_removing_a_vote_is_the_identity(
    counts: tuple[int, int, int, int, int], score: int
) -> None:
    votes = VoteCounts(counts)
    assert votes.with_vote(score).without_vote(score) == votes
    assert votes.with_vote(score).total == votes.total + 1


@given(counts=counts.filter(lambda values: sum(values) > 0))
def test_shares_are_an_exact_distribution(counts: tuple[int, int, int, int, int]) -> None:
    votes = VoteCounts(counts)
    assert sum(votes.shares) == 1
    assert all(0 <= share <= 1 for share in votes.shares)
    cumulative = votes.cumulative_shares
    assert all(low <= high for low, high in pairwise(cumulative))
    assert cumulative[-1] == 1 - Fraction(votes[2], votes.total)


@given(series=st.lists(scores, max_size=200))
def test_counting_scores_keeps_every_vote(series: list[int]) -> None:
    votes = VoteCounts.from_scores(series)
    assert votes.total == len(series)
    assert all(votes[score] == series.count(score) for score in SCORES)


@given(counts=counts, name=st.text(max_size=20))
def test_equal_counts_have_the_same_fingerprint(
    counts: tuple[int, int, int, int, int], name: str
) -> None:
    assert VoteCounts(counts, name).fingerprint() == VoteCounts(list(counts), name).fingerprint()  # type: ignore[arg-type]
