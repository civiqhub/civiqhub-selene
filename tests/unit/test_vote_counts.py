# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import re
from fractions import Fraction

import pytest

from civiqhub_selene.errors import InvalidInputError
from civiqhub_selene.ordinal import SCORES, VoteCounts


def test_scores_of_the_agreement_scale() -> None:
    assert SCORES == (-2, -1, 0, 1, 2)


def test_total_and_cumulative_shares() -> None:
    votes = VoteCounts([0, 0, 10, 0, 10])  # type: ignore[arg-type]
    assert votes.counts == (0, 0, 10, 0, 10)
    assert votes.total == 20
    assert votes.cumulative_shares == (
        Fraction(0),
        Fraction(0),
        Fraction(1, 2),
        Fraction(1, 2),
    )
    assert votes.shares == (Fraction(0), Fraction(0), Fraction(1, 2), Fraction(0), Fraction(1, 2))
    assert sum(votes.shares) == 1


def test_from_scores_counts_each_score() -> None:
    assert VoteCounts.from_scores([2, 2, 0, -1]) == VoteCounts((0, 1, 1, 0, 2))
    assert VoteCounts.from_scores([], name="p-01") == VoteCounts((0, 0, 0, 0, 0), "p-01")
    assert VoteCounts.from_scores(iter([-2, -2])).counts == (2, 0, 0, 0, 0)


@pytest.mark.parametrize(
    "counts",
    [
        (1, 2, 3),
        (1, 2, 3, 4, 5, 6),
        (1, -1, 0, 0, 0),
        (1.0, 0, 0, 0, 0),
        (True, 0, 0, 0, 0),
        ("1", 0, 0, 0, 0),
        (None, 0, 0, 0, 0),
        "12345",
        5,
        None,
    ],
)
def test_invalid_counts_are_refused_and_quoted(counts: object) -> None:
    with pytest.raises(InvalidInputError) as caught:
        VoteCounts(counts)  # type: ignore[arg-type]
    assert repr(counts) in str(caught.value)


@pytest.mark.parametrize("score", [3, -3, 1.0, True, "1", None])
def test_a_score_out_of_the_scale_is_refused_and_named(score: object) -> None:
    with pytest.raises(InvalidInputError, match=re.escape(repr(score))):
        VoteCounts.from_scores([0, score])  # type: ignore[list-item]
    with pytest.raises(InvalidInputError, match=re.escape(repr(score))):
        _ = VoteCounts((1, 1, 1, 1, 1))[score]  # type: ignore[index]


def test_a_name_is_text() -> None:
    with pytest.raises(InvalidInputError, match="name"):
        VoteCounts((0, 0, 0, 0, 0), name=3)  # type: ignore[arg-type]


@pytest.mark.parametrize("counts", [(0, 0, 0, 0, 0), (1, 2, 3, 4, 5), (0, 0, 10, 0, 10)])
@pytest.mark.parametrize("score", SCORES)
def test_adding_then_removing_a_vote_gives_the_same_counts(
    counts: tuple[int, int, int, int, int], score: int
) -> None:
    votes = VoteCounts(counts, "p-01")
    more = votes.with_vote(score)
    assert more.total == votes.total + 1
    assert more[score] == votes[score] + 1
    assert more.name == "p-01"
    assert more.without_vote(score) == votes


def test_removing_a_vote_that_does_not_exist_is_refused() -> None:
    with pytest.raises(InvalidInputError, match="no vote at score \\+0"):
        VoteCounts((0, 0, 0, 0, 0)).without_vote(0)
    with pytest.raises(InvalidInputError, match="no vote at score -2"):
        VoteCounts((0, 1, 1, 1, 1)).without_vote(-2)


def test_reading_one_count_by_its_score() -> None:
    votes = VoteCounts((1, 2, 3, 4, 5))
    assert votes[2] == 5
    assert votes[-2] == 1
    assert votes[0] == 3
    with pytest.raises(InvalidInputError, match="got 5"):
        _ = votes[5]


def test_vote_counts_are_not_a_sequence() -> None:
    with pytest.raises(TypeError, match="not iterable"):
        list(VoteCounts((1, 2, 3, 4, 5)))  # type: ignore[call-overload]


def test_shares_are_undefined_without_a_vote() -> None:
    empty = VoteCounts((0, 0, 0, 0, 0))
    assert empty.total == 0
    with pytest.raises(InvalidInputError, match="without a vote"):
        _ = empty.shares
    with pytest.raises(InvalidInputError, match="without a vote"):
        _ = empty.cumulative_shares


def test_fingerprint_follows_the_counts_and_the_name() -> None:
    votes = VoteCounts((1, 2, 3, 4, 5), "p-01")
    assert votes.fingerprint() == VoteCounts([1, 2, 3, 4, 5], "p-01").fingerprint()  # type: ignore[arg-type]
    assert re.fullmatch(r"[0-9a-f]{16}", votes.fingerprint())
    assert votes.fingerprint() != VoteCounts((1, 2, 3, 4, 6), "p-01").fingerprint()
    assert votes.fingerprint() != VoteCounts((1, 2, 3, 4, 5), "p-02").fingerprint()
    assert votes.fingerprint() != VoteCounts((1, 2, 3, 4, 5)).fingerprint()


def test_vote_counts_are_frozen_and_hashable() -> None:
    votes = VoteCounts((1, 2, 3, 4, 5))
    with pytest.raises(dataclasses.FrozenInstanceError):
        votes.name = "changed"  # type: ignore[misc]
    assert hash(votes) == hash(VoteCounts((1, 2, 3, 4, 5)))
    assert len({votes, VoteCounts((1, 2, 3, 4, 5)), VoteCounts((1, 2, 3, 4, 5), "x")}) == 2


def test_description_fits_on_one_line() -> None:
    assert repr(VoteCounts((0, 0, 10, 0, 10), "p-01")) == (
        "VoteCounts(p-01: 0, 0, 10, 0, 10; 20 votes)"
    )
    assert repr(VoteCounts((0, 0, 10, 0, 10))) == "VoteCounts(0, 0, 10, 0, 10; 20 votes)"
    assert str(VoteCounts((0, 0, 1, 0, 0))) == "VoteCounts(0, 0, 1, 0, 0; 1 vote)"
