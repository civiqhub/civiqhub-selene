# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Vote counts of one item on the five-score agreement scale. Standard library only.

:class:`VoteCounts` is the input of the whole ordinal family: how many votes each score
received, from -2 (strongly disagree) to +2 (strongly agree), 0 being the neutral
answer. Nothing else: neither the identity of voters, nor abstentions.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

from civiqhub_selene.errors import InvalidInputError
from civiqhub_selene.types import Displayable, _fingerprint

__all__ = ["SCORES", "VoteCounts"]

SCORES = (-2, -1, 0, 1, 2)
"""The five scores of the agreement scale, in order."""


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _position(score: object) -> int:
    if not _is_integer(score) or score not in SCORES:
        raise InvalidInputError(f"a score is an integer from -2 to +2, got {score!r}")
    return SCORES.index(score)


@dataclass(frozen=True, repr=False)
class VoteCounts(Displayable):
    """The five counts of votes of one item, from score -2 to score +2.

    ``counts[k]`` is the number of votes at score ``SCORES[k]``. Any sequence of five
    non-negative integers is accepted and stored as a tuple. Shares are exact
    fractions, so that an equality at a threshold on paper is an equality in the code.

    Attributes:
        counts: The five counts, in the order of :data:`SCORES`.
        name: Free name of what was counted, for display and provenance. The engine
            never interprets it.

    Raises:
        InvalidInputError: If there are not five counts, or if a count is negative or
            is not an integer (a boolean and a float are refused).
    """

    counts: tuple[int, int, int, int, int]
    name: str = ""

    __iter__ = None  # votes[score] reads one count; the object is not a sequence

    def __post_init__(self) -> None:
        received = self.counts
        if isinstance(received, (str, bytes)) or not isinstance(received, Iterable):
            raise InvalidInputError(f"vote counts are five integers, got {received!r}")
        counts = tuple(received)
        if len(counts) != len(SCORES):
            raise InvalidInputError(
                f"vote counts are five integers, one per score from -2 to +2, "
                f"got {len(counts)}: {received!r}"
            )
        for count in counts:
            if not _is_integer(count):
                raise InvalidInputError(
                    f"a vote count is an integer, got {count!r} in {received!r}"
                )
            if count < 0:
                raise InvalidInputError(
                    f"a vote count is not negative, got {count!r} in {received!r}"
                )
        if not isinstance(self.name, str):
            raise InvalidInputError(f"the name of vote counts is text, got {self.name!r}")
        object.__setattr__(self, "counts", counts)

    @classmethod
    def from_scores(cls, scores: Iterable[int], *, name: str = "") -> "VoteCounts":
        """Count a series of scores.

        Raises:
            InvalidInputError: If a score is not an integer from -2 to +2.
        """
        counts = [0] * len(SCORES)
        for score in scores:
            counts[_position(score)] += 1
        return cls(tuple(counts), name)  # type: ignore[arg-type]

    def __getitem__(self, score: int) -> int:
        """Return the count of one score, from -2 to +2."""
        return self.counts[_position(score)]

    @property
    def total(self) -> int:
        """The number of votes: the sum of the five counts."""
        return sum(self.counts)

    @property
    def shares(self) -> tuple[Fraction, Fraction, Fraction, Fraction, Fraction]:
        """The five shares of votes, as exact fractions.

        Raises:
            InvalidInputError: If there is no vote.
        """
        total = self._positive_total()
        first, second, third, fourth, fifth = (Fraction(count, total) for count in self.counts)
        return (first, second, third, fourth, fifth)

    @property
    def cumulative_shares(self) -> tuple[Fraction, Fraction, Fraction, Fraction]:
        """Four exact fractions: the share of votes at -2, at most -1, at most 0, at most +1.

        Raises:
            InvalidInputError: If there is no vote.
        """
        total = self._positive_total()
        first, second, third, fourth, _ = self.counts
        return (
            Fraction(first, total),
            Fraction(first + second, total),
            Fraction(first + second + third, total),
            Fraction(first + second + third + fourth, total),
        )

    def with_vote(self, score: int) -> "VoteCounts":
        """Return new counts with one more vote at ``score``."""
        return self._shifted(score, 1)

    def without_vote(self, score: int) -> "VoteCounts":
        """Return new counts with one vote fewer at ``score``.

        Raises:
            InvalidInputError: If there is no vote at this score.
        """
        if self[score] == 0:
            raise InvalidInputError(f"there is no vote at score {score:+d} to remove from {self!r}")
        return self._shifted(score, -1)

    def fingerprint(self) -> str:
        """Return a stable fingerprint of the counts and the name: 16 hexadecimal characters."""
        return _fingerprint(["VoteCounts", self.counts, self.name])

    def _shifted(self, score: int, change: int) -> "VoteCounts":
        counts = list(self.counts)
        counts[_position(score)] += change
        return VoteCounts(tuple(counts), self.name)  # type: ignore[arg-type]

    def _positive_total(self) -> int:
        total = self.total
        if total == 0:
            raise InvalidInputError(f"shares are undefined without a vote: {self!r}")
        return total

    def _describe(self) -> list[str]:
        prefix = f"{self.name}: " if self.name else ""
        counts = ", ".join(str(count) for count in self.counts)
        total = self.total
        return [f"VoteCounts({prefix}{counts}; {total} vote{'' if total == 1 else 's'})"]
