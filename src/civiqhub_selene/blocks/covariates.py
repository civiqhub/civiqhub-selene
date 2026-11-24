# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Voter attributes aligned to a vote matrix, for analysis beside co-clustering."""

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import TypeAlias

from civiqhub_selene.blocks.data import VoteMatrix, _identifiers
from civiqhub_selene.errors import InvalidInputError

__all__ = ["VoterCovariates"]

Covariate: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True)
class VoterCovariates:
    """A table of voter attributes whose columns follow the order of ``voters``.

    Values are text, booleans, numbers, or ``None`` for missing values.
    Use :meth:`from_records` to align a table to a :class:`VoteMatrix` by voter ID.
    """

    voters: tuple[str, ...]
    columns: Mapping[str, tuple[Covariate, ...]]

    def __post_init__(self) -> None:
        voters = _identifiers(self.voters, "voter")
        names = _identifiers(self.columns, "covariate")
        columns = {name: tuple(self.columns[name]) for name in names}
        for name, values in columns.items():
            if len(values) != len(voters):
                raise InvalidInputError(f"covariate {name!r} must have one value per voter")
            if any(
                value is not None
                and (
                    not isinstance(value, (str, bool, int, float))
                    or (isinstance(value, float) and not math.isfinite(value))
                )
                for value in values
            ):
                raise InvalidInputError(f"covariate {name!r} contains an invalid value")
        object.__setattr__(self, "voters", voters)
        object.__setattr__(self, "columns", MappingProxyType(columns))

    @classmethod
    def from_records(
        cls, votes: VoteMatrix, records: Mapping[str, Mapping[str, Covariate]]
    ) -> "VoterCovariates":
        """Align one record per voter to the matrix, with identical fields in each record."""
        if set(records) != set(votes.voters):
            raise InvalidInputError("covariate records must match the voter IDs")
        rows = [records[voter] for voter in votes.voters]
        names = tuple(rows[0]) if rows else ()
        if any(row.keys() != rows[0].keys() for row in rows):
            raise InvalidInputError("covariate records must have the same fields")
        return cls(votes.voters, {name: tuple(row[name] for row in rows) for name in names})

    def fingerprint(self) -> str:
        """Return a stable fingerprint of the voters and columns: 16 hexadecimal characters.

        SHA-256 of the voter identifiers and of each covariate column, in column order.
        """
        return self._fingerprint

    @cached_property
    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(repr((self.voters, tuple(self.columns.items()))).encode())
        return digest.hexdigest()[:16]
