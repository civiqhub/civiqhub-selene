# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Load wide survey CSV records into the inputs for ordinal co-clustering."""

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from civiqhub_selene.blocks.covariates import VoterCovariates
from civiqhub_selene.blocks.data import OrdinalScale, VoteMatrix
from civiqhub_selene.errors import InvalidInputError

__all__ = ["read_survey_csv"]


def read_survey_csv(
    lines: Iterable[str],
    *,
    item_scales: Mapping[str, OrdinalScale],
    voter_column: str = "responseid",
    weight_column: str | None = None,
    covariate_columns: Sequence[str] = (),
    delimiter: str = ",",
    decimal: Literal[".", ","] = ".",
    unshown_label: str | None = None,
    scale: OrdinalScale | None = None,
) -> tuple[VoteMatrix, VoterCovariates]:
    """Read one CSV row per voter and one column per ordinal item.

    The ordered keys of ``item_scales`` select and order the item columns. CSV item
    values are integer scores on the declared scales. Blank scores indicate
    abstentions. Use ``unshown_label`` for cells treated as not shown, including
    ``""`` when the source uses blank scores for that purpose. Covariates are text
    and blanks are returned as ``None``. Other columns are ignored.

    Every item must share the same observation scale. Distinct per-item scales are
    SEL-8.2 (ItemScales).

    Args:
        lines: An iterable of CSV lines, such as an open text file.
        item_scales: A declared scale for every item column to read.
        voter_column: Column holding the unique voter identifier.
        weight_column: Optional column holding positive survey weights.
        covariate_columns: Columns to retain in the aligned covariate table.
        delimiter: CSV field delimiter.
        decimal: Decimal separator for weights.
        unshown_label: Optional marker for a question treated as not shown.
        scale: Optional scale of the matrix when every item already shares it.

    Returns:
        The matrix and the covariate table, in CSV voter order.
    """
    items = tuple(item_scales)
    covariates: dict[str, list[str | None]] = {name: [] for name in covariate_columns}
    required = (voter_column, *items, *covariate_columns)
    if weight_column is not None:
        required = (*required, weight_column)
    if len(set(required)) != len(required):
        raise InvalidInputError("each selected CSV column must have one role")

    if len(delimiter) != 1 or delimiter in "\r\n":
        raise InvalidInputError("CSV delimiter must be one character other than a newline")
    if decimal not in (".", ","):
        raise InvalidInputError("weight decimal separator must be '.' or ','")
    reader = csv.reader(lines, delimiter=delimiter, strict=True)
    voters: list[str] = []
    weights: list[float] = []
    ratings: list[list[int | None]] = []
    shown: list[list[bool]] = []
    try:
        header = next(reader, [])
        if not header:
            raise InvalidInputError("survey CSV must have a header")
        header[0] = header[0].removeprefix("\ufeff")
        header = [name.strip() for name in header]
        if len(set(header)) != len(header):
            raise InvalidInputError("survey CSV column names must be unique")
        missing = sorted(set(required).difference(header))
        if missing:
            raise InvalidInputError(f"survey CSV is missing columns: {', '.join(missing)}")
        positions = {name: index for index, name in enumerate(header)}

        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise InvalidInputError(
                    f"CSV line {reader.line_num} has {len(row)} fields, expected {len(header)}"
                )
            voter = row[positions[voter_column]].strip()
            if not voter:
                raise InvalidInputError(
                    f"empty voter identifier in column {voter_column!r} at CSV line "
                    f"{reader.line_num}"
                )
            voters.append(voter)
            if weight_column is not None:
                weights.append(
                    _parse_weight(
                        row[positions[weight_column]],
                        column=weight_column,
                        line=reader.line_num,
                        decimal=decimal,
                    )
                )
            for name, values in covariates.items():
                values.append(row[positions[name]].strip() or None)

            scores: list[int | None] = []
            displayed: list[bool] = []
            for item in items:
                value = row[positions[item]].strip()
                displayed.append(value != unshown_label)
                if not value or value == unshown_label:
                    scores.append(None)
                    continue
                try:
                    scores.append(int(value))
                except ValueError as error:
                    raise InvalidInputError(
                        f"expected an integer score for {item!r} at CSV line {reader.line_num}"
                    ) from error
            ratings.append(scores)
            shown.append(displayed)
    except csv.Error as error:
        raise InvalidInputError(f"invalid survey CSV at line {reader.line_num}: {error}") from error

    votes = VoteMatrix.from_responses(
        ratings,
        voters=voters,
        item_scales=item_scales,
        scale=scale,
        shown=shown if voters else None,
        voter_weights=weights if weight_column is not None else None,
    )
    return votes, VoterCovariates(
        votes.voters, {name: tuple(values) for name, values in covariates.items()}
    )


def _parse_weight(raw: str, *, column: str, line: int, decimal: str) -> float:
    """Return a finite positive weight, naming the CSV column on error."""
    text = raw.strip()
    try:
        weight = float(text.replace(decimal, "."))
    except ValueError as error:
        raise InvalidInputError(
            f"invalid weight {text!r} in column {column!r} at CSV line {line}"
        ) from error
    if not math.isfinite(weight) or weight <= 0:
        raise InvalidInputError(
            f"weight {weight} in column {column!r} at CSV line {line} must be "
            "finite and strictly positive"
        )
    return weight
