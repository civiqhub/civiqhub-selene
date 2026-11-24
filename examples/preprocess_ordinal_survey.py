# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Split a survey table into one vote base per scale size, plus one metadata base.

A survey mixes two kinds of columns: ordinal questions, which go to co-clustering, and
the description of the respondents (sex, age group, region...), which serves to read
the groups afterwards. This example separates them and, because a block model needs
one scale for all its items, groups the ordinal questions by their number of levels
``K``: one base for the questions with 2 levels, one for 3 levels, and so on up to 7.

Every base carries the same identifier column, so that any vote base joins the metadata
base one to one. If the table has no identifier, the row number (from 1) is used.

Encoding of a vote base, the internal encoding of :class:`VoteMatrix`:

- ``1..K``: the rank of the answer on the scale of the question;
- ``0``: the question was asked, without an answer (don't know, refusal): an abstention;
- empty: the question was not asked to this respondent (filter question).

Which raw codes mean "abstention" and which mean "not asked" is a choice of the user,
written on the command line; a blank cell of the source counts as "not asked" unless
``--blank-is-abstention`` is given.

Usage::

    # synthetic data, as every example of this repository
    python examples/preprocess_ordinal_survey.py --demo --out demo_bases

    # one column per question, number of distinct codes per column, to choose the options
    python examples/preprocess_ordinal_survey.py survey.csv --inspect

    # a real table, kept outside the repository
    python examples/preprocess_ordinal_survey.py survey.csv --out bases --id IDENT \\
        --vote-pattern "^(RE|SA)[0-9_]+$" --abstention-codes 8 9 --k-min 2 --k-max 7

Needs pandas, which is not a dependency of the package (``pip install pandas``), and
NumPy with the ``blocks`` extra to build the vote matrices.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROW_NUMBER = "row_number"
"""Name of the identifier column when the source has none."""

MANIFEST = "manifest.json"


# Specification -------------------------------------------------------------------------


@dataclass(frozen=True)
class SurveySpec:
    """How to read a survey table. Every choice is explicit, so that a split is reproducible.

    Exactly one of ``metadata``, ``votes`` or ``vote_pattern`` decides the split: the
    named columns go to their side, all the other columns (except the identifier) to
    the other side. There is no automatic guess: an age group coded 1 to 5 looks exactly
    like a five-level question.

    Attributes:
        id_column: Column of unique respondent identifiers, or None to number the rows.
        metadata: Columns describing the respondents; the rest are questions.
        votes: Columns that are questions; the rest are metadata.
        vote_pattern: Regular expression matching the names of the questions.
        abstention_codes: Raw codes meaning "asked, no answer" (don't know, refusal).
        not_shown_codes: Raw codes meaning "not asked".
        blank_is_abstention: Read a blank cell as an abstention instead of "not asked".
        levels: Declared ordered codes per question, from low to high. A declared scale
            wins over the codes seen in the data, so that a level nobody chose still
            counts.
        k_min: Smallest number of levels kept.
        k_max: Largest number of levels kept.
    """

    id_column: str | None = None
    metadata: tuple[str, ...] | None = None
    votes: tuple[str, ...] | None = None
    vote_pattern: str | None = None
    abstention_codes: tuple[Any, ...] = ()
    not_shown_codes: tuple[Any, ...] = ()
    blank_is_abstention: bool = False
    levels: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    k_min: int = 2
    k_max: int = 7

    def __post_init__(self) -> None:
        chosen = [self.metadata is not None, self.votes is not None, self.vote_pattern is not None]
        if sum(chosen) != 1:
            raise ValueError("give exactly one of metadata, votes or vote_pattern")
        if not 2 <= self.k_min <= self.k_max:
            raise ValueError(f"need 2 <= k_min <= k_max, got {self.k_min} and {self.k_max}")
        overlap = set(self.abstention_codes) & set(self.not_shown_codes)
        if overlap:
            raise ValueError(f"codes both abstention and not shown: {sorted(map(str, overlap))}")

    @property
    def k_values(self) -> range:
        return range(self.k_min, self.k_max + 1)


@dataclass(frozen=True)
class OrdinalItem:
    """One question kept for co-clustering.

    Attributes:
        name: Column name in the source.
        codes: Raw codes from the lowest to the highest level; level ``k`` is ``codes[k-1]``.
        declared: True if the codes come from ``SurveySpec.levels``, False if from the data.
        n_votes: Answers on the scale.
        n_abstentions: Asked without an answer.
        n_not_shown: Not asked.
        note: A warning to read, empty if none.
    """

    name: str
    codes: tuple[Any, ...]
    declared: bool
    n_votes: int
    n_abstentions: int
    n_not_shown: int
    note: str = ""

    @property
    def k(self) -> int:
        return len(self.codes)


# Splitting -----------------------------------------------------------------------------


class SurveySplitter:
    """Split a survey table according to a :class:`SurveySpec`."""

    def __init__(self, spec: SurveySpec) -> None:
        self.spec = spec

    def split(self, table: pd.DataFrame) -> SplitSurvey:
        """Return the metadata base and one vote base per scale size.

        Raises:
            ValueError: If the identifier is missing, empty or repeated, or if a column
                named in the specification is not in the table.
        """
        table, id_name = self._with_identifier(table)
        vote_columns, metadata_columns = self._sides(table, id_name)

        items: list[OrdinalItem] = []
        skipped: dict[str, str] = {}
        encoded: dict[str, pd.Series] = {}
        for column in vote_columns:
            try:
                item, levels = self._encode(table[column])
            except _SkipError as reason:
                skipped[column] = str(reason)
                continue
            if not self.spec.k_min <= item.k <= self.spec.k_max:
                skipped[column] = (
                    f"{item.k} levels, outside {self.spec.k_min}..{self.spec.k_max} "
                    f"(codes {_show(item.codes)})"
                )
                continue
            items.append(item)
            encoded[column] = levels

        ids = table[id_name]
        bases: dict[int, pd.DataFrame] = {}
        for k in self.spec.k_values:
            names = [item.name for item in items if item.k == k]
            if names:
                bases[k] = pd.concat([ids, *(encoded[name] for name in names)], axis=1)
        metadata = table[[id_name, *metadata_columns]].copy()
        return SplitSurvey(id_name, metadata, bases, items, skipped, self.spec)

    # Identifier, sides ------------------------------------------------------------------

    def _with_identifier(self, table: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        spec = self.spec
        if spec.id_column is None:
            if ROW_NUMBER in table.columns:
                raise ValueError(f"the table already has a column {ROW_NUMBER!r}: pass it as id")
            numbered = table.copy()
            numbered.insert(0, ROW_NUMBER, [str(n) for n in range(1, len(table) + 1)])
            return numbered, ROW_NUMBER
        if spec.id_column not in table.columns:
            raise ValueError(f"no identifier column {spec.id_column!r} in the table")
        ids = table[spec.id_column]
        if ids.isna().any():
            n_blank = int(ids.isna().sum())
            raise ValueError(f"the identifier {spec.id_column!r} has {n_blank} blanks")
        as_text = ids.map(_as_text)
        repeated = as_text[as_text.duplicated()].unique()
        if len(repeated):
            raise ValueError(f"the identifier repeats, for example {list(repeated[:5])}")
        identified = table.copy()
        identified[spec.id_column] = as_text  # VoteMatrix takes text identifiers
        return identified, spec.id_column

    def _sides(self, table: pd.DataFrame, id_name: str) -> tuple[list[str], list[str]]:
        spec = self.spec
        others = [column for column in table.columns if column != id_name]
        if spec.vote_pattern is not None:
            pattern = re.compile(spec.vote_pattern)
            votes = [column for column in others if pattern.search(str(column))]
            if not votes:
                raise ValueError(f"no column matches {spec.vote_pattern!r}")
        elif spec.votes is not None:
            _check_present(spec.votes, table, "question")
            votes = [column for column in others if column in spec.votes]
        else:
            assert spec.metadata is not None
            _check_present(spec.metadata, table, "metadata")
            votes = [column for column in others if column not in spec.metadata]
        metadata = [column for column in others if column not in votes]
        return votes, metadata

    # One question ------------------------------------------------------------------------

    def _encode(self, column: pd.Series) -> tuple[OrdinalItem, pd.Series]:
        """Encode one question: levels 1..K, 0 for an abstention, NA when not asked."""
        spec = self.spec
        name = str(column.name)
        blank = column.isna()
        abstained = _isin(column, spec.abstention_codes) | (blank & spec.blank_is_abstention)
        not_shown = _isin(column, spec.not_shown_codes) | (blank & ~abstained)
        answers = column[~abstained & ~not_shown]

        note = ""
        if name in spec.levels:
            codes = tuple(spec.levels[name])
            unknown = sorted({_as_text(v) for v in answers} - {_as_text(c) for c in codes})
            if unknown:
                raise _SkipError(f"codes {unknown} are not on the declared scale {_show(codes)}")
            declared = True
        else:
            seen = _sorted_codes(answers.unique())
            if seen is None:
                raise _SkipError("non-numeric codes: declare their order with --levels")
            if len(seen) < 2:
                raise _SkipError(f"fewer than two distinct answers ({_show(seen)})")
            codes = seen
            declared = False
            if _has_gap(codes):
                note = (
                    f"codes {_show(codes)} skip a value: a level nobody chose, or a code to "
                    "declare as abstention / not shown?"
                )

        rank = {_as_text(code): position for position, code in enumerate(codes, start=1)}
        levels = pd.Series(pd.NA, index=column.index, dtype="Int64", name=name)
        levels[abstained] = 0
        levels[answers.index] = [rank[_as_text(value)] for value in answers]
        item = OrdinalItem(
            name=name,
            codes=codes,
            declared=declared,
            n_votes=len(answers),
            n_abstentions=int(abstained.sum()),
            n_not_shown=int(not_shown.sum()),
            note=note,
        )
        return item, levels


class _SkipError(Exception):
    """A question that cannot enter a vote base, with the reason."""


# Result --------------------------------------------------------------------------------


@dataclass
class SplitSurvey:
    """The bases of a split survey, joinable on ``id_name``.

    Attributes:
        id_name: The identifier column, present in every base.
        metadata: The identifier and the columns describing the respondents.
        bases: For each number of levels ``K`` with at least one question, the identifier
            and the questions encoded 0..K (NA when not asked).
        items: The questions kept, in the order of the source.
        skipped: The questions left out, with the reason.
        spec: The specification that produced the split.
    """

    id_name: str
    metadata: pd.DataFrame
    bases: dict[int, pd.DataFrame]
    items: list[OrdinalItem]
    skipped: dict[str, str]
    spec: SurveySpec

    def joined(self, k: int) -> pd.DataFrame:
        """The vote base with ``K = k`` levels, joined one to one with the metadata."""
        return self.bases[k].merge(self.metadata, on=self.id_name, how="left", validate="1:1")

    def to_vote_matrix(self, k: int, *, drop_inactive: bool = False) -> Any:
        """Build the :class:`VoteMatrix` of one base, for co-clustering with SELENE.

        The scores of the scale are the ranks ``1..K``: a block model only uses the
        order of the levels.

        Args:
            k: The number of levels of the base.
            drop_inactive: Leave out the respondents without any answer in this base.
        """
        from civiqhub_selene.blocks import OrdinalScale, VoteMatrix

        base = self.bases[k]
        values = base.drop(columns=self.id_name)
        if drop_inactive:
            values = values[(values > 0).any(axis=1).to_numpy()]
        shown = values.notna().to_numpy()
        levels = values.fillna(0).to_numpy(dtype=np.int64)
        return VoteMatrix.from_arrays(
            levels,
            shown=shown,
            voters=base.loc[values.index, self.id_name].tolist(),
            items=[str(column) for column in values.columns],
            scale=OrdinalScale(tuple(range(1, k + 1))),
        )

    def report(self) -> str:
        lines = [f"{len(self.metadata)} respondents, identifier {self.id_name!r}"]
        lines.append(f"metadata: {len(self.metadata.columns) - 1} columns")
        for k in self.spec.k_values:
            names = [item.name for item in self.items if item.k == k]
            lines.append(f"K={k}: {len(names)} questions" + (f"  {names}" if names else ""))
        for item in self.items:
            if item.note:
                lines.append(f"warning {item.name}: {item.note}")
        for name, reason in self.skipped.items():
            lines.append(f"left out {name}: {reason}")
        return "\n".join(lines)

    # Files -------------------------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Write ``metadata.csv``, one ``votes_K{k}.csv`` per base, ``items.csv`` and a manifest."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        files = {"metadata": "metadata.csv"}
        self.metadata.to_csv(out / files["metadata"], index=False)
        for k, base in self.bases.items():
            files[f"K{k}"] = f"votes_K{k}.csv"
            base.to_csv(out / files[f"K{k}"], index=False)
        items = pd.DataFrame(
            [
                {
                    "item": item.name,
                    "K": item.k,
                    "codes_low_to_high": json.dumps(_jsonable(item.codes)),
                    "declared": item.declared,
                    "n_votes": item.n_votes,
                    "n_abstentions": item.n_abstentions,
                    "n_not_shown": item.n_not_shown,
                    "note": item.note,
                }
                for item in self.items
            ]
        )
        items.to_csv(out / "items.csv", index=False)
        manifest = {
            "id_name": self.id_name,
            "files": files,
            "skipped": self.skipped,
            "spec": _jsonable(asdict(self.spec)),
            "encoding": {"1..K": "rank on the scale", "0": "abstention", "empty": "not shown"},
        }
        (out / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        return out

    @staticmethod
    def load_joined(directory: str | Path, k: int) -> pd.DataFrame:
        """Read back one saved vote base joined with the metadata (for post-processing)."""
        out = Path(directory)
        manifest = json.loads((out / MANIFEST).read_text())
        id_name = manifest["id_name"]
        text = {id_name: str}
        votes = pd.read_csv(out / manifest["files"][f"K{k}"], dtype=text)
        votes = votes.astype({c: "Int64" for c in votes.columns if c != id_name})
        metadata = pd.read_csv(out / manifest["files"]["metadata"], dtype=text)
        return votes.merge(metadata, on=id_name, how="left", validate="1:1")


# Helpers -------------------------------------------------------------------------------


def _as_text(value: Any) -> str:
    """Text form of a code or identifier; 3.0 and 3 give "3"."""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _isin(column: pd.Series, codes: Iterable[Any]) -> pd.Series:
    wanted = {_as_text(code) for code in codes}
    if not wanted:
        return pd.Series(False, index=column.index)
    return column.map(lambda v: not pd.isna(v) and _as_text(v) in wanted).astype(bool)


def _sorted_codes(values: Iterable[Any]) -> tuple[Any, ...] | None:
    """Numeric codes in increasing order, or None if one of them is not a number."""
    numbers = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        numbers.append(int(number) if number.is_integer() else number)
    return tuple(sorted(set(numbers)))


def _has_gap(codes: Sequence[Any]) -> bool:
    if not all(isinstance(code, int) for code in codes):
        return False
    return codes[-1] - codes[0] + 1 != len(codes)


def _check_present(columns: Iterable[str], table: pd.DataFrame, kind: str) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{kind} columns not in the table: {missing}")


def _show(codes: Iterable[Any]) -> str:
    return "[" + ", ".join(_as_text(code) for code in codes) + "]"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# Reading, inspecting -------------------------------------------------------------------


def read_table(
    path: str | Path,
    *,
    sep: str | None = None,
    encoding: str | None = None,
    decimal: str = ".",
) -> Any:
    """Read CSV, Parquet, Excel, Stata or SAS by extension.

    ``decimal=","`` reads French numbers such as ``0,666`` as numbers.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".tsv"}:
        return pd.read_csv(path, sep=sep, engine="python", encoding=encoding, decimal=decimal)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".dta":
        return pd.read_stata(path, convert_categoricals=False)
    if suffix in {".sas7bdat", ".xpt"}:
        return pd.read_sas(path, encoding=encoding)
    raise ValueError(f"unknown table format {suffix!r}")


def inspect(table: pd.DataFrame, *, max_codes: int = 12) -> pd.DataFrame:
    """One row per column: type, blanks, number of distinct values and the values."""
    rows = []
    for column in table.columns:
        values = table[column].dropna().unique()
        codes = _sorted_codes(values)
        shown = codes if codes is not None else tuple(sorted(map(str, values)))
        rows.append(
            {
                "column": column,
                "dtype": str(table[column].dtype),
                "blanks": int(table[column].isna().sum()),
                "n_distinct": len(values),
                "values": _show(shown[:max_codes]) + (" ..." if len(shown) > max_codes else ""),
            }
        )
    return pd.DataFrame(rows)


# Synthetic demonstration ---------------------------------------------------------------


def synthetic_survey(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """A synthetic survey with the pitfalls of real ones. No real respondent.

    Codes 8 and 9 are "refusal" and "don't know"; ``Q4b`` is only asked when ``Q4a`` is
    2 (a filter); ``Q5c`` has a level nobody chose; ``Q10`` has ten levels, outside the
    default range; ``comment`` is free text.
    """
    rng = np.random.default_rng(seed)

    def answers(k: int, low: int = 1) -> np.ndarray:
        codes = rng.integers(low, low + k, size=n).astype(float)
        codes[rng.random(n) < 0.06] = 9
        codes[rng.random(n) < 0.02] = 8
        return codes

    survey = pd.DataFrame(
        {
            "IDENT": [f"R{number:05d}" for number in rng.permutation(n) + 1],
            "sex": rng.integers(1, 3, size=n),
            "age_group": rng.integers(1, 6, size=n),
            "region": rng.choice(["IDF", "ARA", "NAQ", "OCC", "HDF"], size=n),
            "Q2a": answers(2),
            "Q2b": answers(2, low=0),
            "Q3a": answers(3),
            "Q4a": answers(4),
            "Q5a": answers(5),
            "Q5b": answers(5),
            "Q6a": answers(6),
            "Q7a": answers(7, low=0),
            "Q10": answers(10, low=0),
        }
    )
    survey["Q4b"] = np.where(survey["Q4a"] == 2, answers(4), np.nan)
    q5c = answers(5)
    q5c[q5c == 4] = 5  # nobody answers 4
    survey["Q5c"] = q5c
    survey["comment"] = rng.choice(["", "trop long", "ok"], size=n)
    return survey


# Command line --------------------------------------------------------------------------


def _code(text: str) -> Any:
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("table", nargs="?", help="CSV, Parquet, Excel, Stata or SAS file")
    parser.add_argument("--demo", action="store_true", help="use a synthetic survey")
    parser.add_argument("--inspect", action="store_true", help="describe the columns and stop")
    parser.add_argument("--out", help="directory of the bases")
    parser.add_argument("--id", dest="id_column", help="identifier column (default: row number)")
    side = parser.add_mutually_exclusive_group()
    side.add_argument("--metadata", nargs="+", help="metadata columns; the rest are questions")
    side.add_argument("--votes", nargs="+", help="question columns; the rest are metadata")
    side.add_argument("--vote-pattern", help="regular expression on the names of the questions")
    parser.add_argument("--abstention-codes", nargs="*", default=[], type=_code)
    parser.add_argument("--not-shown-codes", nargs="*", default=[], type=_code)
    parser.add_argument("--blank-is-abstention", action="store_true")
    parser.add_argument("--levels", help='JSON file {"question": [codes from low to high]}')
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=7)
    parser.add_argument("--sep", help="CSV separator (default: guessed)")
    parser.add_argument("--encoding", help="file encoding, for example latin-1")
    parser.add_argument("--decimal", default=".", help='decimal mark, "," for 0,666')
    parser.add_argument(
        "--rows", help='keep the rows matching a pandas query, e.g. "annee == 2024"'
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.demo:
        table = synthetic_survey()
        if not (args.metadata or args.votes or args.vote_pattern):
            args.id_column = args.id_column or "IDENT"
            args.vote_pattern = r"^Q\d+[a-z]?$"
            args.abstention_codes = args.abstention_codes or [8, 9]
    elif args.table:
        table = read_table(args.table, sep=args.sep, encoding=args.encoding, decimal=args.decimal)
    else:
        _parser().error("give a table or --demo")
    if args.rows:
        before = len(table)
        table = table.query(args.rows).reset_index(drop=True)
        print(f"rows kept by {args.rows!r}: {len(table)} of {before}")

    if args.inspect:
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(inspect(table).to_string(index=False))
        return 0
    if not args.out:
        _parser().error("--out is required to write the bases")

    levels = {}
    if args.levels:
        levels = {k: tuple(v) for k, v in json.loads(Path(args.levels).read_text()).items()}
    spec = SurveySpec(
        id_column=args.id_column,
        metadata=tuple(args.metadata) if args.metadata else None,
        votes=tuple(args.votes) if args.votes else None,
        vote_pattern=args.vote_pattern,
        abstention_codes=tuple(args.abstention_codes),
        not_shown_codes=tuple(args.not_shown_codes),
        blank_is_abstention=args.blank_is_abstention,
        levels=levels,
        k_min=args.k_min,
        k_max=args.k_max,
    )
    split = SurveySplitter(spec).split(table)
    out = split.save(args.out)
    print(split.report())
    print(f"written to {out}/")

    try:
        for k in split.bases:
            print(f"K={k}:", split.to_vote_matrix(k).shape, "VoteMatrix (voters, items)")
    except ImportError:
        print("civiqhub_selene[blocks] not installed: vote matrices not built")
    return 0


if __name__ == "__main__":
    sys.exit(main())
