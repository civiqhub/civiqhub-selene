# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Describe the groups of a co-clustering with the metadata of the respondents.

The groups are built from the votes only; the metadata (age, income, sex, region,
occupation...) did not enter the fit. They play the part of *supplementary variables*,
and the description follows the automatic description of classes of Lebart, Morineau
and Piron, *Statistique exploratoire multidimensionnelle* (Dunod, 1995), section 2.3.2:
each group is compared with the whole population, variable by variable, and the
differences are ranked by their *test value*.

Every statistic is weighted by the membership ``theta[i, q]`` of respondent ``i`` to
group ``q``. For a single-membership model (LBM) it is 0 or 1; for a mixed-membership
model (MM-OLBM) it is the estimated row of theta, which sums to 1. One code path serves
both: a mixed respondent counts in several groups, in proportion to its memberships.

Kinds of variables, declared by the user (nothing is guessed):

- ``continuous``: weighted mean, deciles D1, Q1, median, Q3, D9, missing rate;
- ``semicontinuous``: the same, plus the share of zeros, with the quantiles and a
  ``mean_positive`` computed on the positive values only (income, wealth: a mass at 0
  and a heavy tail); ``mean`` keeps the zeros;
- ``categorical``: share of each category in the group and overall;
- ``ordinal``: like categorical, with an order kept in tables and figures.

Test values (Lebart et al., 2.3.2):

- continuous: ``t = (mean_q - mean) / sqrt(s2 / n_q * (N - n_q) / (N - 1))``, the gap of
  the group mean in standard deviations of a mean drawn without replacement;
- category: with integer counts (single membership), the hypergeometric probability of
  a count at least as extreme, written as a standard normal quantile, signed; with
  fractional counts (mixed membership), the normal approximation of the same law. The
  weighted versions are an extension, not in the book.

A test value is a ranking score: |t| > 2 flags a characteristic feature. Many values are
computed at once, so they are not formal tests.

Usage::

    python examples/profile_clusters.py --demo --out profile_demo           # hard groups
    python examples/profile_clusters.py --demo --soft --out profile_demo    # mixed
    python examples/profile_clusters.py --demo --weighted --out profile_demo  # weights

    python examples/profile_clusters.py --metadata metadata.csv \\
        --membership groups.csv --spec spec.json --out profile

``groups.csv`` has the identifier and either one column ``group`` (single membership)
or one column per group with the memberships (mixed). ``spec.json``::

    {"id": "IDENT",
     "variables": {"age": "continuous", "income": "semicontinuous", "sex": "categorical",
                   "diploma": {"kind": "ordinal", "order": [1, 2, 3, 4]}},
     "labels": {"sex": {"1": "Man", "2": "Woman"}},
     "weight": "poids",
     "min_count": 10}

Needs pandas, SciPy and Matplotlib.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

Kind = Literal["continuous", "semicontinuous", "categorical", "ordinal"]
KINDS: tuple[str, ...] = ("continuous", "semicontinuous", "categorical", "ordinal")
QUANTILES: dict[str, float] = {"D1": 0.1, "Q1": 0.25, "median": 0.5, "Q3": 0.75, "D9": 0.9}
MISSING = "(missing)"
OTHER = "(other)"
OVERALL = "Overall"


# Memberships ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Membership:
    """Degree of membership of each respondent to each group.

    Attributes:
        ids: Respondent identifiers, as text, unique.
        groups: Names of the groups.
        weights: Shape (n, Q), non-negative, each row sums to 1.
    """

    ids: tuple[str, ...]
    groups: tuple[str, ...]
    weights: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=float)
        if weights.shape != (len(self.ids), len(self.groups)):
            raise ValueError(
                f"weights of shape {weights.shape} for {len(self.ids)} ids "
                f"and {len(self.groups)} groups"
            )
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("the identifiers repeat")
        if (weights < 0).any() or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("memberships are non-negative and each row sums to 1")
        object.__setattr__(self, "weights", weights)

    @property
    def is_hard(self) -> bool:
        """True if every respondent belongs to exactly one group."""
        return bool(np.isin(self.weights, (0.0, 1.0)).all())

    @classmethod
    def from_labels(cls, ids: Iterable[Any], labels: Iterable[Any]) -> Membership:
        """Single membership, from one group label per respondent."""
        ids = tuple(_text(i) for i in ids)
        labels = [_text(label) for label in labels]
        groups = tuple(sorted(set(labels), key=_natural))
        position = {group: q for q, group in enumerate(groups)}
        weights = np.zeros((len(ids), len(groups)))
        weights[np.arange(len(ids)), [position[label] for label in labels]] = 1.0
        return cls(ids, groups, weights)

    @classmethod
    def from_weights(
        cls, ids: Iterable[Any], weights: Any, groups: Sequence[Any] | None = None
    ) -> Membership:
        """Mixed membership, from a matrix of shape (n, Q)."""
        weights = np.asarray(weights, dtype=float)
        names = groups if groups is not None else range(1, weights.shape[1] + 1)
        return cls(tuple(_text(i) for i in ids), tuple(_text(g) for g in names), weights)

    @classmethod
    def from_coclustering(cls, result: Any, *, hard: bool = False) -> Membership:
        """From a SELENE ``Coclustering`` result: its voters and group memberships.

        Args:
            result: The result of an estimate, or its value, with ``voters``,
                ``group_index`` and ``group_memberships``.
            hard: Keep only the main group of each voter. The memberships of a
                single-membership model can be posterior degrees, not 0 or 1: pass
                ``hard=True`` to describe its partition.
        """
        result = getattr(result, "value", result)  # a Result, or its value
        names = [str(q + 1) for q in range(result.group_memberships.shape[1])]
        if hard:
            return cls.from_labels(result.voters, [names[q] for q in result.group_index])
        return cls.from_weights(result.voters, result.group_memberships, names)

    @classmethod
    def read_csv(cls, path: str | Path, id_column: str) -> Membership:
        """Read ``id, group`` (single) or ``id, <one column per group>`` (mixed)."""
        table = pd.read_csv(path, dtype={id_column: str})
        if "group" in table.columns:
            return cls.from_labels(table[id_column], table["group"])
        columns = [c for c in table.columns if c != id_column]
        return cls.from_weights(table[id_column], table[columns].to_numpy(), columns)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.weights, index=pd.Index(self.ids), columns=list(self.groups))


# Specification -------------------------------------------------------------------------


@dataclass(frozen=True)
class VariableSpec:
    """How to describe one variable.

    Attributes:
        name: Column of the metadata.
        kind: One of ``continuous``, ``semicontinuous``, ``categorical``, ``ordinal``.
        order: For an ordinal variable, its codes from low to high.
        labels: Readable names of the codes, for example ``{"1": "Man"}``.
    """

    name: str
    kind: Kind
    order: tuple[Any, ...] | None = None
    labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: kind {self.kind!r} is not one of {KINDS}")
        if self.kind == "ordinal" and not self.order:
            raise ValueError(f"{self.name}: an ordinal variable needs its order")

    @property
    def is_numeric(self) -> bool:
        return self.kind in ("continuous", "semicontinuous")


@dataclass(frozen=True)
class ProfileSpec:
    """The variables to describe and the reading rules.

    Attributes:
        id_column: Identifier column of the metadata, the one of the memberships.
        variables: The variables, in the order of the reports.
        min_count: Categories with fewer respondents overall are merged into "(other)".
        missing_as_category: Keep "no answer" as a category of its own.
        threshold: |test value| from which a feature counts as characteristic.
        weight_column: Column of survey weights (for example calibration weights), or
            None. Weights are rescaled to mean 1; every share, mean, quantile and count
            then describes the weighted population.
    """

    id_column: str
    variables: tuple[VariableSpec, ...]
    min_count: int = 10
    missing_as_category: bool = True
    threshold: float = 2.0
    weight_column: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ProfileSpec:
        labels = raw.get("labels", {})
        variables = []
        for name, entry in raw["variables"].items():
            entry = {"kind": entry} if isinstance(entry, str) else dict(entry)
            order = tuple(entry["order"]) if entry.get("order") else None
            names = {str(k): str(v) for k, v in labels.get(name, {}).items()}
            variables.append(VariableSpec(name, entry["kind"], order, names))
        return cls(
            id_column=raw["id"],
            variables=tuple(variables),
            min_count=int(raw.get("min_count", 10)),
            missing_as_category=bool(raw.get("missing_as_category", True)),
            threshold=float(raw.get("threshold", 2.0)),
            weight_column=raw.get("weight"),
        )

    @staticmethod
    def suggest(metadata: pd.DataFrame, id_column: str, max_levels: int = 12) -> dict[str, str]:
        """A first guess of the kinds, to check and correct by hand before use."""
        guess = {}
        for column in metadata.columns:
            if column == id_column or column in ("poids", "weight"):
                continue
            values = metadata[column].dropna()
            numeric = pd.api.types.is_numeric_dtype(values)
            if numeric and values.nunique() > max_levels:
                guess[column] = "semicontinuous" if (values == 0).mean() > 0.05 else "continuous"
            else:
                guess[column] = "categorical"
        return guess


# Profiling -----------------------------------------------------------------------------


class ClusterProfiler:
    """Compute the profile of each group against the whole population."""

    def __init__(self, spec: ProfileSpec) -> None:
        self.spec = spec

    def fit(self, membership: Membership, metadata: pd.DataFrame) -> ClusterProfile:
        """Join the memberships and the metadata, then describe every variable.

        The reference population is the respondents present in both: a respondent
        left out of the clustering is left out of the overall figures too.
        """
        spec = self.spec
        meta = metadata.copy()
        meta[spec.id_column] = meta[spec.id_column].map(_text)
        if meta[spec.id_column].duplicated().any():
            raise ValueError(f"the identifier {spec.id_column!r} repeats in the metadata")
        missing = [v.name for v in spec.variables if v.name not in meta.columns]
        if missing:
            raise ValueError(f"variables not in the metadata: {missing}")
        weights = membership.frame()
        common = weights.index.intersection(pd.Index(meta[spec.id_column]))
        if common.empty:
            raise ValueError("no identifier in common between memberships and metadata")
        meta = meta.set_index(spec.id_column).loc[common]
        weights = weights.loc[common]
        dropped = (len(membership.ids) - len(common), len(metadata) - len(common))

        theta = weights.to_numpy()
        empty = [
            group
            for group, mass in zip(weights.columns, theta.sum(axis=0), strict=True)
            if mass == 0
        ]
        if empty:
            raise ValueError(
                f"groups without any member after joining memberships and metadata: {empty}; "
                "drop them from the memberships"
            )
        hard = membership.is_hard
        s = np.ones(len(common))
        if spec.weight_column is not None:
            if spec.weight_column not in meta.columns:
                raise ValueError(f"no weight column {spec.weight_column!r} in the metadata")
            s = pd.to_numeric(meta[spec.weight_column], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(s).all() or (s <= 0).any():
                raise ValueError(f"the weights {spec.weight_column!r} must be positive numbers")
            s = s / s.mean()
        weighted = spec.weight_column is not None
        rows: list[dict[str, Any]] = []
        association: list[dict[str, Any]] = []
        crosstabs: dict[str, pd.DataFrame] = {}
        quantiles: dict[str, pd.DataFrame] = {}
        for variable in spec.variables:
            column = meta[variable.name]
            if variable.is_numeric:
                new_rows, eta, table = _describe_numeric(
                    variable, column, theta, s, weights.columns
                )
                quantiles[variable.name] = table
                association.append(
                    {
                        "variable": variable.name,
                        "kind": variable.kind,
                        "measure": "eta",
                        "value": eta,
                    }
                )
            else:
                counts, indicator, rows_theta, rows_s = _crosstab(
                    variable, column, theta, s, weights.columns, spec
                )
                new_rows, cramer = _describe_categorical(
                    variable, counts, hard and not weighted, indicator, rows_theta, rows_s
                )
                crosstabs[variable.name] = counts
                association.append(
                    {
                        "variable": variable.name,
                        "kind": variable.kind,
                        "measure": "cramers_v",
                        "value": cramer,
                    }
                )
            rows.extend(new_rows)

        mass = (s[:, None] * theta).sum(axis=0)
        sizes = pd.DataFrame(
            {
                "group": weights.columns,
                "mass": mass,
                "share": mass / s.sum(),
                "n_eff": [_kish(s * theta[:, q]) for q in range(theta.shape[1])],
            }
        )
        if weighted:
            sizes.insert(2, "respondents", theta.sum(axis=0))
        if not hard:
            sizes["mean_membership"] = [
                theta[:, q][theta[:, q] > 0].mean() for q in range(theta.shape[1])
            ]
        return ClusterProfile(
            table=pd.DataFrame(rows),
            sizes=sizes,
            association=pd.DataFrame(association).sort_values("value", ascending=False),
            crosstabs=crosstabs,
            quantiles=quantiles,
            spec=spec,
            hard=hard,
            n=len(common),
            dropped=dropped,
            weighted=weighted,
        )


def _describe_numeric(
    variable: VariableSpec,
    column: pd.Series,
    theta: np.ndarray,
    s: np.ndarray,
    groups: pd.Index,
) -> tuple[list[dict[str, Any]], float, pd.DataFrame]:
    values = pd.to_numeric(column, errors="coerce").to_numpy(dtype=float)
    present = ~np.isnan(values)
    x, w_all, s_x = values[present], theta[present], s[present]
    # weighted mean and variance (divisor: sum of weights, N without weights, as in Lebart)
    mean = _wmean(x, s_x)
    variance = _wmean(np.square(x - mean), s_x)
    positive = x > 0 if variable.kind == "semicontinuous" else np.ones_like(x, dtype=bool)

    def summary(weights: np.ndarray, all_weights: np.ndarray) -> dict[str, float]:
        out = {"mean": _wmean(x, weights), "missing": 1 - weights.sum() / all_weights.sum()}
        if variable.kind == "semicontinuous":
            out["share_zero"] = 1 - _wmean(positive.astype(float), weights)
            out["mean_positive"] = _wmean(x[positive], weights[positive])
        for name, p in QUANTILES.items():
            out[name] = _wquantile(x[positive], weights[positive], p)
        return out

    overall = summary(s_x, s)
    rows, table = [], {OVERALL: overall}
    between = 0.0
    for q, group in enumerate(groups):
        w = w_all[:, q] * s_x
        n_q = w.sum()
        own = summary(w, theta[:, q] * s)
        table[group] = own
        between += n_q * (own["mean"] - mean) ** 2
        for statistic, value in own.items():
            reference = overall[statistic]
            t = math.nan
            if statistic == "mean":
                t = _test_value_weighted(x, w_all[:, q], s_x)
            elif statistic == "share_zero":
                t = -_test_value_weighted(positive.astype(float), w_all[:, q], s_x)
            rows.append(_row(group, variable, statistic, value, reference, t, _kish(w), lift=False))
    eta = math.sqrt(between / (s_x.sum() * variance)) if variance > 0 else math.nan
    return rows, eta, pd.DataFrame(table).T


def _crosstab(
    variable: VariableSpec,
    column: pd.Series,
    theta: np.ndarray,
    s: np.ndarray,
    groups: pd.Index,
    spec: ProfileSpec,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted counts, categories by groups, rare categories merged (on respondent
    counts); also the 0/1 indicators of the categories, and the memberships and survey
    weights of the rows kept."""
    codes = column.map(lambda v: MISSING if pd.isna(v) else _text(v))
    keep = np.ones(len(codes), dtype=bool)
    if not spec.missing_as_category:
        keep = (codes != MISSING).to_numpy()
    codes = codes[keep]
    totals = codes.value_counts()
    rare = [c for c in totals.index if totals[c] < spec.min_count and c != MISSING]
    if len(rare) > 1:  # merging a single category would only rename it
        codes = codes.where(~codes.isin(rare), OTHER)
    indicator = pd.get_dummies(codes).astype(float)
    counts = pd.DataFrame(
        indicator.to_numpy().T @ (theta[keep] * s[keep, None]),
        index=indicator.columns,
        columns=groups,
    )
    order = _category_order(variable, list(counts.index))
    counts = counts.loc[order]
    indicator = indicator[order].to_numpy()
    counts.index = [variable.labels.get(c, c) for c in counts.index]
    counts.index.name = variable.name
    return counts, indicator, theta[keep], s[keep]


def _describe_categorical(
    variable: VariableSpec,
    counts: pd.DataFrame,
    exact: bool,
    indicator: np.ndarray,
    theta: np.ndarray,
    s: np.ndarray,
) -> tuple[list[dict[str, Any]], float]:
    """Shares of each category. ``exact``: integer counts (single membership, no survey
    weights), for the hypergeometric test value."""
    n_total = counts.to_numpy().sum()
    n_category = counts.sum(axis=1)
    n_group = counts.sum(axis=0)
    rows = []
    for q, group in enumerate(counts.columns):
        n_q, n_eff = n_group[group], _kish(theta[:, q] * s)
        for m, category in enumerate(counts.index):
            n_qm = counts.at[category, group]
            share, reference = n_qm / n_q, n_category[category] / n_total
            if exact:
                t = _test_value_hypergeometric(
                    round(n_qm), round(n_total), round(n_category[category]), round(n_q)
                )
            else:
                t = _test_value_weighted(indicator[:, m], theta[:, q], s)
            row = _row(group, variable, category, share, reference, t, n_eff, lift=True)
            low, high = _wilson(share, n_eff)
            row.update(ci_low=low, ci_high=high)
            rows.append(row)
    return rows, _cramers_v(counts.to_numpy())


def _row(
    group: str,
    variable: VariableSpec,
    level: str,
    value: float,
    reference: float,
    test_value: float,
    n_eff: float,
    *,
    lift: bool,
) -> dict[str, Any]:
    return {
        "group": group,
        "variable": variable.name,
        "kind": variable.kind,
        "level": level,
        "group_value": value,
        "overall_value": reference,
        "diff": value - reference,
        "lift": value / reference if lift and reference > 0 else math.nan,
        "test_value": test_value,
        "n_eff": n_eff,
    }


# Statistics ----------------------------------------------------------------------------


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    total = w.sum()
    return float(np.dot(x, w) / total) if total > 0 else math.nan


def _wquantile(x: np.ndarray, w: np.ndarray, p: float) -> float:
    """Weighted quantile: the smallest x whose weighted distribution function reaches p.

    With weights 0 or 1, this is numpy's ``inverted_cdf`` quantile of the group.
    """
    keep = w > 0
    if not keep.any():
        return math.nan
    order = np.argsort(x[keep], kind="stable")
    sorted_x, cumulative = x[keep][order], np.cumsum(w[keep][order])
    position = np.searchsorted(cumulative, p * cumulative[-1] - 1e-12, side="left")
    return float(sorted_x[min(position, len(sorted_x) - 1)])


def _kish(w: np.ndarray) -> float:
    """Kish effective sample size, (sum w)^2 / sum w^2."""
    squares = np.square(w).sum()
    return float(w.sum() ** 2 / squares) if squares > 0 else 0.0


def _test_value_weighted(x: np.ndarray, theta: np.ndarray, s: np.ndarray | None = None) -> float:
    """Test value of a group mean of x, under random permutation of the memberships.

    Null hypothesis of Lebart et al. 2.3.2: the group is drawn at random, without
    replacement, among the N respondents. With a = s * (x - m), where m is the mean of
    x weighted by the survey weights s, the group gap is carried by the linear statistic
    T = sum theta_i a_i, of mean 0 and, under a random permutation of theta,
    Var(T) = sum (theta_i - mean theta)^2 * sum (a_j - mean a)^2 / (N - 1).

    Without survey weights and with memberships 0 or 1, T / sqrt(Var T) is exactly the
    test value (mean_q - mean) / sqrt(s2 / n_q * (N - n_q) / (N - 1)) of section 2.3.2 a.
    With memberships it extends it exactly; with survey weights, T is the first-order
    (linearised) form of the gap of the weighted group mean, an extension not in the book.
    """
    n = len(x)
    s = np.ones(n) if s is None else s
    if n < 2 or s.sum() <= 0:
        return math.nan
    a = s * (x - np.dot(s, x) / s.sum())
    spread_a = np.square(a - a.mean()).sum()
    spread_theta = np.square(theta - theta.mean()).sum()
    if spread_a <= 0 or spread_theta <= 0:
        return math.nan
    return float(np.dot(theta, a) / math.sqrt(spread_a * spread_theta / (n - 1)))


def _test_value_hypergeometric(k: int, n: int, n_category: int, n_group: int) -> float:
    """Lebart et al. 2.3.2 b: hypergeometric tail as a signed normal quantile."""
    if n_group in (0, n) or n_category in (0, n):
        return math.nan
    law = stats.hypergeom(n, n_category, n_group)
    expected = n_group * n_category / n
    if k > expected:
        return float(stats.norm.isf(max(law.sf(k - 1), 1e-300)))
    if k < expected:
        return -float(stats.norm.isf(max(law.cdf(k), 1e-300)))
    return 0.0


def _wilson(p: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _cramers_v(table: np.ndarray) -> float:
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    if min(table.shape) < 2:
        return math.nan
    n = table.sum()
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    chi2 = ((table - expected) ** 2 / expected).sum()
    return float(math.sqrt(chi2 / (n * (min(table.shape) - 1))))


# Result --------------------------------------------------------------------------------


@dataclass
class ClusterProfile:
    """The description of the groups, as one long table and its by-products.

    Attributes:
        table: One row per group, variable and category (or statistic): ``group_value``,
            ``overall_value``, ``diff``, ``lift``, ``test_value``, ``n_eff`` and, for
            categories, the Wilson interval ``ci_low``, ``ci_high``.
        sizes: Mass, share and Kish effective size of each group.
        association: Cramér's V (categorical) or correlation ratio eta = sqrt(eta2)
            (numeric), both on the scale of a correlation, so that they compare.
        crosstabs: Weighted counts, categories by groups, per categorical variable.
        quantiles: Group and overall statistics per numeric variable.
        hard: True for single memberships.
        n: Respondents in the reference population.
        dropped: Respondents only in the memberships, only in the metadata.
        weighted: True if the figures use survey weights.
    """

    table: pd.DataFrame
    sizes: pd.DataFrame
    association: pd.DataFrame
    crosstabs: dict[str, pd.DataFrame]
    quantiles: dict[str, pd.DataFrame]
    spec: ProfileSpec
    hard: bool
    n: int
    dropped: tuple[int, int]
    weighted: bool = False

    @property
    def groups(self) -> list[str]:
        return list(self.sizes["group"])

    def characteristic(self, group: str, top: int | None = None) -> pd.DataFrame:
        """Features of a group with |test value| above the threshold, strongest first.

        Kept: categories, and the mean and share of zeros of numeric variables.
        """
        rows = self.table[
            (self.table["group"] == group)
            & self.table["level"].isin(_ranked_levels(self.table))
            & (self.table["test_value"].abs() >= self.spec.threshold)
        ]
        over = rows[rows.test_value > 0].sort_values("test_value", ascending=False)
        under = rows[rows.test_value < 0].sort_values("test_value")
        if top is not None:
            over, under = over.head(top), under.head(top)
        return pd.concat([over, under])

    def label(self, group: str, top: int = 3) -> str:
        """A short automatic label: the most over-represented features."""
        rows = self.characteristic(group)
        rows = rows[rows.test_value > 0].head(top)
        phrases = [_phrase(row, short=True) for row in rows.itertuples()]
        return ", ".join(phrases) or "no salient feature"

    def identity_cards(self, top: int = 8) -> str:
        """One Markdown card per group: size, then over- and under-represented features."""
        lines = [
            "# Group profiles",
            "",
            f"{self.n} respondents; {'single' if self.hard else 'mixed'} membership; "
            + (f"weighted by {self.spec.weight_column!r}; " if self.weighted else "")
            + f"features with |test value| >= {self.spec.threshold}.",
        ]
        if any(self.dropped):
            lines.append(
                f"Left out: {self.dropped[0]} respondents without metadata, "
                f"{self.dropped[1]} without membership."
            )
        for size in self.sizes.itertuples():
            rows = self.characteristic(size.group)
            lines += [
                "",
                f"## Group {size.group}: {self.label(size.group)}",
                "",
                f"Share {size.share:.1%}, mass {size.mass:.1f}"
                + ("" if size.mass >= 30 else " (small group: read with care)"),
            ]
            for title, part in (
                ("Over-represented", rows[rows.test_value > 0]),
                ("Under-represented", rows[rows.test_value < 0]),
            ):
                if len(part):
                    lines += ["", f"{title}:"]
                    lines += [f"- {_phrase(row)}" for row in part.head(top).itertuples()]
        lines += ["", "## Variables ranked by association with the groups", ""]
        lines += [
            f"- {row.variable} ({row.measure} = {row.value:.3f})"
            for row in self.association.itertuples()
        ]
        return "\n".join(lines) + "\n"

    def save(self, directory: str | Path) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.table.to_csv(out / "profile.csv", index=False)
        self.sizes.to_csv(out / "sizes.csv", index=False)
        self.association.to_csv(out / "association.csv", index=False)
        (out / "identity_cards.md").write_text(self.identity_cards())
        return out


def _ranked_levels(table: pd.DataFrame) -> set[str]:
    numeric_levels = {"mean", "share_zero"}
    categories = set(table.loc[~table["kind"].isin(["continuous", "semicontinuous"]), "level"])
    return numeric_levels | categories


def _phrase(row: Any, short: bool = False) -> str:
    if row.kind in ("continuous", "semicontinuous") and row.level == "mean":
        text = (
            f"{row.variable} mean {_number(row.group_value)} (overall {_number(row.overall_value)})"
        )
    elif row.level == "share_zero":
        text = f"{row.variable} = 0 for {row.group_value:.0%} (overall {row.overall_value:.0%})"
    else:
        text = (
            f"{row.variable} = {row.level}: {row.group_value:.0%} "
            f"(overall {row.overall_value:.0%}, {row.diff * 100:+.0f} pts)"
        )
    if short:
        return text.split(" (")[0]
    return f"{text}, t = {row.test_value:+.1f}"


# Figures -------------------------------------------------------------------------------


def _pyplot() -> Any:
    import matplotlib.pyplot as plt

    return plt


def plot_heatmap(
    profile: ClusterProfile,
    variables: Sequence[str] | None = None,
    value: Literal["test_value", "diff"] = "test_value",
) -> Any:
    """Groups by features, colored by test value (all variables) or gap in points."""
    plt = _pyplot()
    table = profile.table[profile.table["level"].isin(_ranked_levels(profile.table))]
    if value == "diff":
        table = table[~table["kind"].isin(["continuous", "semicontinuous"])]
    if variables is not None:
        table = table[table["variable"].isin(variables)]
    table = table.assign(feature=table["variable"] + ": " + table["level"].astype(str))
    features = list(dict.fromkeys(table["feature"]))
    matrix = table.pivot(index="feature", columns="group", values=value).loc[
        features, profile.groups
    ]
    shown = matrix * (100 if value == "diff" else 1)
    limit = np.nanmax(np.abs(shown.to_numpy())) or 1.0
    fig, ax = plt.subplots(figsize=(1.2 * len(profile.groups) + 4, 0.28 * len(features) + 1.5))
    image = ax.imshow(shown.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(profile.groups)), [f"Group {g}" for g in profile.groups])
    ax.set_yticks(range(len(features)), features, fontsize=8)
    for (i, j), v in np.ndenumerate(shown.to_numpy()):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
    label = "test value" if value == "test_value" else "gap with overall (points)"
    fig.colorbar(image, ax=ax, label=label, shrink=0.6)
    fig.tight_layout()
    return fig


def plot_stacked(profile: ClusterProfile, variable: str) -> Any:
    """Share of each category, one 100 % bar per group and one for the whole population."""
    plt = _pyplot()
    counts = profile.crosstabs[variable]
    shares = (counts / counts.sum(axis=0)).T
    shares.loc[OVERALL] = counts.sum(axis=1) / counts.to_numpy().sum()
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(shares) + 1.5))
    left = np.zeros(len(shares))
    colours = plt.get_cmap("viridis" if _is_ordinal(profile, variable) else "tab20")(
        np.linspace(0, 1, len(shares.columns))
    )
    labels = [f"Group {g}" for g in profile.groups] + [OVERALL]
    for category, colour in zip(shares.columns, colours, strict=True):
        ax.barh(labels, shares[category], left=left, color=colour, label=str(category))
        left += shares[category].to_numpy()
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share")
    ax.set_title(variable)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_quantiles(profile: ClusterProfile, variable: str, log: bool = False) -> Any:
    """Box D1-Q1-median-Q3-D9 per group, mean as a diamond, overall median dashed."""
    plt = _pyplot()
    table = profile.quantiles[variable]
    rows = [*profile.groups, OVERALL]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(rows) + 1.5))
    for y, name in enumerate(rows):
        s = table.loc[name]
        ax.plot([s.D1, s.D9], [y, y], color="0.4", lw=1)
        ax.barh(y, s.Q3 - s.Q1, left=s.Q1, height=0.5, color="C0" if name != OVERALL else "0.7")
        ax.plot([s["median"]] * 2, [y - 0.25, y + 0.25], color="black", lw=2)
        ax.plot(s.get("mean_positive", s["mean"]), y, marker="D", color="C3", ms=5)
    ax.axvline(table.loc[OVERALL, "median"], color="0.3", ls="--", lw=1)
    ax.set_yticks(range(len(rows)), [f"Group {g}" for g in profile.groups] + [OVERALL])
    ax.invert_yaxis()
    kind = next(v.kind for v in profile.spec.variables if v.name == variable)
    title = variable + (
        " (positive values; share of zeros aside)" if kind == "semicontinuous" else ""
    )
    if kind == "semicontinuous":
        for y, name in enumerate(rows):
            ax.annotate(
                f"0: {table.loc[name, 'share_zero']:.0%}",
                (1.01, y),
                xycoords=("axes fraction", "data"),
                va="center",
                fontsize=8,
            )
    if log:
        ax.set_xscale("log")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_association(profile: ClusterProfile) -> Any:
    """Variables ranked by Cramér's V (categorical) or correlation ratio eta (numeric)."""
    plt = _pyplot()
    data = profile.association.sort_values("value")
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(data) + 1.2))
    colours = ["C0" if m == "cramers_v" else "C1" for m in data["measure"]]
    ax.barh(data["variable"], data["value"], color=colours)
    ax.set_xlabel("Cramér's V (blue) or correlation ratio η (orange)")
    ax.set_xlim(0, max(0.05, float(data["value"].max()) * 1.1))
    fig.tight_layout()
    return fig


# Correspondence analysis ---------------------------------------------------------------


@dataclass
class CorrespondenceAnalysis:
    """Correspondence analysis of the groups by categories table.

    Attributes:
        eigenvalues: Principal inertias, at most Q - 1.
        rows: Principal coordinates of the groups.
        columns: Principal coordinates of the categories.
    """

    eigenvalues: np.ndarray
    rows: pd.DataFrame
    columns: pd.DataFrame

    @property
    def inertia_shares(self) -> np.ndarray:
        return self.eigenvalues / self.eigenvalues.sum()

    @classmethod
    def fit(cls, profile: ClusterProfile, variables: Sequence[str] | None = None) -> Any:
        """Analyse the juxtaposed table: groups in rows, the categories of the chosen
        categorical variables in columns (Lebart et al., chapter 1.3)."""
        names = list(variables or profile.crosstabs)
        blocks = [profile.crosstabs[name].T.add_prefix(f"{name}: ") for name in names]
        table = pd.concat(blocks, axis=1)
        table = table.loc[:, table.sum(axis=0) > 0]
        p = table.to_numpy() / table.to_numpy().sum()
        r, c = p.sum(axis=1), p.sum(axis=0)
        residuals = (p - np.outer(r, c)) / np.sqrt(np.outer(r, c))
        u, singular, vt = np.linalg.svd(residuals, full_matrices=False)
        keep = singular > 1e-12
        u, singular, vt = u[:, keep], singular[keep], vt[keep]
        rows = (u * singular) / np.sqrt(r)[:, None]
        columns = (vt.T * singular) / np.sqrt(c)[:, None]
        axes = [f"F{k + 1}" for k in range(len(singular))]
        return cls(
            eigenvalues=singular**2,
            rows=pd.DataFrame(rows, index=table.index, columns=axes),
            columns=pd.DataFrame(columns, index=table.columns, columns=axes),
        )

    def plot(self, max_labels: int = 40) -> Any:
        """Symmetric map of the first plane (one axis only when there are two groups)."""
        plt = _pyplot()
        fig, ax = plt.subplots(figsize=(8, 7))
        two_d = self.rows.shape[1] >= 2
        x_rows, x_cols = self.rows["F1"].to_numpy(), self.columns["F1"].to_numpy()
        y_rows = self.rows["F2"].to_numpy() if two_d else np.zeros(len(self.rows))
        y_cols = self.columns["F2"].to_numpy() if two_d else np.zeros(len(self.columns))
        strongest = (self.columns.iloc[:, : 2 if two_d else 1] ** 2).sum(axis=1)
        shown = strongest.sort_values(ascending=False).index[:max_labels]
        ax.scatter(x_cols, y_cols, s=10, color="0.6")
        for name in shown:
            position = self.columns.index.get_loc(name)
            ax.annotate(name, (x_cols[position], y_cols[position]), fontsize=7, color="0.3")
        ax.scatter(x_rows, y_rows, s=80, color="C3", marker="s")
        for position, group in enumerate(self.rows.index):
            ax.annotate(
                f"Group {group}",
                (x_rows[position], y_rows[position]),
                fontsize=10,
                color="C3",
                weight="bold",
            )
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        share = self.inertia_shares
        ax.set_xlabel(f"F1 ({share[0]:.0%} of inertia)")
        if two_d:
            ax.set_ylabel(f"F2 ({share[1]:.0%} of inertia)")
        fig.tight_layout()
        return fig


# Helpers -------------------------------------------------------------------------------


def _text(value: Any) -> str:
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _number(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ") if abs(value) >= 1000 else f"{value:.3g}"


def _natural(text: str) -> tuple[int, Any]:
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)


def _category_order(variable: VariableSpec, present: list[str]) -> list[str]:
    if variable.order:
        declared = [_text(code) for code in variable.order]
        head = [c for c in declared if c in present]
    else:
        head = sorted((c for c in present if c not in (MISSING, OTHER)), key=_natural)
    tail = [c for c in present if c not in head and c not in (MISSING, OTHER)]
    return head + tail + [c for c in (OTHER, MISSING) if c in present]


def _is_ordinal(profile: ClusterProfile, variable: str) -> bool:
    return any(v.name == variable and v.kind == "ordinal" for v in profile.spec.variables)


# Synthetic demonstration ---------------------------------------------------------------


def synthetic_profile_data(
    n: int = 900, seed: int = 0, soft: bool = False, weighted: bool = False
) -> tuple[pd.DataFrame, Membership, ProfileSpec]:
    """Synthetic metadata and three groups with planted profiles. No real respondent.

    Group 1 is older, more often retired and female; group 2 younger, more often
    managers, in Ile-de-France and richer; group 3 is close to the whole population.
    """
    rng = np.random.default_rng(seed)
    group = rng.choice(3, size=n, p=[0.35, 0.3, 0.35])
    age = np.clip(
        rng.normal(np.choose(group, [58, 36, 47]), np.choose(group, [12, 10, 15])), 18, 95
    )
    sex = np.where(rng.random(n) < np.choose(group, [0.62, 0.45, 0.5]), "2", "1")
    regions = ["IDF", "ARA", "NAQ", "OCC", "HDF", "GES", "PDL", "BRE", "COR"]
    base = np.array([0.19, 0.12, 0.09, 0.09, 0.09, 0.08, 0.06, 0.05, 0.005])
    base /= base.sum()
    idf = np.where(group == 1, 0.35, base[0])
    region = [rng.choice(regions, p=np.r_[p0, base[1:] / base[1:].sum() * (1 - p0)]) for p0 in idf]
    occupations = ["1", "2", "3", "4", "5", "6", "7", "8"]
    weights = {
        0: [0.01, 0.05, 0.08, 0.12, 0.14, 0.08, 0.45, 0.07],
        1: [0.01, 0.06, 0.32, 0.22, 0.17, 0.10, 0.02, 0.10],
        2: [0.02, 0.06, 0.16, 0.18, 0.18, 0.14, 0.18, 0.08],
    }
    occupation = [rng.choice(occupations, p=weights[g]) for g in group]
    income = rng.lognormal(np.choose(group, [7.3, 7.9, 7.5]), 0.5)
    income[rng.random(n) < np.choose(group, [0.08, 0.05, 0.15])] = 0.0
    wealth = rng.lognormal(np.choose(group, [11.5, 10.8, 10.9]), 1.2)
    wealth[rng.random(n) < 0.2] = 0.0
    diploma = rng.choice(["1", "2", "3", "4"], size=n)
    diploma = np.where((group == 1) & (rng.random(n) < 0.4), "4", diploma)
    income[rng.random(n) < 0.04] = np.nan  # no answer
    metadata = pd.DataFrame(
        {
            "IDENT": [f"R{k:05d}" for k in range(1, n + 1)],
            "age": age.round(),
            "income": income.round(),
            "wealth": wealth.round(-2),
            "sex": sex,
            "region": region,
            "occupation": occupation,
            "diploma": diploma,
            "weight": np.round(rng.lognormal(0.0, 0.3, n), 4),
        }
    )
    if soft:
        concentration = np.full((n, 3), 0.3)
        concentration[np.arange(n), group] = 4.0
        theta = np.vstack([rng.dirichlet(row) for row in concentration])
        membership = Membership.from_weights(metadata["IDENT"], theta, ["1", "2", "3"])
    else:
        membership = Membership.from_labels(metadata["IDENT"], group + 1)
    spec = ProfileSpec.from_dict(
        {
            "id": "IDENT",
            "variables": {
                "age": "continuous",
                "income": "semicontinuous",
                "wealth": "semicontinuous",
                "sex": "categorical",
                "region": "categorical",
                "occupation": "categorical",
                "diploma": {"kind": "ordinal", "order": ["1", "2", "3", "4"]},
            },
            "labels": {
                "sex": {"1": "man", "2": "woman"},
                "occupation": {
                    "1": "farmer",
                    "2": "craft/trade",
                    "3": "manager",
                    "4": "intermediate",
                    "5": "employee",
                    "6": "worker",
                    "7": "retired",
                    "8": "inactive",
                },
                "diploma": {"1": "none", "2": "secondary", "3": "bac", "4": "higher"},
            },
            "min_count": 10,
            "weight": "weight" if weighted else None,
        }
    )
    return metadata, membership, spec


# Command line --------------------------------------------------------------------------


def write_figures(profile: ClusterProfile, directory: str | Path) -> list[Path]:
    """Every figure of the profile, as PNG files."""
    plt = _pyplot()
    out = Path(directory) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "association": plot_association(profile),
        "heatmap_test_values": plot_heatmap(profile),
        "heatmap_gaps": plot_heatmap(profile, value="diff"),
    }
    for variable in profile.crosstabs:
        figures[f"bars_{variable}"] = plot_stacked(profile, variable)
    for variable in profile.quantiles:
        kind = next(v.kind for v in profile.spec.variables if v.name == variable)
        figures[f"quantiles_{variable}"] = plot_quantiles(
            profile, variable, log=kind == "semicontinuous"
        )
    if profile.crosstabs and len(profile.groups) >= 2:
        analysis = CorrespondenceAnalysis.fit(profile)
        figures["correspondence_analysis"] = analysis.plot()
    paths = []
    for name, figure in figures.items():
        path = out / f"{name}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--demo", action="store_true", help="use synthetic data")
    parser.add_argument("--soft", action="store_true", help="demo with mixed memberships")
    parser.add_argument("--weighted", action="store_true", help="demo with survey weights")
    parser.add_argument("--metadata", help="CSV of the metadata, with the identifier")
    parser.add_argument("--membership", help="CSV: id + group, or id + one column per group")
    parser.add_argument("--spec", help="JSON specification of the variables")
    parser.add_argument("--suggest", action="store_true", help="print a guess of the kinds")
    parser.add_argument("--id", help="identifier column, for --suggest without --spec")
    parser.add_argument("--out", help="directory of the tables and figures")
    parser.add_argument("--no-figures", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.demo:
        metadata, membership, spec = synthetic_profile_data(soft=args.soft, weighted=args.weighted)
    else:
        if not args.metadata:
            parser.error("give --metadata, or --demo")
        if args.suggest:
            raw = json.loads(Path(args.spec).read_text()) if args.spec else {}
            id_column = args.id or raw.get("id", "row_number")
            table = pd.read_csv(args.metadata, dtype={id_column: str})
            print(json.dumps(ProfileSpec.suggest(table, id_column), indent=2))
            return 0
        if not (args.membership and args.spec):
            parser.error("give --membership and --spec")
        spec = ProfileSpec.from_dict(json.loads(Path(args.spec).read_text()))
        metadata = pd.read_csv(args.metadata, dtype={spec.id_column: str})
        membership = Membership.read_csv(args.membership, spec.id_column)
    if not args.out:
        parser.error("--out is required")

    profile = ClusterProfiler(spec).fit(membership, metadata)
    out = profile.save(args.out)
    print(profile.identity_cards())
    if not args.no_figures:
        import matplotlib

        matplotlib.use("Agg")
        paths = write_figures(profile, out)
        print(f"{len(paths)} figures in {out / 'figures'}/")
    print(f"tables in {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
