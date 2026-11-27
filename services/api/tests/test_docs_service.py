# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""The page ``docs/service.md`` follows the settings and the results of the service.

Each setting of ``coclustering.single`` has its row, with its default, in the table of
the section "From the settings to the estimator", and each field of the value and of
the table of co-clusters of ``result/1`` and ``result/2`` has its row, with the results
that carry it, in the table of the section "From the engine result to the contract
result". A field added to the contract without its row fails here.
"""

import json
import pathlib
from typing import Any

import pytest
from pydantic import BaseModel

from selene_api.adapters.coclustering_single import (
    _COCLUSTER_FIELDS,
    _COCLUSTER_FIELDS_V2,
    _VALUE_FIELDS,
    _VALUE_FIELDS_V2,
)
from selene_api.schemas import (
    CoclusteringSingleV1Parameters,
    CoclusteringSingleV2Parameters,
    ThresholdSettings,
)

PAGE = pathlib.Path(__file__).resolve().parents[3] / "docs" / "service.md"
SETTINGS = "## From the settings to the estimator"
RESULTS = "## From the engine result to the contract result"
# The fields that result/2 adds in place of those of one scale (result_document_v2).
BY_SCALE_VALUE = ("scales", "scale_index")
BY_SCALE_TABLE = ("vote_shares_by_scale",)


def _section(title: str) -> str:
    text = PAGE.read_text(encoding="utf-8")
    start = text.index(title + "\n")
    end = text.find("\n## ", start + len(title))
    return text[start : end if end >= 0 else len(text)]


def _rows(title: str) -> list[list[str]]:
    """The cells of the first table of a section, without its header and its rule."""
    lines: list[str] = []
    for line in _section(title).splitlines():
        if line.startswith("|"):
            lines.append(line)
        elif lines:
            break
    rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]
    return rows[2:]


def _code(cell: str) -> str:
    """The first code span of a cell, without its backticks."""
    start = cell.index("`") + 1
    return cell[start : cell.index("`", start)]


def _settings() -> dict[str, str]:
    """Each setting of the table, by its name, with the text of its default."""
    return {_code(row[0]): _code(row[1]) for row in _rows(SETTINGS)}


def _defaults(model: type[BaseModel], prefix: str = "") -> dict[str, Any]:
    """The default of each field of a model of settings, as JSON reads it."""
    defaults: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        value = field.get_default(call_default_factory=True)
        defaults[prefix + name] = value.model_dump() if isinstance(value, BaseModel) else value
    return defaults


@pytest.mark.parametrize(
    "defaults",
    [
        _defaults(CoclusteringSingleV1Parameters),
        _defaults(CoclusteringSingleV2Parameters),
        _defaults(ThresholdSettings, "thresholds."),
    ],
    ids=["schema-1", "schema-2", "thresholds"],
)
def test_each_setting_has_its_row_with_its_default(defaults: dict[str, Any]) -> None:
    rows = _settings()
    for name, default in defaults.items():
        assert name in rows, f"docs/service.md has no row for the setting {name}"
        assert json.loads(rows[name]) == default, f"the default of {name} on the page"


def test_the_table_of_the_settings_names_no_other_setting() -> None:
    known = (
        _defaults(CoclusteringSingleV2Parameters).keys()
        | _defaults(ThresholdSettings, "thresholds.").keys()
    )
    assert set(_settings()) <= known


def _results() -> dict[str, set[str]]:
    """Each field of the table of the results, by its path, with the results it is in."""
    rows: dict[str, set[str]] = {}
    for row in _rows(RESULTS):
        path = _code(row[0])
        rows[path] = {part.strip() for part in row[1].split(",")}
    return rows


def _expected() -> dict[str, set[str]]:
    """The fields of result/1 and result/2, by their path, from the adapter."""
    expected: dict[str, set[str]] = {}
    fields = [
        ("value.", _VALUE_FIELDS, "1"),
        ("value.", (*_VALUE_FIELDS_V2, *BY_SCALE_VALUE), "2"),
        ("value.coclusters.", _COCLUSTER_FIELDS, "1"),
        ("value.coclusters.", (*_COCLUSTER_FIELDS_V2, *BY_SCALE_TABLE), "2"),
    ]
    for prefix, names, result in fields:
        for name in names:
            if name != "coclusters":
                expected.setdefault(prefix + name, set()).add(result)
    return expected


def test_each_field_of_the_results_has_its_row() -> None:
    rows = _results()
    for path, results in _expected().items():
        assert path in rows, f"docs/service.md has no row for {path}"
        assert rows[path] == results, f"the results that carry {path} on the page"


def test_the_table_of_the_results_names_no_other_field() -> None:
    assert set(_results()) == set(_expected())
