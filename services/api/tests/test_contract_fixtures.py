# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""The fixtures follow the contract, and the reference results still come from the engine."""

import copy
import pathlib
from collections.abc import Callable
from typing import Any

import pytest
from jsonschema import Draft202012Validator

import make_fixtures
from civiqhub_selene.blocks import LatentBlockModel, RandomStart
from civiqhub_selene.blocks.responses import OrdinalProbit
from civiqhub_selene.types import UnavailableReason

Validator = Callable[[str], Draft202012Validator]
Reader = Callable[[str], Any]

INVALID = sorted(
    path.stem
    for path in (make_fixtures.FIXTURES / "coclustering-single-1" / "invalid").glob("*.json")
)


def test_the_fixtures_are_up_to_date() -> None:
    """Regenerate them with ``python services/api/contract/make_fixtures.py``."""
    stale, extra = make_fixtures.stale_fixtures()
    assert not stale, [path.name for path in stale]
    assert not extra, [path.name for path in extra]


class TestFixtureComparison:
    """The staleness check tolerates the last bits of a float, nothing else."""

    def test_floats_within_the_tolerance_match(self) -> None:
        assert make_fixtures.equivalent({"a": [1.0 + 1e-12, 2]}, {"a": [1.0, 2]})

    def test_floats_outside_the_tolerance_differ(self) -> None:
        assert not make_fixtures.equivalent({"a": 1.0 + 1e-6}, {"a": 1.0})

    def test_integers_texts_and_booleans_match_exactly(self) -> None:
        assert not make_fixtures.equivalent([0, 1], [1, 0])
        assert not make_fixtures.equivalent("a", "b")
        assert not make_fixtures.equivalent(True, 1)
        assert not make_fixtures.equivalent({"a": 1}, {"a": 1, "b": 2})

    def test_the_same_partition_under_other_names_matches(self) -> None:
        stored = {"group_index": [0, 0, 1], "block_index": [0, 1], "objective": -1.5}
        found = {"group_index": [1, 1, 0], "block_index": [1, 0], "objective": -1.5}
        assert make_fixtures.equivalent({"expected": found}, {"expected": stored})

    def test_another_partition_differs(self) -> None:
        stored = {"group_index": [0, 0, 1], "block_index": [0, 1]}
        found = {"group_index": [0, 1, 1], "block_index": [0, 1]}
        assert not make_fixtures.equivalent(found, stored)
        assert make_fixtures.first_difference(found, stored) == (
            "$.group_index[1]: computed 1, stored 0"
        )

    def test_the_environment_of_the_computation_is_not_compared(self) -> None:
        found = {"engine_version": "0.2.0", "dependencies": {"numpy": "2.6.0"}, "x": 1}
        stored = {"engine_version": "0.1.0", "dependencies": {"numpy": "2.5.5"}, "x": 1}
        assert make_fixtures.equivalent(found, stored)


@pytest.mark.parametrize("name", ["reference", "no-votes"])
def test_the_submissions_follow_the_contract(
    schema_validator: Validator, read_fixture: Reader, name: str
) -> None:
    submission = read_fixture(f"coclustering-single-1/{name}.submission.json")
    schema_validator("JobSubmission").validate(submission)


@pytest.mark.parametrize("name", INVALID)
def test_each_negative_case_says_whether_the_schema_refuses_it(
    schema_validator: Validator, read_fixture: Reader, name: str
) -> None:
    case = read_fixture(f"coclustering-single-1/invalid/{name}.json")
    valid = schema_validator("JobSubmission").is_valid(case["body"])
    assert valid is case["schema_valid"]
    assert case["expect"]["status"] == 422


@pytest.mark.parametrize(
    ("name", "schema"),
    [
        ("job-running", "Job"),
        ("job-succeeded", "Job"),
        ("job-failed-timeout", "Job"),
        ("error-busy", "ErrorResponse"),
        ("error-conflict", "ErrorResponse"),
        ("error-invalid", "ErrorResponse"),
        ("error-job-not-found", "ErrorResponse"),
        ("error-unauthenticated", "ErrorResponse"),
    ],
)
def test_the_examples_follow_the_contract(
    schema_validator: Validator, read_fixture: Reader, name: str, schema: str
) -> None:
    schema_validator(schema).validate(read_fixture(f"examples/{name}.json"))


def test_the_reference_holds_every_cell_state(read_fixture: Reader) -> None:
    payload = read_fixture("coclustering-single-1/reference.submission.json")["payload"]
    scores = [score for _row, _column, score in payload["votes"]]
    assert 0 in scores, "a neutral vote"
    assert {-2, -1, 1, 2} <= set(scores)
    assert payload["exposure"]["mode"] == "listed"
    assert payload["exposure"]["shown_without_vote"], "cells shown without a vote"
    cells = len(payload["voters"]) * len(payload["items"])
    shown = len(payload["votes"]) + len(payload["exposure"]["shown_without_vote"])
    assert shown < cells, "cells that were not shown"


def test_the_reference_finds_the_planted_collective(read_fixture: Reader) -> None:
    expected = read_fixture("coclustering-single-1/reference.expected.json")["expected"]
    assert make_fixtures.same_partition(expected["group_index"], make_fixtures.PLANTED_GROUPS)
    assert make_fixtures.same_partition(expected["block_index"], make_fixtures.PLANTED_BLOCKS)


def test_the_fixtures_are_marked_test_only(fixtures_dir: pathlib.Path) -> None:
    for path in fixtures_dir.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        if not path.name.startswith("error-"):  # error bodies hold no data
            assert "TEST ONLY" in text or "test-only" in text or "TESTONLY" in text, path.name
        if path.stem != "voter-looks-like-email":  # the case that must be refused
            assert "@" not in text, f"{path.name} holds what looks like an e-mail address"


def test_the_contract_defaults_are_the_engine_defaults() -> None:
    defaults = make_fixtures.contract_defaults()
    engine = LatentBlockModel()
    assert defaults["n_groups"] == engine.n_groups
    assert defaults["n_blocks"] == engine.n_blocks
    assert defaults["n_starts"] == engine.n_starts
    assert defaults["max_iterations"] == engine.max_iterations
    assert defaults["tolerance"] == engine.tolerance
    assert defaults["response_model"] == "ordinal_probit"
    assert type(engine.response_model) is OrdinalProbit
    assert defaults["initialization"] == "random"
    assert type(engine.initialization) is RandomStart


def test_the_reasons_of_absence_are_the_engine_reasons(openapi: dict[str, Any]) -> None:
    result = openapi["components"]["schemas"]["CoclusteringSingleV1Result"]
    reasons = result["properties"]["unavailable"]["properties"]["reason"]["enum"]
    assert reasons == [reason.value for reason in UnavailableReason]


class TestContentFingerprint:
    """The fingerprint follows the normalisation the contract defines."""

    def base(self) -> dict[str, Any]:
        return make_fixtures.reference_submission()

    def test_it_ignores_the_order_of_votes_and_listed_cells(self) -> None:
        submission = self.base()
        shuffled = copy.deepcopy(submission)
        shuffled["payload"]["votes"].reverse()
        shuffled["payload"]["exposure"]["shown_without_vote"].reverse()
        assert make_fixtures.content_fingerprint(shuffled) == make_fixtures.content_fingerprint(
            submission
        )

    def test_it_fills_the_defaults(self) -> None:
        submission = self.base()
        explicit = copy.deepcopy(submission)
        explicit["payload"]["parameters"] = make_fixtures.contract_defaults() | {
            "initialization": "spectral"
        }
        implicit = copy.deepcopy(submission)
        implicit["payload"]["parameters"] = {"initialization": "spectral"}
        assert make_fixtures.content_fingerprint(explicit) == make_fixtures.content_fingerprint(
            implicit
        )

    def test_it_ignores_the_key_and_the_caller_reference(self) -> None:
        submission = self.base()
        other = copy.deepcopy(submission)
        other["submission_key"] = "another-key-0001"
        del other["caller_reference"]
        assert make_fixtures.content_fingerprint(other) == make_fixtures.content_fingerprint(
            submission
        )

    @pytest.mark.parametrize(
        "change",
        [
            lambda body: body["payload"]["voters"].reverse(),
            lambda body: body["payload"]["items"][0].update(revision="itm-a-rev-2"),
            lambda body: body["payload"].update(seed=1),
            lambda body: body["payload"]["scope"].update(project="prj-other"),
            lambda body: body["payload"]["votes"][0].__setitem__(2, 0),
            lambda body: body["payload"]["exposure"]["shown_without_vote"].pop(),
        ],
    )
    def test_it_follows_the_content(self, change: Callable[[dict[str, Any]], None]) -> None:
        submission = self.base()
        changed = copy.deepcopy(submission)
        change(changed)
        assert make_fixtures.content_fingerprint(changed) != make_fixtures.content_fingerprint(
            submission
        )
