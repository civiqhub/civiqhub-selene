# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""The adapter of coclustering.single/1 calls the real engine and maps it faithfully."""

import ast
import copy
import json
import pathlib
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from jsonschema import Draft202012Validator

import make_fixtures
import selene_contract
from selene_api.adapters import InstanceLimits, adapter_for, kinds
from selene_api.adapters.coclustering_single import CoclusteringSingleV1, work_of
from selene_api.errors import ApiError

Reader = Callable[[str], Any]
Validator = Callable[[str], Draft202012Validator]
ADAPTER = CoclusteringSingleV1()
WIDE = InstanceLimits(max_voters=20_000, max_items=5_000, max_cells=2_000_000, max_work=10**13)
SOURCE = pathlib.Path(ADAPTER.prepare.__code__.co_filename).resolve().parents[1]


def raw(submission: dict[str, Any]) -> bytes:
    return json.dumps(submission).encode()


def reference() -> dict[str, Any]:
    return make_fixtures.reference_submission()


def test_the_transfer_object_defaults_are_those_of_the_contract() -> None:
    from selene_api.schemas import CoclusteringSingleV1Parameters

    assert CoclusteringSingleV1Parameters().model_dump() == make_fixtures.contract_defaults()


def test_the_only_kind_is_coclustering_single_1() -> None:
    assert kinds() == {"coclustering.single": [1]}
    assert isinstance(adapter_for("coclustering.single", 1), CoclusteringSingleV1)
    assert adapter_for("coclustering.single", 2) is None
    assert adapter_for("coclustering.mixed", 1) is None


class TestPreparation:
    def test_the_fingerprint_is_the_one_the_contract_defines(self) -> None:
        submission = reference()
        task = ADAPTER.prepare(raw(submission), WIDE)
        assert task.content_fingerprint == make_fixtures.content_fingerprint(submission)

    def test_the_payload_is_normalised(self) -> None:
        submission = reference()
        shuffled = copy.deepcopy(submission)
        shuffled["payload"]["votes"].reverse()
        shuffled["payload"]["exposure"]["shown_without_vote"].reverse()
        first = ADAPTER.prepare(raw(submission), WIDE)
        second = ADAPTER.prepare(raw(shuffled), WIDE)
        assert first.payload == second.payload
        assert first.content_fingerprint == second.content_fingerprint
        assert first.payload["parameters"] == make_fixtures.contract_defaults() | {
            "n_groups": 2,
            "n_blocks": 2,
            "initialization": "spectral",
        }

    def test_an_integer_tolerance_is_the_same_content(self) -> None:
        submission = reference()
        submission["payload"]["parameters"]["tolerance"] = 0
        as_float = copy.deepcopy(submission)
        as_float["payload"]["parameters"]["tolerance"] = 0.0
        assert (
            ADAPTER.prepare(raw(submission), WIDE).content_fingerprint
            == ADAPTER.prepare(raw(as_float), WIDE).content_fingerprint
        )

    def test_the_key_and_reference_are_kept_out_of_the_payload(self) -> None:
        task = ADAPTER.prepare(raw(reference()), WIDE)
        assert task.submission_key == "test-only:reference:0001"
        assert task.caller_reference == "test-only/generation-1"
        assert set(task.worker_input()) == {"kind", "schema_version", "payload"}

    @pytest.mark.parametrize(
        ("name", "code"),
        [
            ("exposure-unknown", "unsupported_exposure"),
            ("duplicate-vote", "invalid_request"),
            ("duplicate-item", "invalid_request"),
            ("vote-row-out-of-range", "invalid_request"),
            ("listed-cell-out-of-range", "invalid_request"),
            ("voted-cell-listed", "invalid_request"),
            ("score-off-scale", "invalid_request"),
            ("score-as-text", "invalid_request"),
            ("engine-scale", "invalid_request"),
            ("voter-looks-like-email", "invalid_request"),
            ("duplicate-voter", "invalid_request"),
            ("unknown-field", "invalid_request"),
            ("missing-seed", "invalid_request"),
        ],
    )
    def test_each_refusal_has_its_code(self, read_fixture: Reader, name: str, code: str) -> None:
        case = read_fixture(f"coclustering-single-1/invalid/{name}.json")
        with pytest.raises(ApiError) as caught:
            ADAPTER.prepare(raw(case["body"]), WIDE)
        assert (caught.value.status, caught.value.code) == (422, code)
        assert "someone@example.org" not in json.dumps(caught.value.body())

    def test_a_refusal_names_the_place_not_the_value(self) -> None:
        submission = reference()
        submission["payload"]["votes"][4][2] = 7
        with pytest.raises(ApiError) as caught:
            ADAPTER.prepare(raw(submission), WIDE)
        assert caught.value.body()["error"]["fields"] == [
            {"path": "payload.votes[4][2]", "issue": "above the maximum"}
        ]

    @pytest.mark.parametrize(
        ("limits", "path"),
        [
            (
                InstanceLimits(max_voters=11, max_items=5_000, max_cells=10**6, max_work=10**13),
                "payload.voters",
            ),
            (
                InstanceLimits(max_voters=20_000, max_items=5, max_cells=10**6, max_work=10**13),
                "payload.items",
            ),
            (
                InstanceLimits(max_voters=20_000, max_items=5_000, max_cells=71, max_work=10**13),
                "payload",
            ),
            (
                InstanceLimits(max_voters=20_000, max_items=5_000, max_cells=10**6, max_work=10),
                "payload.parameters",
            ),
        ],
    )
    def test_the_limits_refuse_before_any_computation(
        self, limits: InstanceLimits, path: str
    ) -> None:
        with pytest.raises(ApiError) as caught:
            ADAPTER.prepare(raw(reference()), limits)
        assert caught.value.code == "limit_exceeded"
        assert [issue.path for issue in caught.value.fields] == [path]

    def test_the_budget_counts_every_factor(self) -> None:
        parameters = {"n_starts": 6, "max_iterations": 200, "n_groups": 2, "n_blocks": 3}
        assert work_of(12, 6, parameters) == 6 * 200 * 72 * 6 * 5


class TestComputation:
    def test_the_adapter_equals_a_local_engine_call(self) -> None:
        submission = reference()
        local = make_fixtures.result_document(make_fixtures.engine_estimate(submission))
        task = ADAPTER.prepare(raw(submission), WIDE)
        assert ADAPTER.compute(task.payload) == local

    def test_the_result_follows_the_contract(self, schema_validator: Validator) -> None:
        task = ADAPTER.prepare(raw(reference()), WIDE)
        schema_validator("CoclusteringSingleV1Result").validate(ADAPTER.compute(task.payload))

    def test_the_result_matches_the_stored_reference(self, read_fixture: Reader) -> None:
        submission = reference()
        task = ADAPTER.prepare(raw(submission), WIDE)
        job = {
            "content_fingerprint": task.content_fingerprint,
            "state": "succeeded",
            "result": ADAPTER.compute(task.payload),
        }
        expected = read_fixture("coclustering-single-1/reference.expected.json")
        assert selene_contract.compare_with_reference(job, expected) == []

    def test_the_order_of_rows_and_columns_is_kept(self) -> None:
        task = ADAPTER.prepare(raw(reference()), WIDE)
        value = ADAPTER.compute(task.payload)["value"]
        assert value["voters"] == task.payload["voters"]
        assert value["items"] == [item["id"] for item in task.payload["items"]]
        assert value["scores"] == [-2, -1, 0, 1, 2]

    def test_no_vote_is_an_explained_absence(self, read_fixture: Reader) -> None:
        submission = read_fixture("coclustering-single-1/no-votes.submission.json")
        result = ADAPTER.compute(ADAPTER.prepare(raw(submission), WIDE).payload)
        assert result["available"] is False
        assert result["unavailable"]["reason"] == "no_votes"
        assert "value" not in result

    def test_too_few_items_for_the_shape_is_an_explained_absence(self) -> None:
        submission = reference()
        submission["payload"]["parameters"]["n_blocks"] = 10
        result = ADAPTER.compute(ADAPTER.prepare(raw(submission), WIDE).payload)
        assert (result["available"], result["unavailable"]["reason"]) == (False, "not_enough_data")

    def test_the_same_seed_gives_the_same_result(self) -> None:
        payload = ADAPTER.prepare(raw(reference()), WIDE).payload
        assert ADAPTER.compute(payload) == ADAPTER.compute(copy.deepcopy(payload))


class TestCellStates:
    """Neutral vote, shown without a vote and not shown stay three different things."""

    def matrix(self, submission: dict[str, Any]) -> Any:
        return ADAPTER.votes_of(ADAPTER.prepare(raw(submission), WIDE).payload)

    def test_a_neutral_vote_is_a_vote_at_the_middle_rank(self) -> None:
        votes = self.matrix(reference())
        row, column = 2, 1  # vtr-03 voted 0 on itm-b
        assert votes.levels[row, column] == 3
        assert votes.shown[row, column]
        assert votes.voted[row, column]

    def test_scores_become_ranks_through_the_engine_scale(self) -> None:
        votes = self.matrix(reference())
        for row, column, score in make_fixtures.reference_submission()["payload"]["votes"]:
            assert votes.levels[row, column] == score + 3

    def test_a_listed_cell_without_a_vote_is_shown_without_a_vote(self) -> None:
        votes = self.matrix(reference())
        row, column = 1, 5  # vtr-02 saw itm-f and did not vote
        assert votes.shown[row, column]
        assert not votes.voted[row, column]

    def test_a_cell_neither_voted_nor_listed_is_not_shown(self) -> None:
        votes = self.matrix(reference())
        row, column = 2, 4  # vtr-03 never saw itm-e
        assert not votes.shown[row, column]
        assert votes.levels[row, column] == 0

    def test_complete_exposure_shows_every_cell(self) -> None:
        submission = reference()
        submission["payload"]["exposure"] = {"mode": "complete"}
        votes = self.matrix(submission)
        assert votes.shown.all()
        assert int(votes.voted.sum()) == len(submission["payload"]["votes"])

    def test_the_absence_of_a_vote_is_never_read_as_exposure(self) -> None:
        submission = reference()
        submission["payload"]["exposure"]["shown_without_vote"] = []
        votes = self.matrix(submission)
        np.testing.assert_array_equal(votes.shown, votes.voted)


def test_the_server_side_never_imports_numpy() -> None:
    code = (
        "import sys; import selene_api.app, selene_api.server; "
        "from selene_api.adapters import InstanceLimits; "
        "from selene_api.adapters.coclustering_single import CoclusteringSingleV1; "
        "limits = InstanceLimits(20000, 5000, 2000000, 10**13); "
        "CoclusteringSingleV1().prepare(open(sys.argv[1], 'rb').read(), limits); "
        "print('numpy' in sys.modules)"
    )
    fixture = make_fixtures.FIXTURES / "coclustering-single-1" / "reference.submission.json"
    answer = subprocess.run(
        [sys.executable, "-c", code, str(fixture)], capture_output=True, text=True, check=True
    )
    assert answer.stdout.strip() == "False"


def test_no_formula_of_the_engine_is_copied_into_the_service() -> None:
    """Only the adapters and the worker import the engine, and never NumPy directly."""
    for path in sorted(SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        relative = path.relative_to(SOURCE).as_posix()
        if any(module.startswith("civiqhub_selene.blocks") for module in modules):
            assert relative in {"adapters/coclustering_single.py"}, relative
        if relative != "worker.py":
            assert "numpy" not in modules, relative
