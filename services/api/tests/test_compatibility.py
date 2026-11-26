# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""The compatibility check refuses what breaks a client of /v1 and accepts what only adds."""

import copy
from collections.abc import Callable
from typing import Any

import pytest

import check_compatibility

Change = Callable[[dict[str, Any]], None]


def schemas(document: dict[str, Any]) -> dict[str, Any]:
    found: dict[str, Any] = document["components"]["schemas"]
    return found


def bumped(change: Change, version: str = "1.99.0") -> Change:
    def apply(document: dict[str, Any]) -> None:
        change(document)
        document["info"]["version"] = version

    return apply


def remove_operation(document: dict[str, Any]) -> None:
    del document["paths"]["/v1/jobs/{job_id}"]["delete"]


def remove_status(document: dict[str, Any]) -> None:
    del document["paths"]["/v1/jobs"]["post"]["responses"]["409"]


def drop_security(document: dict[str, Any]) -> None:
    document["paths"]["/v1/jobs"]["post"]["security"] = []


def require_new_field(document: dict[str, Any]) -> None:
    payload = schemas(document)["CoclusteringSingleV1Input"]
    payload["properties"]["label"] = {"type": "string"}
    payload["required"].append("label")


def remove_request_field(document: dict[str, Any]) -> None:
    del schemas(document)["CoclusteringSingleV1Parameters"]["properties"]["tolerance"]


def tighten_bound(document: dict[str, Any]) -> None:
    schemas(document)["CoclusteringSingleV1Parameters"]["properties"]["n_groups"]["maximum"] = 5


def narrow_enum(document: dict[str, Any]) -> None:
    schemas(document)["CoclusteringSingleV1Parameters"]["properties"]["response_model"]["enum"] = [
        "ordinal_probit"
    ]


def change_pattern(document: dict[str, Any]) -> None:
    schemas(document)["OpaqueId"]["pattern"] = "^[a-z]{1,8}$"


def drop_response_field(document: dict[str, Any]) -> None:
    schemas(document)["Job"]["required"].remove("expires_at")


def add_state(document: dict[str, Any]) -> None:
    schemas(document)["Job"]["properties"]["state"]["enum"].append("queued")


def null_identifier(document: dict[str, Any]) -> None:
    schemas(document)["JobId"]["type"] = ["string", "null"]


def change_result_schema(document: dict[str, Any]) -> None:
    result = schemas(document)["CoclusteringSingleV1Result"]["properties"]["schema"]
    result["const"] = "selene.coclustering.single.result/2"


def change_without_bump(document: dict[str, Any]) -> None:
    document["paths"]["/healthz"]["get"]["summary"] = "Another summary."


def new_major(document: dict[str, Any]) -> None:
    document["info"]["version"] = "2.0.0"


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (bumped(remove_operation), "DELETE /v1/jobs/{job_id}: removed"),
        (bumped(remove_status), "POST /v1/jobs: no longer answers ['409']"),
        (bumped(drop_security), "POST /v1/jobs: no longer requires authentication"),
        (bumped(require_new_field), "CoclusteringSingleV1Input: newly required ['label']"),
        (bumped(remove_request_field), "CoclusteringSingleV1Parameters.tolerance: removed"),
        (bumped(tighten_bound), "CoclusteringSingleV1Parameters.n_groups: maximum is stricter"),
        (
            bumped(narrow_enum),
            "CoclusteringSingleV1Parameters.response_model: accepts fewer values",
        ),
        (bumped(change_pattern), "OpaqueId: the pattern changed"),
        (bumped(drop_response_field), "Job: no longer always present ['expires_at']"),
        (bumped(add_state), "Job.state: may hold a new value"),
        (bumped(null_identifier), "JobId: may hold another type"),
        (bumped(change_result_schema), "CoclusteringSingleV1Result.schema: the constant changed"),
        (change_without_bump, "info.version: the document changed but stays"),
        (new_major, "info.version: major 2.0.0"),
    ],
)
def test_a_breaking_change_is_refused(
    openapi: dict[str, Any], change: Change, problem: str
) -> None:
    head = copy.deepcopy(openapi)
    change(head)
    problems = check_compatibility.breaking_changes(openapi, head)
    assert any(found.startswith(problem) for found in problems), problems


def add_optional_request_field(document: dict[str, Any]) -> None:
    payload = schemas(document)["CoclusteringSingleV1Input"]
    payload["properties"]["note"] = {"type": "string", "maxLength": 64}


def add_response_field(document: dict[str, Any]) -> None:
    schemas(document)["Job"]["properties"]["progress"] = {"type": "number"}


def add_operation(document: dict[str, Any]) -> None:
    document["paths"]["/v1/kinds"] = {"get": {"responses": {"200": {"description": "The kinds."}}}}


def relax_bound(document: dict[str, Any]) -> None:
    schemas(document)["CoclusteringSingleV1Parameters"]["properties"]["n_groups"]["maximum"] = 20


def add_status(document: dict[str, Any]) -> None:
    document["paths"]["/v1/jobs"]["post"]["responses"]["429"] = {"description": "Too many."}


@pytest.mark.parametrize(
    "change",
    [add_optional_request_field, add_response_field, add_operation, relax_bound, add_status],
)
def test_an_additive_change_is_accepted(openapi: dict[str, Any], change: Change) -> None:
    head = copy.deepcopy(openapi)
    bumped(change)(head)
    assert check_compatibility.breaking_changes(openapi, head) == []


def test_the_same_document_is_compatible(openapi: dict[str, Any]) -> None:
    assert check_compatibility.breaking_changes(openapi, copy.deepcopy(openapi)) == []


def test_a_base_without_operations_has_nothing_to_keep(
    tmp_path: Any, openapi: dict[str, Any]
) -> None:
    import yaml

    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"openapi": "3.1.1", "info": {"version": "0.0.1"}, "paths": {}}))
    head = tmp_path / "head.yaml"
    head.write_text(yaml.safe_dump(openapi))
    assert check_compatibility.main([str(base), str(head)]) == 0
    assert check_compatibility.main([str(head), str(head)]) == 0
