# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""The OpenAPI document is valid and every operation states its security and errors."""

from typing import Any

import pytest
from openapi_spec_validator import validate

PROBES = {("/healthz", "get"), ("/readyz", "get")}
METHODS = {"get", "post", "put", "patch", "delete"}


def operations(openapi: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (path, method, operation)
        for path, item in openapi["paths"].items()
        for method, operation in item.items()
        if method in METHODS
    ]


def test_the_document_is_valid_openapi_3_1(openapi: dict[str, Any]) -> None:
    assert openapi["openapi"].startswith("3.1.")
    validate(openapi)


def test_the_contract_version_is_semantic_and_major_1(openapi: dict[str, Any]) -> None:
    major, minor, patch = openapi["info"]["version"].split(".")
    assert (major, minor.isdigit(), patch.isdigit()) == ("1", True, True)


def test_the_operations_are_the_ones_of_v1(openapi: dict[str, Any]) -> None:
    found = {(path, method) for path, method, _operation in operations(openapi)}
    assert found == {
        ("/healthz", "get"),
        ("/readyz", "get"),
        ("/v1/service", "get"),
        ("/v1/jobs", "post"),
        ("/v1/jobs/{job_id}", "get"),
        ("/v1/jobs/{job_id}", "delete"),
    }


def test_every_versioned_path_lives_under_v1(openapi: dict[str, Any]) -> None:
    for path in openapi["paths"]:
        assert path in {"/healthz", "/readyz"} or path.startswith("/v1/")


def test_the_machine_token_is_required_by_default(openapi: dict[str, Any]) -> None:
    assert openapi["security"] == [{"machineToken": []}]
    scheme = openapi["components"]["securitySchemes"]["machineToken"]
    assert (scheme["type"], scheme["scheme"]) == ("http", "bearer")


@pytest.mark.parametrize("probe", sorted(PROBES))
def test_only_the_probes_are_public(openapi: dict[str, Any], probe: tuple[str, str]) -> None:
    for path, method, operation in operations(openapi):
        if (path, method) in PROBES:
            assert operation["security"] == []
        else:
            assert "security" not in operation, f"{method} {path} overrides the token"
    path, method = probe
    assert openapi["paths"][path][method]["security"] == []


def test_every_secured_operation_answers_401_and_500(openapi: dict[str, Any]) -> None:
    for path, method, operation in operations(openapi):
        if (path, method) in PROBES:
            continue
        responses = operation["responses"]
        assert responses["401"] == {"$ref": "#/components/responses/Unauthenticated"}, (
            method,
            path,
        )
        assert responses["500"] == {"$ref": "#/components/responses/InternalError"}, (method, path)


def _schema_ref(openapi: dict[str, Any], response: dict[str, Any]) -> str | None:
    if "$ref" in response:
        name = response["$ref"].rsplit("/", 1)[-1]
        response = openapi["components"]["responses"][name]
    content = response.get("content", {}).get("application/json")
    return None if content is None else content["schema"].get("$ref")


def test_every_error_status_uses_the_error_schema(openapi: dict[str, Any]) -> None:
    for path, method, operation in operations(openapi):
        for status, response in operation["responses"].items():
            if int(status) < 400 or (path, method) == ("/readyz", "get"):
                continue
            assert _schema_ref(openapi, response) == "#/components/schemas/ErrorResponse", (
                method,
                path,
                status,
            )


def test_every_success_has_a_schema_except_delete(openapi: dict[str, Any]) -> None:
    for path, method, operation in operations(openapi):
        for status, response in operation["responses"].items():
            if 200 <= int(status) < 300 and status != "204":
                assert _schema_ref(openapi, response) is not None, (method, path, status)


def test_submission_answers_busy_with_retry_after(openapi: dict[str, Any]) -> None:
    busy = openapi["paths"]["/v1/jobs"]["post"]["responses"]["503"]
    assert "Retry-After" in busy["headers"]
    accepted = openapi["paths"]["/v1/jobs"]["post"]["responses"]["202"]
    assert "Location" in accepted["headers"]


def test_request_schemas_refuse_unknown_fields(openapi: dict[str, Any]) -> None:
    schemas = openapi["components"]["schemas"]
    for name in ("CoclusteringSingleV1Submission", "CoclusteringSingleV1Input"):
        assert schemas[name]["additionalProperties"] is False
    assert schemas["CoclusteringSingleV1Parameters"]["additionalProperties"] is False
    for branch in schemas["Exposure"]["oneOf"]:
        assert branch["additionalProperties"] is False


def test_response_schemas_let_clients_ignore_new_fields(openapi: dict[str, Any]) -> None:
    schemas = openapi["components"]["schemas"]
    for name in (
        "Job",
        "JobError",
        "Error",
        "ServiceInfo",
        "Limits",
        "CoclusteringSingleV1Result",
        "SingleCoclusteringValue",
    ):
        assert schemas[name].get("additionalProperties", True) is not False, name


def test_job_states_are_closed(openapi: dict[str, Any]) -> None:
    state = openapi["components"]["schemas"]["Job"]["properties"]["state"]
    assert state["enum"] == ["running", "succeeded", "failed"]


def test_every_job_field_is_always_present(openapi: dict[str, Any]) -> None:
    job = openapi["components"]["schemas"]["Job"]
    assert set(job["required"]) == set(job["properties"])


def test_the_only_supported_submission_is_coclustering_single_1(openapi: dict[str, Any]) -> None:
    branches = openapi["components"]["schemas"]["JobSubmission"]["oneOf"]
    assert branches == [{"$ref": "#/components/schemas/CoclusteringSingleV1Submission"}]
    submission = openapi["components"]["schemas"]["CoclusteringSingleV1Submission"]
    assert submission["properties"]["kind"]["const"] == "coclustering.single"
    assert submission["properties"]["schema_version"]["const"] == 1


def test_the_product_scale_is_minus_two_to_plus_two(openapi: dict[str, Any]) -> None:
    schemas = openapi["components"]["schemas"]
    assert schemas["ProductScore"]["minimum"] == -2
    assert schemas["ProductScore"]["maximum"] == 2
    scale = schemas["CoclusteringSingleV1Input"]["properties"]["scale"]
    assert scale["properties"]["scores"]["const"] == [-2, -1, 0, 1, 2]


def test_exposure_names_its_three_modes(openapi: dict[str, Any]) -> None:
    modes = [
        branch["properties"]["mode"]["const"]
        for branch in openapi["components"]["schemas"]["Exposure"]["oneOf"]
    ]
    assert modes == ["complete", "listed", "unknown"]


def test_every_parameter_is_bounded_with_a_default(openapi: dict[str, Any]) -> None:
    parameters = openapi["components"]["schemas"]["CoclusteringSingleV1Parameters"]
    for name, spec in parameters["properties"].items():
        assert "default" in spec, name
        assert ("minimum" in spec and "maximum" in spec) or "enum" in spec, name


def test_the_seed_fits_a_javascript_integer(openapi: dict[str, Any]) -> None:
    seed = openapi["components"]["schemas"]["CoclusteringSingleV1Input"]["properties"]["seed"]
    assert seed["maximum"] == 2**53 - 1


def test_the_result_schema_is_versioned(openapi: dict[str, Any]) -> None:
    result = openapi["components"]["schemas"]["CoclusteringSingleV1Result"]
    assert result["properties"]["schema"]["const"] == "selene.coclustering.single.result/1"
    reasons = result["properties"]["unavailable"]["properties"]["reason"]["enum"]
    assert reasons == ["no_votes", "not_enough_data", "unsupported_input"]
