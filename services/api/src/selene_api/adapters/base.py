# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""What every engine adapter offers, and the helpers they share.

An adapter serves one ``(kind, schema_version)``. It has two sides:

- ``prepare`` runs in the server process. It validates a submission against its payload
  schema and the limits of the instance, normalises it and computes its content
  fingerprint. It never imports the engine, so the server stays light and responsive.
- ``compute`` and ``describe`` run in the computation process only. They import the
  engine, call it, and map its result to the result schema of the contract.

The HTTP layer and the job manager only see :class:`PreparedTask` and plain JSON data:
no engine class crosses this boundary.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from selene_api.errors import ApiError, FieldIssue, field_path

__all__ = [
    "Adapter",
    "InstanceLimits",
    "PreparedTask",
    "canonical_json",
    "content_fingerprint",
    "validated",
]


@dataclass(frozen=True)
class InstanceLimits:
    """The technical limits of this deployment that an adapter enforces."""

    max_voters: int
    max_items: int
    max_cells: int
    max_work: int


@dataclass(frozen=True)
class PreparedTask:
    """A validated submission, ready to run.

    Attributes:
        kind: The kind of the submission.
        schema_version: Its payload schema version.
        submission_key: The key the caller chose.
        caller_reference: The opaque reference of the caller, or ``None``.
        payload: The normalised payload, plain JSON data, every setting explicit.
        content_fingerprint: ``sha256:`` and the hash of the normalised content.
    """

    kind: str
    schema_version: int
    submission_key: str
    caller_reference: str | None
    payload: Mapping[str, Any]
    content_fingerprint: str

    def worker_input(self) -> dict[str, Any]:
        """Return what the computation process reads: the kind and the payload only."""
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "payload": self.payload,
        }


class Adapter(Protocol):
    """One payload schema of one kind, between the contract and the engine."""

    kind: str
    schema_version: int

    def prepare(self, raw: bytes, limits: InstanceLimits) -> PreparedTask:
        """Validate and normalise a submission body, in the server process.

        Raises:
            ApiError: ``422`` with the code the contract gives to the refusal.
        """
        ...

    def compute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Run the engine on a normalised payload and return the contract result.

        Runs in the computation process only.
        """
        ...

    def describe(self) -> dict[str, Any]:
        """Return the identity of the engine method behind this adapter.

        Runs in the computation process only.
        """
        ...


def canonical_json(value: object) -> bytes:
    """Serialise to the canonical JSON of the content fingerprint.

    Sorted keys, no spaces, UTF-8, no NaN and no infinity.
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def content_fingerprint(kind: str, schema_version: int, payload: Mapping[str, Any]) -> str:
    """Return ``sha256:`` and the SHA-256 of the canonical JSON of the normalised content."""
    content = {"kind": kind, "schema_version": schema_version, "payload": payload}
    return "sha256:" + hashlib.sha256(canonical_json(content)).hexdigest()


_ISSUES = {
    "missing": "required",
    "extra_forbidden": "not a field of this schema",
    "int_type": "not an integer",
    "float_type": "not a number",
    "string_type": "not a text",
    "bool_type": "not a boolean",
    "list_type": "not an array",
    "tuple_type": "not an array",
    "dict_type": "not an object",
    "model_type": "not an object",
    "model_attributes_type": "not an object",
    "string_pattern_mismatch": "not an allowed identifier",
    "greater_than_equal": "below the minimum",
    "less_than_equal": "above the maximum",
    "too_short": "too few elements",
    "too_long": "too many elements",
    "literal_error": "not an allowed value",
    "union_tag_invalid": "not an allowed value",
    "union_tag_not_found": "required",
    "finite_number": "not a finite number",
    "value_error": "not an allowed value",
}


ModelT = TypeVar("ModelT", bound=BaseModel)


def validated(model: type[ModelT], raw: bytes) -> ModelT:
    """Validate a JSON body against a model of the contract.

    Strict JSON validation: an array is read as a tuple, but ``"2"`` is not 2.

    Raises:
        ApiError: ``422 invalid_request``, with the locations and kinds of the problems
            and never their values.
    """
    try:
        return model.model_validate_json(raw)
    except ValidationError as error:
        issues: Sequence[FieldIssue] = [
            FieldIssue(field_path(detail["loc"]), _ISSUES.get(detail["type"], "not valid"))
            for detail in error.errors(
                include_url=False, include_input=False, include_context=False
            )
        ]
        raise ApiError.invalid(issues) from None
