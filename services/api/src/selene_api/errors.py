# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Errors the service answers, as contract v1 writes them.

Every error body is ``{"error": {"code", "message", "retryable", "fields"?}}``. A
message is a fixed English sentence chosen here: it never carries a value of the
request, and ``fields`` names locations, never values.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ApiError", "FieldIssue", "field_path"]

_MAX_FIELD_ISSUES = 20


@dataclass(frozen=True)
class FieldIssue:
    """Where a request is wrong, and what is wrong, without the value."""

    path: str
    issue: str


def field_path(location: Iterable[str | int]) -> str:
    """Write a location as ``payload.votes[12][2]``."""
    text = ""
    for step in location:
        if isinstance(step, int):
            text += f"[{step}]"
        else:
            text += f".{step}" if text else step
    return text or "body"


@dataclass(eq=False)
class ApiError(Exception):
    """An error the HTTP layer turns into an error answer.

    Not frozen: raising an exception sets its traceback and cause attributes.
    """

    status: int
    code: str
    message: str
    retryable: bool = False
    fields: Sequence[FieldIssue] = ()
    headers: Mapping[str, str] = field(default_factory=dict)

    def body(self) -> dict[str, Any]:
        """Return the error body of the contract."""
        error: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.fields:
            error["fields"] = [
                {"path": issue.path, "issue": issue.issue}
                for issue in self.fields[:_MAX_FIELD_ISSUES]
            ]
        return {"error": error}

    # The errors of the contract, by code -----------------------------------------------

    @classmethod
    def malformed(cls) -> "ApiError":
        return cls(400, "malformed_request", "The request body is not a JSON document.")

    @classmethod
    def unauthenticated(cls) -> "ApiError":
        return cls(
            401,
            "unauthenticated",
            "A valid machine token is required.",
            headers={"WWW-Authenticate": 'Bearer realm="selene"'},
        )

    @classmethod
    def not_found(cls) -> "ApiError":
        return cls(404, "not_found", "No operation has this path.")

    @classmethod
    def method_not_allowed(cls) -> "ApiError":
        return cls(405, "method_not_allowed", "This operation does not accept this method.")

    @classmethod
    def job_not_found(cls) -> "ApiError":
        return cls(404, "job_not_found", "No retained job has this identifier.")

    @classmethod
    def conflict(cls) -> "ApiError":
        return cls(
            409,
            "submission_conflict",
            "This submission key is bound to a retained job with other content.",
        )

    @classmethod
    def too_large(cls) -> "ApiError":
        return cls(413, "payload_too_large", "The request body is larger than the limit.")

    @classmethod
    def media_type(cls) -> "ApiError":
        return cls(415, "unsupported_media_type", "The request body must be application/json.")

    @classmethod
    def invalid(cls, fields: Sequence[FieldIssue] = ()) -> "ApiError":
        return cls(
            422, "invalid_request", "The submission does not follow the contract.", False, fields
        )

    @classmethod
    def unsupported_kind(cls) -> "ApiError":
        return cls(422, "unsupported_kind", "This instance does not compute this kind.")

    @classmethod
    def unsupported_schema_version(cls) -> "ApiError":
        return cls(
            422,
            "unsupported_schema_version",
            "This instance does not accept this payload schema version for this kind.",
        )

    @classmethod
    def unsupported_exposure(cls) -> "ApiError":
        return cls(
            422,
            "unsupported_exposure",
            "This payload schema needs a known exposure: complete or listed.",
            False,
            (FieldIssue("payload.exposure.mode", "unknown exposure is not supported"),),
        )

    @classmethod
    def limit_exceeded(cls, fields: Sequence[FieldIssue]) -> "ApiError":
        return cls(
            422,
            "limit_exceeded",
            "The submission exceeds a limit of this instance; see GET /v1/service.",
            False,
            fields,
        )

    @classmethod
    def busy(cls, retry_after: int) -> "ApiError":
        return cls(
            503,
            "busy",
            "The computation slot is taken; retry later.",
            True,
            headers={"Retry-After": str(retry_after)},
        )

    @classmethod
    def shutting_down(cls) -> "ApiError":
        return cls(503, "shutting_down", "This instance is shutting down.", True)

    @classmethod
    def not_ready(cls) -> "ApiError":
        return cls(503, "not_ready", "This instance is not ready yet.", True)

    @classmethod
    def internal(cls) -> "ApiError":
        return cls(500, "internal_error", "An unexpected error occurred.", True)
