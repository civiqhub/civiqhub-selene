# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Transfer objects of contract v1, mirrored from ``openapi.yaml``.

They validate what crosses the HTTP boundary and nothing more: no computation, no
engine class, and no default that ``openapi.yaml`` does not write. The contract tests
check that these models accept and refuse the fixtures exactly as the JSON schemas of
``openapi.yaml`` do.

Every model refuses unknown fields, and every scalar is strict: ``"2"``, ``2.0`` and
``true`` are not the integer 2. Arrays are read as tuples where the contract fixes
their length.
"""

from itertools import pairwise
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    field_validator,
    model_validator,
)

__all__ = [
    "CONTRACT_VERSION",
    "MAX_JS_INTEGER",
    "PRODUCT_SCORES",
    "CoclusteringSingleV1Input",
    "CoclusteringSingleV1Parameters",
    "CoclusteringSingleV1Submission",
    "CoclusteringSingleV2Input",
    "CoclusteringSingleV2Parameters",
    "CoclusteringSingleV2Submission",
    "CompleteExposure",
    "ListedExposure",
    "UnknownExposure",
]

CONTRACT_VERSION = "1.2.0"
"""``info.version`` of ``openapi.yaml``; the contract tests keep them equal."""

PRODUCT_SCORES = (-2, -1, 0, 1, 2)
"""The product scale of payload schema 1, from disagree to agree; 0 is a neutral vote."""

MAX_SCALE_LEVELS = 11
"""The most scores a scale of payload schema 2 has."""

MAX_SCALES = 64
"""The most scales a payload of schema 2 declares."""

MAX_JS_INTEGER = 2**53 - 1
"""The largest integer a JavaScript client keeps exactly."""

OpaqueId = Annotated[StrictStr, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")]
SubmissionKey = Annotated[
    StrictStr, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
]
CallerReference = Annotated[
    StrictStr, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
]
Position = Annotated[StrictInt, Field(ge=0)]
ProductScore = Annotated[StrictInt, Field(ge=-2, le=2)]
ScaleScore = Annotated[StrictInt, Field(ge=-1000, le=1000)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Scope(_Model):
    project: OpaqueId


class Scale(_Model):
    scores: tuple[StrictInt, ...]

    @field_validator("scores")
    @classmethod
    def _product_scale(cls, scores: tuple[int, ...]) -> tuple[int, ...]:
        if scores != PRODUCT_SCORES:
            raise ValueError("schema 1 accepts the scale -2, -1, 0, 1, 2 only")
        return scores


class Item(_Model):
    id: OpaqueId
    revision: OpaqueId


class CompleteExposure(_Model):
    """Every item was shown to every voter."""

    mode: Literal["complete"]


class ListedExposure(_Model):
    """The voted cells and the listed cells were shown; no other cell was."""

    mode: Literal["listed"]
    shown_without_vote: Annotated[list[tuple[Position, Position]], Field(max_length=2_000_000)]


class UnknownExposure(_Model):
    """The caller does not know which items were shown."""

    mode: Literal["unknown"]


Exposure = Annotated[
    CompleteExposure | ListedExposure | UnknownExposure, Field(discriminator="mode")
]


class CoclusteringSingleV1Parameters(_Model):
    """Settings of the latent block model; the defaults are the engine defaults."""

    n_groups: Annotated[StrictInt, Field(ge=1, le=10)] = 2
    n_blocks: Annotated[StrictInt, Field(ge=1, le=10)] = 2
    n_starts: Annotated[StrictInt, Field(ge=1, le=20)] = 6
    max_iterations: Annotated[StrictInt, Field(ge=1, le=1000)] = 200
    tolerance: Annotated[StrictInt | StrictFloat, Field(ge=0, le=1, allow_inf_nan=False)] = 1e-6
    response_model: Literal["ordinal_probit", "categorical"] = "ordinal_probit"
    initialization: Literal["random", "spectral"] = "random"


class CoclusteringSingleV1Input(_Model):
    """The payload of ``coclustering.single``, schema 1. See ``openapi.yaml``."""

    scope: Scope
    scale: Scale
    voters: Annotated[list[OpaqueId], Field(min_length=1, max_length=20_000)]
    items: Annotated[list[Item], Field(min_length=1, max_length=5_000)]
    votes: Annotated[list[tuple[Position, Position, ProductScore]], Field(max_length=2_000_000)]
    exposure: Exposure
    parameters: CoclusteringSingleV1Parameters = CoclusteringSingleV1Parameters()
    seed: Annotated[StrictInt, Field(ge=0, le=MAX_JS_INTEGER)]

    @field_validator("voters")
    @classmethod
    def _unique_voters(cls, voters: list[str]) -> list[str]:
        if len(set(voters)) != len(voters):
            raise ValueError("voters must be unique")
        return voters


class CoclusteringSingleV1Submission(_Model):
    """A submission of ``coclustering.single``, schema 1."""

    kind: Literal["coclustering.single"]
    schema_version: Annotated[StrictInt, Field(ge=1, le=1)]
    submission_key: SubmissionKey
    caller_reference: CallerReference | None = None
    payload: CoclusteringSingleV1Input

    @model_validator(mode="before")
    @classmethod
    def _no_null_reference(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("caller_reference", "") is None:
            raise ValueError("caller_reference is a text when present, never null")
        return data


# Payload schema 2: each item on its declared scale -----------------------------------


class DeclaredScale(_Model):
    """A scale of payload schema 2: an identifier, and its scores in the order of agreement."""

    id: OpaqueId
    scores: Annotated[tuple[ScaleScore, ...], Field(min_length=2, max_length=MAX_SCALE_LEVELS)]

    @field_validator("scores")
    @classmethod
    def _increasing(cls, scores: tuple[int, ...]) -> tuple[int, ...]:
        if any(low >= high for low, high in pairwise(scores)):
            raise ValueError("the scores of a scale are strictly increasing")
        return scores


class ItemV2(_Model):
    id: OpaqueId
    revision: OpaqueId
    scale: OpaqueId


class ThresholdSettings(_Model):
    """Where the thresholds of the ordinal probit lie; see ``openapi.yaml``."""

    axis: Literal["auto", "levels", "unit"] = "auto"
    mode: Literal["fixed", "estimated"] = "fixed"


class CoclusteringSingleV2Parameters(CoclusteringSingleV1Parameters):
    """The settings of schema 1, and where the thresholds of the ordinal probit lie."""

    thresholds: ThresholdSettings = ThresholdSettings()


class CoclusteringSingleV2Input(_Model):
    """The payload of ``coclustering.single``, schema 2. See ``openapi.yaml``."""

    scope: Scope
    scales: Annotated[list[DeclaredScale], Field(min_length=1, max_length=MAX_SCALES)]
    voters: Annotated[list[OpaqueId], Field(min_length=1, max_length=20_000)]
    items: Annotated[list[ItemV2], Field(min_length=1, max_length=5_000)]
    votes: Annotated[list[tuple[Position, Position, ScaleScore]], Field(max_length=2_000_000)]
    exposure: Exposure
    parameters: CoclusteringSingleV2Parameters = CoclusteringSingleV2Parameters()
    seed: Annotated[StrictInt, Field(ge=0, le=MAX_JS_INTEGER)]

    @field_validator("voters")
    @classmethod
    def _unique_voters(cls, voters: list[str]) -> list[str]:
        if len(set(voters)) != len(voters):
            raise ValueError("voters must be unique")
        return voters


class CoclusteringSingleV2Submission(_Model):
    """A submission of ``coclustering.single``, schema 2."""

    kind: Literal["coclustering.single"]
    schema_version: Annotated[StrictInt, Field(ge=2, le=2)]
    submission_key: SubmissionKey
    caller_reference: CallerReference | None = None
    payload: CoclusteringSingleV2Input

    @model_validator(mode="before")
    @classmethod
    def _no_null_reference(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("caller_reference", "") is None:
            raise ValueError("caller_reference is a text when present, never null")
        return data
