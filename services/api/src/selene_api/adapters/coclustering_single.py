# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Adapter of ``coclustering.single``, payload schema 1, to ``LatentBlockModel``.

The server side checks and normalises a submission without the engine. The computation
side maps the normalised payload to the engine explicitly, calls
``LatentBlockModel.estimate`` and maps ``Result.to_dict()`` to
``selene.coclustering.single.result/1``:

- product scores -2..+2 become ranks of ``OrdinalScale((-2, -1, 0, 1, 2))`` through
  the engine, never by arithmetic here; a neutral vote (0) is a vote;
- ``exposure`` becomes the ``shown`` mask of the engine: ``complete`` shows every cell,
  ``listed`` shows the voted cells and the listed ones; the absence of a vote is never
  read as an exposure;
- rows and columns keep the order of ``voters`` and ``items``.

No formula of the engine is copied here.
"""

from collections.abc import Mapping
from typing import Any

from selene_api.adapters.base import (
    InstanceLimits,
    PreparedTask,
    content_fingerprint,
    validated,
)
from selene_api.errors import ApiError, FieldIssue
from selene_api.schemas import PRODUCT_SCORES, CoclusteringSingleV1Submission

__all__ = ["RESULT_SCHEMA", "CoclusteringSingleV1", "work_of"]

RESULT_SCHEMA = "selene.coclustering.single.result/1"

_VALUE_FIELDS = (
    "voters",
    "items",
    "scores",
    "group_index",
    "block_index",
    "group_memberships",
    "block_memberships",
    "coclusters",
    "objective",
    "integrated_likelihood",
    "history",
    "computation",
)
_PROVENANCE_FIELDS = (
    "method",
    "engine_version",
    "parameters",
    "input_fingerprint",
    "randomness",
    "dependencies",
    "fingerprint",
)


def work_of(n_voters: int, n_items: int, parameters: Mapping[str, Any]) -> int:
    """Return the computation budget of a submission, as ``Limits.max_work`` defines it."""
    cells = n_voters * n_items
    return (
        int(parameters["n_starts"])
        * int(parameters["max_iterations"])
        * cells
        * (len(PRODUCT_SCORES) + 1)
        * (int(parameters["n_groups"]) + int(parameters["n_blocks"]))
    )


class CoclusteringSingleV1:
    """``coclustering.single``, schema 1: latent block model, one group per voter."""

    kind = "coclustering.single"
    schema_version = 1

    # Server side: no engine import -------------------------------------------------------

    def prepare(self, raw: bytes, limits: InstanceLimits) -> PreparedTask:
        """Validate, check the limits and normalise a submission body."""
        submission = validated(CoclusteringSingleV1Submission, raw)
        payload = submission.payload
        n_voters, n_items = len(payload.voters), len(payload.items)

        too_large = []
        if n_voters > limits.max_voters:
            too_large.append(FieldIssue("payload.voters", "more voters than max_voters"))
        if n_items > limits.max_items:
            too_large.append(FieldIssue("payload.items", "more items than max_items"))
        if n_voters * n_items > limits.max_cells:
            too_large.append(FieldIssue("payload", "more cells than max_cells"))
        if too_large:
            raise ApiError.limit_exceeded(too_large)

        if payload.exposure.mode == "unknown":
            raise ApiError.unsupported_exposure()

        issues: list[FieldIssue] = []
        seen_items: set[str] = set()
        for position, item in enumerate(payload.items):
            if item.id in seen_items:
                issues.append(FieldIssue(f"payload.items[{position}].id", "repeats an item"))
            seen_items.add(item.id)

        voted: set[tuple[int, int]] = set()
        for position, (row, column, _score) in enumerate(payload.votes):
            if row >= n_voters:
                issues.append(FieldIssue(f"payload.votes[{position}][0]", "no such voter row"))
            elif column >= n_items:
                issues.append(FieldIssue(f"payload.votes[{position}][1]", "no such item column"))
            elif (row, column) in voted:
                issues.append(FieldIssue(f"payload.votes[{position}]", "second vote on a cell"))
            voted.add((row, column))
            if len(issues) > 20:
                break

        listed: list[list[int]] = []
        if payload.exposure.mode == "listed":
            seen_cells: set[tuple[int, int]] = set()
            for position, (row, column) in enumerate(payload.exposure.shown_without_vote):
                path = f"payload.exposure.shown_without_vote[{position}]"
                if row >= n_voters or column >= n_items:
                    issues.append(FieldIssue(path, "no such cell"))
                elif (row, column) in voted:
                    issues.append(FieldIssue(path, "cell also voted"))
                elif (row, column) in seen_cells:
                    issues.append(FieldIssue(path, "cell listed twice"))
                seen_cells.add((row, column))
                if len(issues) > 20:
                    break
            listed = sorted([row, column] for row, column in seen_cells)
        if issues:
            raise ApiError.invalid(issues)

        parameters = payload.parameters.model_dump()
        parameters["tolerance"] = float(parameters["tolerance"])
        if work_of(n_voters, n_items, parameters) > limits.max_work:
            raise ApiError.limit_exceeded(
                [FieldIssue("payload.parameters", "computation budget above max_work")]
            )

        exposure: dict[str, Any] = {"mode": payload.exposure.mode}
        if payload.exposure.mode == "listed":
            exposure["shown_without_vote"] = listed
        normalised = {
            "scope": {"project": payload.scope.project},
            "scale": {"scores": list(payload.scale.scores)},
            "voters": list(payload.voters),
            "items": [{"id": item.id, "revision": item.revision} for item in payload.items],
            "votes": sorted([row, column, score] for row, column, score in payload.votes),
            "exposure": exposure,
            "parameters": parameters,
            "seed": payload.seed,
        }
        return PreparedTask(
            kind=self.kind,
            schema_version=self.schema_version,
            submission_key=submission.submission_key,
            caller_reference=submission.caller_reference,
            payload=normalised,
            content_fingerprint=content_fingerprint(self.kind, self.schema_version, normalised),
        )

    # Computation side: imports the engine ------------------------------------------------

    def _model(self, parameters: Mapping[str, Any]) -> Any:
        from civiqhub_selene.blocks import LatentBlockModel, RandomStart, SpectralStart
        from civiqhub_selene.blocks.responses import Categorical, OrdinalProbit

        response_models = {"ordinal_probit": OrdinalProbit, "categorical": Categorical}
        initializations = {"random": RandomStart, "spectral": SpectralStart}
        return LatentBlockModel(
            n_groups=parameters["n_groups"],
            n_blocks=parameters["n_blocks"],
            n_starts=parameters["n_starts"],
            max_iterations=parameters["max_iterations"],
            tolerance=parameters["tolerance"],
            response_model=response_models[parameters["response_model"]](),
            initialization=initializations[parameters["initialization"]](),
        )

    def votes_of(self, payload: Mapping[str, Any]) -> Any:
        """Return the engine ``VoteMatrix`` of a normalised payload."""
        from civiqhub_selene.blocks import OrdinalScale, VoteMatrix

        voters = list(payload["voters"])
        items = [item["id"] for item in payload["items"]]
        records = [(voters[row], items[column], score) for row, column, score in payload["votes"]]
        exposure = payload["exposure"]
        if exposure["mode"] == "complete":
            shown = None
        elif exposure["mode"] == "listed":
            shown = [(voters[row], items[column]) for row, column in exposure["shown_without_vote"]]
        else:
            raise ValueError("schema 1 computes a known exposure only")
        return VoteMatrix.from_records(
            records,
            shown=shown,
            voters=voters,
            items=items,
            scale=OrdinalScale(tuple(payload["scale"]["scores"])),
        )

    def compute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Run ``LatentBlockModel`` and return ``selene.coclustering.single.result/1``."""
        model = self._model(payload["parameters"])
        result = model.estimate(self.votes_of(payload), seed=payload["seed"])
        return self.result_document(result.to_dict())

    def result_document(self, engine: Mapping[str, Any]) -> dict[str, Any]:
        """Map ``Result.to_dict()`` (``civiqhub_selene.result/0``) to the contract result."""
        if engine.get("schema") != "civiqhub_selene.result/0":
            raise ValueError(f"unknown engine result schema {engine.get('schema')!r}")
        provenance = engine["provenance"]
        document: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "available": engine["available"],
            "provenance": {name: provenance[name] for name in _PROVENANCE_FIELDS},
        }
        if engine["available"]:
            document["value"] = {name: engine["value"][name] for name in _VALUE_FIELDS}
        else:
            document["unavailable"] = {
                "reason": engine["unavailable"]["reason"],
                "detail": engine["unavailable"]["detail"],
            }
        return document

    def describe(self) -> dict[str, Any]:
        """Return the engine method behind this adapter."""
        from civiqhub_selene.blocks import LatentBlockModel

        method = LatentBlockModel.method()
        return {"name": method.name, "version": method.version}
