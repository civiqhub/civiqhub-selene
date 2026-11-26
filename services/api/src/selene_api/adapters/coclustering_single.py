# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Adapters of ``coclustering.single``, payload schemas 1 and 2, to ``LatentBlockModel``.

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

Payload schema 2 declares the scales of the items: each item names one of the declared
scales, each vote is a score of the scale of its item, and the engine reads each item on
its scale (``ItemScales``), the ordinal probit giving each scale its grid of thresholds.
Its result, ``selene.coclustering.single.result/2``, gives everything read per answer
scale by scale, even when every item has the same scale. Schema 1 is unchanged.

No formula of the engine is copied here.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from selene_api.adapters.base import (
    InstanceLimits,
    PreparedTask,
    content_fingerprint,
    validated,
)
from selene_api.errors import ApiError, FieldIssue
from selene_api.schemas import (
    PRODUCT_SCORES,
    CoclusteringSingleV1Submission,
    CoclusteringSingleV2Submission,
    ListedExposure,
)

__all__ = [
    "RESULT_SCHEMA",
    "RESULT_SCHEMA_V2",
    "CoclusteringSingleV1",
    "CoclusteringSingleV2",
    "work_of",
]

RESULT_SCHEMA = "selene.coclustering.single.result/1"
RESULT_SCHEMA_V2 = "selene.coclustering.single.result/2"
_MAX_ISSUES = 20

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
# The fields of the table of co-clusters that result/1 defines. The engine writes more
# since a table can carry the grids of the ordinal probit (SEL-8.3): result/1 keeps its
# document, field for field, and a later result schema may add them.
_COCLUSTER_FIELDS = (
    "scores",
    "vote_rate",
    "vote_shares",
    "n_votes",
    "latent_mean",
    "latent_spread",
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


def work_of(
    n_voters: int, n_items: int, parameters: Mapping[str, Any], n_levels: int = len(PRODUCT_SCORES)
) -> int:
    """Return the computation budget of a submission, as ``Limits.max_work`` defines it.

    ``n_levels`` is the number of scores of the longest scale: the states of a cell are
    one more.
    """
    cells = n_voters * n_items
    return (
        int(parameters["n_starts"])
        * int(parameters["max_iterations"])
        * cells
        * (n_levels + 1)
        * (int(parameters["n_groups"]) + int(parameters["n_blocks"]))
    )


def _too_large(n_voters: int, n_items: int, limits: InstanceLimits) -> None:
    """Refuse a snapshot beyond the limits of the instance (``limit_exceeded``)."""
    too_large = []
    if n_voters > limits.max_voters:
        too_large.append(FieldIssue("payload.voters", "more voters than max_voters"))
    if n_items > limits.max_items:
        too_large.append(FieldIssue("payload.items", "more items than max_items"))
    if n_voters * n_items > limits.max_cells:
        too_large.append(FieldIssue("payload", "more cells than max_cells"))
    if too_large:
        raise ApiError.limit_exceeded(too_large)


def _cell_issues(
    votes: Sequence[tuple[int, int, int]],
    exposure: Any,
    n_voters: int,
    n_items: int,
    issues: list[FieldIssue],
) -> list[list[int]]:
    """Check the voted and the listed cells; return the listed cells, sorted.

    A vote names an existing cell, once; a listed cell names an existing cell that is
    not voted, once. The issues are appended, at most about twenty.
    """
    voted: set[tuple[int, int]] = set()
    for position, (row, column, _score) in enumerate(votes):
        if row >= n_voters:
            issues.append(FieldIssue(f"payload.votes[{position}][0]", "no such voter row"))
        elif column >= n_items:
            issues.append(FieldIssue(f"payload.votes[{position}][1]", "no such item column"))
        elif (row, column) in voted:
            issues.append(FieldIssue(f"payload.votes[{position}]", "second vote on a cell"))
        voted.add((row, column))
        if len(issues) > _MAX_ISSUES:
            break
    if not isinstance(exposure, ListedExposure):
        return []
    seen_cells: set[tuple[int, int]] = set()
    for position, (row, column) in enumerate(exposure.shown_without_vote):
        path = f"payload.exposure.shown_without_vote[{position}]"
        if row >= n_voters or column >= n_items:
            issues.append(FieldIssue(path, "no such cell"))
        elif (row, column) in voted:
            issues.append(FieldIssue(path, "cell also voted"))
        elif (row, column) in seen_cells:
            issues.append(FieldIssue(path, "cell listed twice"))
        seen_cells.add((row, column))
        if len(issues) > _MAX_ISSUES:
            break
    return sorted([row, column] for row, column in seen_cells)


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
        _too_large(n_voters, n_items, limits)
        if payload.exposure.mode == "unknown":
            raise ApiError.unsupported_exposure()

        issues: list[FieldIssue] = []
        seen_items: set[str] = set()
        for position, item in enumerate(payload.items):
            if item.id in seen_items:
                issues.append(FieldIssue(f"payload.items[{position}].id", "repeats an item"))
            seen_items.add(item.id)
        listed = _cell_issues(payload.votes, payload.exposure, n_voters, n_items, issues)
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
            value = {name: engine["value"][name] for name in _VALUE_FIELDS}
            value["coclusters"] = {
                name: field
                for name, field in engine["value"]["coclusters"].items()
                if name in _COCLUSTER_FIELDS
            }
            document["value"] = value
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


_COCLUSTER_FIELDS_V2 = (
    "vote_rate",
    "n_votes",
    "latent_mean",
    "latent_spread",
    "thresholds",
    "has_fixed_spread",
)
_VALUE_FIELDS_V2 = (
    "voters",
    "items",
    "group_index",
    "block_index",
    "group_memberships",
    "block_memberships",
    "objective",
    "integrated_likelihood",
    "history",
    "computation",
)


class CoclusteringSingleV2(CoclusteringSingleV1):
    """``coclustering.single``, schema 2: latent block model, each item on its declared scale.

    The payload declares its scales once, each with an identifier and its scores in the
    order of agreement; each item names its scale, and each vote is a score of the scale
    of its item. The server refuses, as ``invalid_request``: two scales with the same
    identifier or the same scores (they would be one scale), an item whose scale is not
    declared, a declared scale that no item uses, and a vote off the scale of its item.

    The normalised payload lists the scales in the order in which the items first use
    them, the order of the engine, so that two payloads that declare the same scales in
    another order have the same content fingerprint.
    """

    schema_version = 2

    # Server side: no engine import -------------------------------------------------------

    def prepare(self, raw: bytes, limits: InstanceLimits) -> PreparedTask:
        """Validate, check the limits and normalise a submission body."""
        submission = validated(CoclusteringSingleV2Submission, raw)
        payload = submission.payload
        n_voters, n_items = len(payload.voters), len(payload.items)
        _too_large(n_voters, n_items, limits)
        if payload.exposure.mode == "unknown":
            raise ApiError.unsupported_exposure()

        issues: list[FieldIssue] = []
        declared: dict[str, tuple[int, ...]] = {}
        seen_scores: set[tuple[int, ...]] = set()
        for position, scale in enumerate(payload.scales):
            if scale.id in declared:
                issues.append(FieldIssue(f"payload.scales[{position}].id", "repeats a scale"))
            elif scale.scores in seen_scores:
                issues.append(
                    FieldIssue(
                        f"payload.scales[{position}].scores", "repeats the scores of a scale"
                    )
                )
            declared.setdefault(scale.id, scale.scores)
            seen_scores.add(scale.scores)
        seen_items: set[str] = set()
        used: list[str] = []
        for position, item in enumerate(payload.items):
            if item.id in seen_items:
                issues.append(FieldIssue(f"payload.items[{position}].id", "repeats an item"))
            seen_items.add(item.id)
            if item.scale not in declared:
                issues.append(FieldIssue(f"payload.items[{position}].scale", "no such scale"))
            elif item.scale not in used:
                used.append(item.scale)
        for position, scale in enumerate(payload.scales):
            if scale.id in declared and scale.id not in used and not issues:
                issues.append(FieldIssue(f"payload.scales[{position}]", "used by no item"))
        listed = _cell_issues(payload.votes, payload.exposure, n_voters, n_items, issues)
        if not issues:
            for position, (_row, column, score) in enumerate(payload.votes):
                if score not in declared[payload.items[column].scale]:
                    issues.append(
                        FieldIssue(f"payload.votes[{position}][2]", "not a score of its scale")
                    )
                if len(issues) > _MAX_ISSUES:
                    break
        if issues:
            raise ApiError.invalid(issues)

        parameters = payload.parameters.model_dump()
        parameters["tolerance"] = float(parameters["tolerance"])
        longest = max(len(declared[scale]) for scale in used)
        if work_of(n_voters, n_items, parameters, longest) > limits.max_work:
            raise ApiError.limit_exceeded(
                [FieldIssue("payload.parameters", "computation budget above max_work")]
            )

        exposure: dict[str, Any] = {"mode": payload.exposure.mode}
        if payload.exposure.mode == "listed":
            exposure["shown_without_vote"] = listed
        normalised = {
            "scope": {"project": payload.scope.project},
            "scales": [{"id": scale, "scores": list(declared[scale])} for scale in used],
            "voters": list(payload.voters),
            "items": [
                {"id": item.id, "revision": item.revision, "scale": item.scale}
                for item in payload.items
            ],
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
        from civiqhub_selene.blocks.responses import Categorical, OrdinalProbit, Thresholds

        thresholds = parameters["thresholds"]
        if parameters["response_model"] == "categorical":
            response_model: Any = Categorical()
        elif thresholds == {"axis": "auto", "mode": "fixed"}:
            # The default grids are those of OrdinalProbit(): the same computation as a
            # submission of schema 1 on the product scale has the same provenance.
            response_model = OrdinalProbit()
        else:
            response_model = OrdinalProbit(
                thresholds=Thresholds(axis=thresholds["axis"], mode=thresholds["mode"])
            )
        initializations = {"random": RandomStart, "spectral": SpectralStart}
        return LatentBlockModel(
            n_groups=parameters["n_groups"],
            n_blocks=parameters["n_blocks"],
            n_starts=parameters["n_starts"],
            max_iterations=parameters["max_iterations"],
            tolerance=parameters["tolerance"],
            response_model=response_model,
            initialization=initializations[parameters["initialization"]](),
        )

    def votes_of(self, payload: Mapping[str, Any]) -> Any:
        """Return the engine ``VoteMatrix`` of a normalised payload, each item on its scale."""
        from civiqhub_selene.blocks import OrdinalScale, VoteMatrix

        scales = {scale["id"]: OrdinalScale(tuple(scale["scores"])) for scale in payload["scales"]}
        voters = list(payload["voters"])
        items = [item["id"] for item in payload["items"]]
        records = [(voters[row], items[column], score) for row, column, score in payload["votes"]]
        exposure = payload["exposure"]
        if exposure["mode"] == "complete":
            shown = None
        elif exposure["mode"] == "listed":
            shown = [(voters[row], items[column]) for row, column in exposure["shown_without_vote"]]
        else:
            raise ValueError("schema 2 computes a known exposure only")
        return VoteMatrix.from_records(
            records,
            shown=shown,
            voters=voters,
            items=items,
            item_scales={item["id"]: scales[item["scale"]] for item in payload["items"]},
        )

    def compute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Run ``LatentBlockModel`` and return ``selene.coclustering.single.result/2``."""
        model = self._model(payload["parameters"])
        result = model.estimate(self.votes_of(payload), seed=payload["seed"])
        return self.result_document_v2(result.to_dict(), payload)

    def result_document_v2(
        self, engine: Mapping[str, Any], payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Map ``Result.to_dict()`` to ``selene.coclustering.single.result/2``.

        Everything read per answer is given scale by scale, even when the items share
        one scale: ``scales`` (each with the identifier the payload declared),
        ``scale_index`` and ``coclusters.vote_shares_by_scale``. The engine writes one
        scale in the form of one scale; this only puts it in a list.
        """
        if engine.get("schema") != "civiqhub_selene.result/0":
            raise ValueError(f"unknown engine result schema {engine.get('schema')!r}")
        provenance = engine["provenance"]
        document: dict[str, Any] = {
            "schema": RESULT_SCHEMA_V2,
            "available": engine["available"],
            "provenance": {name: provenance[name] for name in _PROVENANCE_FIELDS},
        }
        if not engine["available"]:
            document["unavailable"] = {
                "reason": engine["unavailable"]["reason"],
                "detail": engine["unavailable"]["detail"],
            }
            return document
        value = engine["value"]
        table = value["coclusters"]
        identifiers = {tuple(scale["scores"]): scale["id"] for scale in payload["scales"]}
        if "scales" in value:
            scores_by_scale = [tuple(scores) for scores in value["scales"]]
            scale_index = list(value["scale_index"])
            shares = list(table["vote_shares_by_scale"])
        else:
            scores_by_scale = [tuple(value["scores"])]
            scale_index = [0] * len(value["items"])
            shares = [table["vote_shares"]]
        coclusters = {name: table[name] for name in _COCLUSTER_FIELDS_V2 if name in table}
        coclusters["vote_shares_by_scale"] = shares
        document["value"] = {
            **{name: value[name] for name in _VALUE_FIELDS_V2},
            "scales": [
                {"id": identifiers[scores], "scores": list(scores)} for scores in scores_by_scale
            ],
            "scale_index": scale_index,
            "coclusters": coclusters,
        }
        return document
