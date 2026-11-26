# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Write the synthetic fixtures of contract v1 and their reference results. TEST ONLY.

Usage, from the root of the repository, with the engine installed with its ``blocks``
extra and PyYAML::

    python services/api/contract/make_fixtures.py          # write the fixtures
    python services/api/contract/make_fixtures.py --check  # fail if they are stale

Every vote, voter and item is invented: no fixture comes from a real participant. The
reference result is computed here, locally, by calling the engine directly; it is the
"local" side of the local/service equivalence that the contract tests check. This script
reads ``openapi.yaml`` for the defaults and never imports the service.

Changing the output of a fixture is a scientific change when the engine numbers move:
say so in the pull request and in CHANGELOG.md.
"""

import argparse
import copy
import hashlib
import json
import math
import pathlib
import sys
from typing import Any

import yaml

from civiqhub_selene.blocks import (
    LatentBlockModel,
    OrdinalScale,
    RandomStart,
    SpectralStart,
    VoteMatrix,
)
from civiqhub_selene.blocks.responses import Categorical, OrdinalProbit, Thresholds
from selene_contract import canonical_value

HERE = pathlib.Path(__file__).resolve().parent
CONTRACT = HERE.parent / "openapi.yaml"
FIXTURES = HERE / "fixtures" / "v1"

KIND = "coclustering.single"
SCHEMA_VERSION = 1
RESULT_SCHEMA = "selene.coclustering.single.result/1"
PRODUCT_SCORES = (-2, -1, 0, 1, 2)
# Another processor rounds the last bits differently, and the iterative fits of the
# response model carry that through to about 1e-9 relative; partitions must not move.
TOLERANCE = {"relative": 1e-6, "absolute": 1e-9}

# The planted collective: two groups of six voters, two blocks of three items. Group A
# agrees with block X and disagrees with block Y, group B does the opposite. A number is
# a vote at that product score (0 is a neutral vote), "A" a cell shown without a vote,
# "-" a cell that was not shown. On so few cells the classification algorithm can stop at
# a local optimum from a random start; the spectral start finds the planted arrangement.
ITEMS = ("itm-a", "itm-b", "itm-c", "itm-d", "itm-e", "itm-f")
TABLE = (
    ("vtr-01", (2, 2, 1, -2, -1, -2)),
    ("vtr-02", (2, 1, 2, -1, -2, "A")),
    ("vtr-03", (1, 0, 2, -2, "-", -1)),
    ("vtr-04", (2, 2, "A", -1, -2, -2)),
    ("vtr-05", ("-", 1, 2, -2, -1, -1)),
    ("vtr-06", (2, 2, 1, "A", -2, -2)),
    ("vtr-07", (-2, -1, -2, 2, 1, 2)),
    ("vtr-08", (-1, -2, "A", 2, 2, 1)),
    ("vtr-09", (-2, -2, -1, 1, 0, 2)),
    ("vtr-10", ("A", -1, -2, 2, 2, "-")),
    ("vtr-11", (-2, "-", -1, 1, 2, 2)),
    ("vtr-12", (-1, -2, -2, 2, "A", 1)),
)
PLANTED_GROUPS = (0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)
PLANTED_BLOCKS = (0, 0, 0, 1, 1, 1)
SEED = 20261125

# Payload schema 2: a planted survey whose items use three scales. Questions of two
# answers (0 no, 1 yes), of four (1 to 4) and of five (-2 to +2) alternate. Group A agrees
# with block X (q1 to q4) and disagrees with block Y (q5 to q9); group B does the opposite.
RESULT_SCHEMA_V2 = "selene.coclustering.single.result/2"
SURVEY_SCALES = (
    ("yes-no", (0, 1)),
    ("agree-4", (1, 2, 3, 4)),
    ("agree-5", (-2, -1, 0, 1, 2)),
)
SURVEY_ITEMS = (
    ("q1", "yes-no"),
    ("q2", "agree-4"),
    ("q3", "agree-5"),
    ("q4", "yes-no"),
    ("q5", "agree-4"),
    ("q6", "agree-5"),
    ("q7", "yes-no"),
    ("q8", "agree-4"),
    ("q9", "agree-5"),
)
SURVEY_TABLE = (
    ("vtr-01", (1, 4, 2, 1, 1, -2, 0, 1, -2)),
    ("vtr-02", (1, 3, 2, "A", 2, -1, 0, 1, -2)),
    ("vtr-03", (1, 4, 1, 1, 1, "-", 0, 2, -1)),
    ("vtr-04", ("A", 4, 2, 1, 1, -2, 0, 1, -1)),
    ("vtr-05", (1, "-", 2, 1, 2, -2, "A", 1, -2)),
    ("vtr-06", (1, 4, 2, 0, 1, -1, 0, 1, -2)),
    ("vtr-07", (0, 1, -2, 0, 4, 2, 1, 4, 2)),
    ("vtr-08", (0, 2, -1, "A", 4, 2, 1, 3, 1)),
    ("vtr-09", (0, 1, -2, 0, 3, "-", 1, 4, 2)),
    ("vtr-10", ("A", 1, -2, 0, 4, 1, 1, 4, 2)),
    ("vtr-11", (0, "-", -1, 0, 4, 2, "A", 4, 2)),
    ("vtr-12", (1, 1, -2, 0, 4, 2, 1, 3, 2)),
)
SURVEY_PLANTED_GROUPS = (0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)
SURVEY_PLANTED_BLOCKS = (0, 0, 0, 0, 1, 1, 1, 1, 1)
SURVEY_SEED = 20261126


def contract_version() -> str:
    """Return ``info.version`` of openapi.yaml."""
    version: str = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))["info"]["version"]
    return version


def contract_defaults(schema_version: int = 1) -> dict[str, Any]:
    """Return the parameter defaults that openapi.yaml declares for a payload schema."""
    document = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    name = f"CoclusteringSingleV{schema_version}Parameters"
    properties = document["components"]["schemas"][name]["properties"]
    defaults: dict[str, Any] = {}
    for setting, spec in properties.items():
        if "default" in spec:
            defaults[setting] = spec["default"]
        else:
            defaults[setting] = {
                inner: inner_spec["default"] for inner, inner_spec in spec["properties"].items()
            }
    return defaults


def reference_submission() -> dict[str, Any]:
    """Return the reference submission: known exposure, neutral votes and abstentions."""
    votes: list[list[int]] = []
    shown_without_vote: list[list[int]] = []
    for row, (_voter, cells) in enumerate(TABLE):
        for column, cell in enumerate(cells):
            if cell == "A":
                shown_without_vote.append([row, column])
            elif isinstance(cell, int):
                votes.append([row, column, cell])
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "submission_key": "test-only:reference:0001",
        "caller_reference": "test-only/generation-1",
        "payload": {
            "scope": {"project": "prj-test-only"},
            "scale": {"scores": list(PRODUCT_SCORES)},
            "voters": [voter for voter, _cells in TABLE],
            "items": [{"id": item, "revision": f"{item}-rev-1"} for item in ITEMS],
            "votes": votes,
            "exposure": {"mode": "listed", "shown_without_vote": shown_without_vote},
            "parameters": {"n_groups": 2, "n_blocks": 2, "initialization": "spectral"},
            "seed": SEED,
        },
    }


def no_votes_submission() -> dict[str, Any]:
    """Return a submission whose answer is an explained absence: no vote at all."""
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "submission_key": "test-only:no-votes:0001",
        "payload": {
            "scope": {"project": "prj-test-only"},
            "scale": {"scores": list(PRODUCT_SCORES)},
            "voters": ["vtr-01", "vtr-02", "vtr-03"],
            "items": [
                {"id": "itm-a", "revision": "itm-a-rev-1"},
                {"id": "itm-b", "revision": "itm-b-rev-1"},
            ],
            "votes": [],
            "exposure": {"mode": "complete"},
            "seed": 0,
        },
    }


def normalised(submission: dict[str, Any]) -> dict[str, Any]:
    """Return the normalised content of a submission, as the contract defines it.

    The kind, the schema version and the payload, with every parameter at its effective
    value, the votes sorted by cell and the listed cells sorted. The key and the caller
    reference are not content.
    """
    payload = copy.deepcopy(submission["payload"])
    version = submission["schema_version"]
    parameters = {**contract_defaults(version), **payload.get("parameters", {})}
    if version == 2:
        parameters["thresholds"] = {
            **contract_defaults(2)["thresholds"],
            **payload.get("parameters", {}).get("thresholds", {}),
        }
        declared = {scale["id"]: scale["scores"] for scale in payload["scales"]}
        used = list(dict.fromkeys(item["scale"] for item in payload["items"]))
        payload["scales"] = [{"id": scale, "scores": declared[scale]} for scale in used]
    parameters["tolerance"] = float(parameters["tolerance"])
    payload["parameters"] = parameters
    payload["votes"] = sorted(payload["votes"])
    if payload["exposure"]["mode"] == "listed":
        payload["exposure"]["shown_without_vote"] = sorted(
            payload["exposure"]["shown_without_vote"]
        )
    return {
        "kind": submission["kind"],
        "schema_version": submission["schema_version"],
        "payload": payload,
    }


def canonical_json(value: object) -> bytes:
    """Serialise to the canonical JSON the content fingerprint hashes."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def content_fingerprint(submission: dict[str, Any]) -> str:
    """Return the content fingerprint of a submission."""
    return "sha256:" + hashlib.sha256(canonical_json(normalised(submission))).hexdigest()


def engine_estimate(submission: dict[str, Any]) -> Any:
    """Compute the submission locally, by calling the engine directly."""
    payload = normalised(submission)["payload"]
    voters = payload["voters"]
    items = [item["id"] for item in payload["items"]]
    records = [(voters[row], items[column], score) for row, column, score in payload["votes"]]
    exposure = payload["exposure"]
    if exposure["mode"] == "complete":
        shown = None
    elif exposure["mode"] == "listed":
        shown = [(voters[row], items[column]) for row, column in exposure["shown_without_vote"]]
    else:
        raise ValueError(f"exposure {exposure['mode']!r} has no local computation")
    parameters = payload["parameters"]
    response_model: Any = {"ordinal_probit": OrdinalProbit, "categorical": Categorical}[
        parameters["response_model"]
    ]()
    if submission["schema_version"] == 2:
        scales = {scale["id"]: OrdinalScale(tuple(scale["scores"])) for scale in payload["scales"]}
        votes = VoteMatrix.from_records(
            records,
            shown=shown,
            voters=voters,
            items=items,
            item_scales={item["id"]: scales[item["scale"]] for item in payload["items"]},
        )
        thresholds = parameters["thresholds"]
        if parameters["response_model"] == "ordinal_probit" and thresholds != {
            "axis": "auto",
            "mode": "fixed",
        }:
            response_model = OrdinalProbit(thresholds=Thresholds(**thresholds))
    else:
        votes = VoteMatrix.from_records(
            records, shown=shown, voters=voters, items=items, scale=OrdinalScale(PRODUCT_SCORES)
        )
    model = LatentBlockModel(
        n_groups=parameters["n_groups"],
        n_blocks=parameters["n_blocks"],
        n_starts=parameters["n_starts"],
        max_iterations=parameters["max_iterations"],
        tolerance=parameters["tolerance"],
        response_model=response_model,
        initialization={"random": RandomStart, "spectral": SpectralStart}[
            parameters["initialization"]
        ](),
    )
    return model.estimate(votes, seed=payload["seed"])


VALUE_FIELDS = (
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
# The fields of the table of co-clusters in result/1; the engine may write more.
COCLUSTER_FIELDS = (
    "scores",
    "vote_rate",
    "vote_shares",
    "n_votes",
    "latent_mean",
    "latent_spread",
)


def result_document(result: Any) -> dict[str, Any]:
    """Return the engine result in the shape of CoclusteringSingleV1Result."""
    data = result.to_dict()
    provenance = data["provenance"]
    document: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "available": data["available"],
        "provenance": {
            "method": provenance["method"],
            "engine_version": provenance["engine_version"],
            "parameters": provenance["parameters"],
            "input_fingerprint": provenance["input_fingerprint"],
            "randomness": provenance["randomness"],
            "dependencies": provenance["dependencies"],
            "fingerprint": provenance["fingerprint"],
        },
    }
    if data["available"]:
        value = {name: data["value"][name] for name in VALUE_FIELDS}
        value["coclusters"] = {
            name: field
            for name, field in data["value"]["coclusters"].items()
            if name in COCLUSTER_FIELDS
        }
        document["value"] = value
    else:
        document["unavailable"] = data["unavailable"]
    return document


def result_document_v2(result: Any, submission: dict[str, Any]) -> dict[str, Any]:
    """Return the engine result in the shape of CoclusteringSingleV2Result.

    Written here from the engine value, apart from the adapter of the service, so that
    the fixture checks the adapter rather than copies it.
    """
    data = result.to_dict()
    provenance = data["provenance"]
    document: dict[str, Any] = {
        "schema": RESULT_SCHEMA_V2,
        "available": data["available"],
        "provenance": {
            name: provenance[name]
            for name in (
                "method",
                "engine_version",
                "parameters",
                "input_fingerprint",
                "randomness",
                "dependencies",
                "fingerprint",
            )
        },
    }
    if not data["available"]:
        document["unavailable"] = data["unavailable"]
        return document
    value = data["value"]
    table = value["coclusters"]
    identifiers = {tuple(scale["scores"]): scale["id"] for scale in submission["payload"]["scales"]}
    if "scales" in value:
        scores = [tuple(one) for one in value["scales"]]
        scale_index = value["scale_index"]
        shares = table["vote_shares_by_scale"]
    else:
        scores = [tuple(value["scores"])]
        scale_index = [0] * len(value["items"])
        shares = [table["vote_shares"]]
    coclusters = {
        name: table[name]
        for name in (
            "vote_rate",
            "n_votes",
            "latent_mean",
            "latent_spread",
            "thresholds",
            "has_fixed_spread",
        )
        if name in table
    }
    coclusters["vote_shares_by_scale"] = shares
    names = (
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
    document["value"] = {
        **{name: value[name] for name in names},
        "scales": [{"id": identifiers[one], "scores": list(one)} for one in scores],
        "scale_index": scale_index,
        "coclusters": coclusters,
    }
    return document


def survey_submission() -> dict[str, Any]:
    """Return the reference submission of payload schema 2: items on three scales."""
    votes: list[list[int]] = []
    shown_without_vote: list[list[int]] = []
    for row, (_voter, cells) in enumerate(SURVEY_TABLE):
        for column, cell in enumerate(cells):
            if cell == "A":
                shown_without_vote.append([row, column])
            elif isinstance(cell, int):
                votes.append([row, column, cell])
    return {
        "kind": KIND,
        "schema_version": 2,
        "submission_key": "test-only:survey:0001",
        "caller_reference": "test-only/survey-1",
        "payload": {
            "scope": {"project": "prj-test-only"},
            "scales": [{"id": scale, "scores": list(scores)} for scale, scores in SURVEY_SCALES],
            "voters": [voter for voter, _cells in SURVEY_TABLE],
            "items": [
                {"id": item, "revision": f"{item}-rev-1", "scale": scale}
                for item, scale in SURVEY_ITEMS
            ],
            "votes": votes,
            "exposure": {"mode": "listed", "shown_without_vote": shown_without_vote},
            "parameters": {"n_groups": 2, "n_blocks": 2, "initialization": "spectral"},
            "seed": SURVEY_SEED,
        },
    }


def invalid_survey_submissions(base: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the negative submissions proper to payload schema 2."""

    def variant(change: Any) -> dict[str, Any]:
        body = copy.deepcopy(base)
        change(body)
        return body

    def unknown_scale(body: dict[str, Any]) -> None:
        body["payload"]["items"][0]["scale"] = "agree-7"

    def repeated_scale_id(body: dict[str, Any]) -> None:
        body["payload"]["scales"][1]["id"] = body["payload"]["scales"][0]["id"]

    def repeated_scores(body: dict[str, Any]) -> None:
        body["payload"]["scales"].append({"id": "yes-no-again", "scores": [0, 1]})

    def unused_scale(body: dict[str, Any]) -> None:
        body["payload"]["scales"].append({"id": "agree-7", "scores": [1, 2, 3, 4, 5, 6, 7]})

    def off_its_scale(body: dict[str, Any]) -> None:
        # A score of the scale of five answers, on an item of two.
        row, column, _score = next(vote for vote in body["payload"]["votes"] if vote[1] == 0)
        body["payload"]["votes"] = [
            vote for vote in body["payload"]["votes"] if (vote[0], vote[1]) != (row, column)
        ] + [[row, column, -2]]

    def scores_not_increasing(body: dict[str, Any]) -> None:
        body["payload"]["scales"][1]["scores"] = [1, 3, 2, 4]

    def one_score(body: dict[str, Any]) -> None:
        body["payload"]["scales"][0]["scores"] = [1]

    def unknown_axis(body: dict[str, Any]) -> None:
        body["payload"]["parameters"]["thresholds"] = {"axis": "log"}

    cases: dict[str, tuple[bool, int, str, Any]] = {
        # name: (valid against the JSON schema, status, code, change)
        "unknown-scale": (True, 422, "invalid_request", unknown_scale),
        "repeated-scale-id": (True, 422, "invalid_request", repeated_scale_id),
        "repeated-scores": (True, 422, "invalid_request", repeated_scores),
        "unused-scale": (True, 422, "invalid_request", unused_scale),
        "score-off-its-scale": (True, 422, "invalid_request", off_its_scale),
        "scores-not-increasing": (True, 422, "invalid_request", scores_not_increasing),
        "one-score": (False, 422, "invalid_request", one_score),
        "unknown-axis": (False, 422, "invalid_request", unknown_axis),
    }
    return {
        name: {
            "description": f"TEST ONLY. Negative case {name!r} of payload schema 2.",
            "schema_valid": schema_valid,
            "expect": {"status": status, "code": code},
            "body": variant(change),
        }
        for name, (schema_valid, status, code, change) in cases.items()
    }


def reference_document(submission: dict[str, Any], result: Any) -> dict[str, Any]:
    """Return what a computation of ``submission`` must reproduce, within ``tolerance``."""
    document = (
        result_document_v2(result, submission)
        if submission["schema_version"] == 2
        else result_document(result)
    )
    reference: dict[str, Any] = {
        "description": (
            "TEST ONLY. Synthetic reference computed locally with the engine. A service "
            "computation of the same submission with the same engine wheel reproduces "
            "it: identical partitions, numbers within the tolerance."
        ),
        "content_fingerprint": content_fingerprint(submission),
        "method": document["provenance"]["method"],
        "input_fingerprint": document["provenance"]["input_fingerprint"],
        "computation": document["provenance"]["fingerprint"],
        "available": document["available"],
        "tolerance": TOLERANCE,
    }
    if document["available"]:
        value = document["value"]
        reference["expected"] = {
            "group_index": value["group_index"],
            "block_index": value["block_index"],
            "objective": value["objective"],
            "integrated_likelihood": value["integrated_likelihood"],
            "group_memberships": value["group_memberships"],
            "block_memberships": value["block_memberships"],
        }
    else:
        reference["expected"] = {"reason": document["unavailable"]["reason"]}
    return reference


def same_partition(found: list[int], planted: tuple[int, ...]) -> bool:
    """Whether two partitions are equal up to the names of their classes."""
    pairs = set(zip(found, planted, strict=True))
    return len(pairs) == len(set(found)) == len(set(planted))


def invalid_submissions(base: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return negative submissions, each with the refusal the contract requires."""

    def variant(change: Any) -> dict[str, Any]:
        body = copy.deepcopy(base)
        change(body)
        return body

    def set_path(*path: Any, value: Any) -> Any:
        def change(body: dict[str, Any]) -> None:
            target: Any = body
            for step in path[:-1]:
                target = target[step]
            target[path[-1]] = value

        return change

    def drop(*path: Any) -> Any:
        def change(body: dict[str, Any]) -> None:
            target: Any = body
            for step in path[:-1]:
                target = target[step]
            del target[path[-1]]

        return change

    def duplicate_vote(body: dict[str, Any]) -> None:
        row, column, _score = body["payload"]["votes"][0]
        body["payload"]["votes"].append([row, column, 1])

    def duplicate_item(body: dict[str, Any]) -> None:
        body["payload"]["items"][1]["id"] = body["payload"]["items"][0]["id"]

    def voted_and_listed(body: dict[str, Any]) -> None:
        row, column, _score = body["payload"]["votes"][0]
        body["payload"]["exposure"]["shown_without_vote"].append([row, column])

    cases: dict[str, tuple[bool, int, str, Any]] = {
        # name: (valid against the JSON schema, status, code, change)
        "unknown-kind": (False, 422, "unsupported_kind", set_path("kind", value="spectral.embed")),
        # 0 is never a payload schema version: the case stays true when versions are added.
        "schema-version-0": (
            False,
            422,
            "unsupported_schema_version",
            set_path("schema_version", value=0),
        ),
        "exposure-unknown": (
            True,
            422,
            "unsupported_exposure",
            set_path("payload", "exposure", value={"mode": "unknown"}),
        ),
        "unknown-field": (False, 422, "invalid_request", set_path("payload", "notes", value="x")),
        "missing-seed": (False, 422, "invalid_request", drop("payload", "seed")),
        "score-off-scale": (
            False,
            422,
            "invalid_request",
            set_path("payload", "votes", 0, 2, value=3),
        ),
        "score-as-text": (
            False,
            422,
            "invalid_request",
            set_path("payload", "votes", 0, 2, value="2"),
        ),
        "engine-scale": (
            False,
            422,
            "invalid_request",
            set_path("payload", "scale", "scores", value=[1, 2, 3, 4, 5]),
        ),
        "voter-looks-like-email": (
            False,
            422,
            "invalid_request",
            set_path("payload", "voters", 0, value="someone@example.org"),
        ),
        "duplicate-voter": (
            False,
            422,
            "invalid_request",
            set_path("payload", "voters", 1, value="vtr-01"),
        ),
        "too-many-groups": (
            False,
            422,
            "invalid_request",
            set_path("payload", "parameters", "n_groups", value=11),
        ),
        "seed-negative": (False, 422, "invalid_request", set_path("payload", "seed", value=-1)),
        "short-submission-key": (
            False,
            422,
            "invalid_request",
            set_path("submission_key", value="short"),
        ),
        # Valid against the JSON schema, refused by the semantic checks of the service.
        "duplicate-vote": (True, 422, "invalid_request", duplicate_vote),
        "duplicate-item": (True, 422, "invalid_request", duplicate_item),
        "vote-row-out-of-range": (
            True,
            422,
            "invalid_request",
            set_path("payload", "votes", 0, 0, value=len(TABLE)),
        ),
        "listed-cell-out-of-range": (
            True,
            422,
            "invalid_request",
            set_path("payload", "exposure", "shown_without_vote", 0, 1, value=len(ITEMS)),
        ),
        "voted-cell-listed": (True, 422, "invalid_request", voted_and_listed),
    }
    return {
        name: {
            "description": f"TEST ONLY. Negative case {name!r} of contract v1.",
            "schema_valid": schema_valid,
            "expect": {"status": status, "code": code},
            "body": variant(change),
        }
        for name, (schema_valid, status, code, change) in cases.items()
    }


def example_jobs(reference: dict[str, Any], result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return examples of every job state, for the schema tests and the readers."""
    versions = {
        "service_version": "0.1.0",
        "contract_version": contract_version(),
        "engine_version": result["provenance"]["engine_version"],
    }
    common = {
        "id": "job_TESTONLY0000000000000001",
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "submission_key": "test-only:reference:0001",
        "caller_reference": "test-only/generation-1",
        "content_fingerprint": reference["content_fingerprint"],
        "created_at": "2026-11-25T12:00:00Z",
        "service": versions,
    }
    return {
        "job-running": {
            **common,
            "state": "running",
            "started_at": "2026-11-25T12:00:00Z",
            "finished_at": None,
            "expires_at": None,
            "result": None,
            "error": None,
        },
        "job-succeeded": {
            **common,
            "state": "succeeded",
            "started_at": "2026-11-25T12:00:00Z",
            "finished_at": "2026-11-25T12:00:02Z",
            "expires_at": "2026-11-25T12:15:02Z",
            "result": result,
            "error": None,
        },
        "job-failed-timeout": {
            **common,
            "state": "failed",
            "started_at": "2026-11-25T12:00:00Z",
            "finished_at": "2026-11-25T12:02:00Z",
            "expires_at": "2026-11-25T12:17:00Z",
            "result": None,
            "error": {
                "code": "timeout",
                "message": "The computation ran longer than 120 seconds and was stopped.",
                "retryable": False,
            },
        },
    }


def example_errors() -> dict[str, dict[str, Any]]:
    """Return examples of error bodies."""
    return {
        "error-busy": {
            "error": {
                "code": "busy",
                "message": "The computation slot is taken; retry later.",
                "retryable": True,
            }
        },
        "error-conflict": {
            "error": {
                "code": "submission_conflict",
                "message": "This submission key is bound to a retained job with other content.",
                "retryable": False,
            }
        },
        "error-invalid": {
            "error": {
                "code": "invalid_request",
                "message": "The submission does not follow the contract.",
                "retryable": False,
                "fields": [{"path": "payload.votes[0][2]", "issue": "not a product score"}],
            }
        },
        "error-job-not-found": {
            "error": {
                "code": "job_not_found",
                "message": "No retained job has this identifier.",
                "retryable": False,
            }
        },
        "error-unauthenticated": {
            "error": {
                "code": "unauthenticated",
                "message": "A valid machine token is required.",
                "retryable": False,
            }
        },
    }


def rendered() -> dict[pathlib.Path, str]:
    """Return every fixture file and its content."""
    files: dict[pathlib.Path, str] = {}

    def put(relative: str, value: object) -> None:
        files[FIXTURES / relative] = json.dumps(value, indent=2, ensure_ascii=False) + "\n"

    reference = reference_submission()
    result = engine_estimate(reference)
    value = result.value
    if not (
        same_partition(value.group_index.tolist(), PLANTED_GROUPS)
        and same_partition(value.block_index.tolist(), PLANTED_BLOCKS)
    ):
        sys.exit("the engine does not recover the planted groups and blocks of the reference")
    put("coclustering-single-1/reference.submission.json", reference)
    put("coclustering-single-1/reference.expected.json", reference_document(reference, result))

    no_votes = no_votes_submission()
    no_votes_result = engine_estimate(no_votes)
    put("coclustering-single-1/no-votes.submission.json", no_votes)
    put(
        "coclustering-single-1/no-votes.expected.json",
        reference_document(no_votes, no_votes_result),
    )

    for name, case in invalid_submissions(reference).items():
        put(f"coclustering-single-1/invalid/{name}.json", case)
    for name, job in example_jobs(
        reference_document(reference, result), result_document(result)
    ).items():
        put(f"examples/{name}.json", job)
    for name, error in example_errors().items():
        put(f"examples/{name}.json", error)

    survey = survey_submission()
    survey_result = engine_estimate(survey)
    survey_value = survey_result.value
    if not (
        same_partition(survey_value.group_index.tolist(), SURVEY_PLANTED_GROUPS)
        and same_partition(survey_value.block_index.tolist(), SURVEY_PLANTED_BLOCKS)
    ):
        sys.exit("the engine does not recover the planted groups and blocks of the survey")
    put("coclustering-single-2/reference.submission.json", survey)
    put(
        "coclustering-single-2/reference.expected.json",
        reference_document(survey, survey_result),
    )
    put(
        "examples/job-succeeded-scales.json",
        {
            **example_jobs(reference_document(reference, result), result_document(result))[
                "job-succeeded"
            ],
            "id": "job_TESTONLY0000000000000002",
            "schema_version": 2,
            "submission_key": survey["submission_key"],
            "caller_reference": survey["caller_reference"],
            "content_fingerprint": content_fingerprint(survey),
            "result": result_document_v2(survey_result, survey),
        },
    )
    for name, case in invalid_survey_submissions(survey).items():
        put(f"coclustering-single-2/invalid/{name}.json", case)
    return files


# Fields that describe how a fixture was computed rather than what it is: the environment
# (another Python, NumPy or engine release writes other values for the same numbers) and
# the history of the iterations (from a start built on singular vectors, another
# processor may follow another path to the same optimum). The reference never promises
# them; it promises the partition, the optimum and the memberships.
ENVIRONMENT_KEYS = frozenset({"dependencies", "engine_version", "history"})


def canonical(document: object) -> object:
    """Rename the groups and blocks of every co-clustering value of a fixture canonically.

    Another processor may find the same partition under other names (see
    ``selene_contract.canonical_value``): fixtures compare partitions, not names.
    """
    if isinstance(document, dict):
        if "group_index" in document and "block_index" in document:
            document = canonical_value(document)
        return {key: canonical(value) for key, value in document.items()}
    if isinstance(document, list):
        return [canonical(value) for value in document]
    return document


def equivalent(found: object, stored: object) -> bool:
    """Whether a fixture computed here matches the stored one.

    Partitions match up to the names of their groups and blocks; integers, texts,
    booleans and structure match exactly; numbers with a fraction match within
    ``TOLERANCE``, since another processor may round the last bits differently;
    ``ENVIRONMENT_KEYS`` are not compared.
    """
    return first_difference(found, stored) is None


def first_difference(found: object, stored: object) -> str | None:
    """Return where two fixtures first differ once canonical, with both values, or ``None``."""
    return _difference(canonical(found), canonical(stored), "$")


def _difference(found: object, stored: object, path: str) -> str | None:
    if isinstance(stored, dict):
        if not isinstance(found, dict) or found.keys() != stored.keys():
            return f"{path}: other fields"
        for key in stored:
            if key not in ENVIRONMENT_KEYS:
                difference = _difference(found[key], stored[key], f"{path}.{key}")
                if difference:
                    return difference
        return None
    if isinstance(stored, list):
        if not isinstance(found, list) or len(found) != len(stored):
            return f"{path}: other length"
        for index, (one, other) in enumerate(zip(found, stored, strict=True)):
            difference = _difference(one, other, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if isinstance(stored, float) or isinstance(found, float):
        same = (
            isinstance(found, (int, float))
            and isinstance(stored, (int, float))
            and not isinstance(found, bool)
            and not isinstance(stored, bool)
            and math.isclose(
                found, stored, rel_tol=TOLERANCE["relative"], abs_tol=TOLERANCE["absolute"]
            )
        )
    else:
        same = type(found) is type(stored) and found == stored
    return None if same else f"{path}: computed {found!r}, stored {stored!r}"


def stale_fixtures() -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    """Return the fixtures that differ from what the engine computes now, and the extra ones."""
    files = rendered()
    stale = [
        path
        for path, content in files.items()
        if not path.exists()
        or not equivalent(json.loads(content), json.loads(path.read_text(encoding="utf-8")))
    ]
    extra = sorted(path for path in FIXTURES.rglob("*.json") if path not in files)
    return stale, extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if a fixture is stale")
    arguments = parser.parse_args()
    if arguments.check:
        stale, extra = stale_fixtures()
        files = rendered()
        for path in stale:
            stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            where = first_difference(json.loads(files[path]), stored)
            print(f"stale fixture: {path.relative_to(HERE)}: {where}")
        for path in extra:
            print(f"extra fixture: {path.relative_to(HERE)}")
        sys.exit(1 if stale or extra else 0)
    files = rendered()
    for path in FIXTURES.rglob("*.json"):
        if path not in files:
            path.unlink()
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(f"wrote {len(files)} fixtures under {FIXTURES.relative_to(HERE.parent.parent.parent)}")


if __name__ == "__main__":
    main()
