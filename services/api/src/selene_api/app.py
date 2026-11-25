# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""HTTP layer of contract v1: authentication, limits, validation, errors and routes.

The routes know transfer objects and the job manager, never an engine class. Checks of a
submission run in the order the contract fixes: authentication, media type, size, JSON,
kind and schema version, schema, semantics, key, capacity. Every error answer follows
the error schema and never echoes the request. Probes need no token and never report
whether a computation runs.
"""

import json
import logging
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from selene_api import logs
from selene_api.adapters import ADAPTERS, adapter_for, kinds
from selene_api.config import Settings
from selene_api.errors import ApiError, FieldIssue
from selene_api.jobs import BusyError, ConflictError, Job, JobManager, ShuttingDownError
from selene_api.runner import Runner, RunOutcome
from selene_api.security import TokenStore, bearer_token
from selene_api.versions import Versions

__all__ = ["BUSY_RETRY_AFTER_SECONDS", "Service", "create_app"]

BUSY_RETRY_AFTER_SECONDS = 5
"""The ``Retry-After`` of a busy answer."""

_JOB_ID = re.compile(r"^job_[A-Za-z0-9_-]{16,64}$")
_PROBES = frozenset({"/healthz", "/readyz"})
_STARTUP_TIMEOUT_SECONDS = 120.0

Phase = Literal["starting", "ready", "shutting_down"]


@dataclass
class Service:
    """What the routes share: settings, tokens, jobs, runner and versions."""

    settings: Settings
    tokens: TokenStore
    runner: Runner
    versions: Versions
    manager: JobManager
    phase: Phase = "starting"
    methods: dict[tuple[str, int], dict[str, str]] = field(default_factory=dict)

    @classmethod
    def build(
        cls, settings: Settings, tokens: TokenStore, runner: Runner, versions: Versions
    ) -> "Service":
        """Assemble a service and its job manager from its parts."""
        manager = JobManager(
            runner,
            job_timeout_seconds=settings.job_timeout_seconds,
            result_ttl_seconds=settings.result_ttl_seconds,
            max_retained_jobs=settings.max_retained_jobs,
            on_finish=_log_finish,
        )
        return cls(settings, tokens, runner, versions, manager)

    async def start(self) -> None:
        """Check that a computation process starts and serves every kind, then get ready.

        Raises:
            RuntimeError: If the computation process does not describe the same engine
                and every adapter.
        """
        description = await self.runner.describe(timeout=_STARTUP_TIMEOUT_SECONDS)
        if description.get("engine_version") != self.versions.engine:
            raise RuntimeError("the computation process runs another engine version")
        methods = {
            (entry["kind"], entry["schema_version"]): entry["method"]
            for entry in description.get("kinds", [])
        }
        if set(methods) != set(ADAPTERS):
            raise RuntimeError("the computation process does not serve every kind")
        self.methods = methods
        self.phase = "ready"

    async def stop(self) -> None:
        """Refuse new jobs, stop the running computation and forget every job."""
        self.phase = "shutting_down"
        await self.manager.shutdown()


def _log_finish(job: Job, outcome: RunOutcome | None) -> None:
    error = job.error or {}
    logs.event(
        "job_finished",
        job_id=job.id,
        kind=job.kind,
        state=job.state,
        code=error.get("code"),
        peak_rss_bytes=None if outcome is None else outcome.peak_rss_bytes,
    )


# Requests -----------------------------------------------------------------------------


def _authenticate(service: Service, request: Request) -> None:
    if not service.tokens.verify(bearer_token(request.headers.get("authorization"))):
        raise ApiError.unauthenticated()


async def _body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > limit:
                raise ApiError.too_large()
        except ValueError:
            raise ApiError.malformed() from None
    received = bytearray()
    async for chunk in request.stream():
        received += chunk
        if len(received) > limit:
            raise ApiError.too_large()
    return bytes(received)


def _no_constant(name: str) -> object:
    raise ValueError(f"{name} is not JSON")


def _decoded(raw: bytes) -> Any:
    try:
        return json.loads(raw, parse_constant=_no_constant)
    except (ValueError, RecursionError):
        raise ApiError.malformed() from None


def _adapter(document: Any) -> Any:
    if not isinstance(document, dict):
        raise ApiError.invalid([FieldIssue("body", "not an object")])
    kind = document.get("kind")
    version = document.get("schema_version")
    if not isinstance(kind, str):
        raise ApiError.invalid([FieldIssue("kind", "required" if kind is None else "not a text")])
    if kind not in kinds():
        raise ApiError.unsupported_kind()
    if type(version) is not int:
        issue = "required" if version is None else "not an integer"
        raise ApiError.invalid([FieldIssue("schema_version", issue)])
    adapter = adapter_for(kind, version)
    if adapter is None:
        raise ApiError.unsupported_schema_version()
    return adapter


def _service(request: Request) -> Service:
    service: Service = request.app.state.service
    return service


def _json(body: object, status: int = 200, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(body, status_code=status, headers=headers)


# Routes -------------------------------------------------------------------------------


async def health(request: Request) -> Response:
    return _json({"status": "ok"})


async def readiness(request: Request) -> Response:
    phase = _service(request).phase
    return _json({"status": phase}, 200 if phase == "ready" else 503)


async def service_info(request: Request) -> Response:
    service = _service(request)
    _authenticate(service, request)
    settings = service.settings
    versions = service.versions
    manager = service.manager
    return _json(
        {
            "service": {"name": "selene-api", "version": versions.service},
            "contract": {"version": versions.contract},
            "engine": {
                "distribution": "civiqhub-selene",
                "version": versions.engine,
                "wheel_sha256": versions.build.engine_wheel_sha256,
            },
            "build": {"commit": versions.build.commit, "source": versions.build.source},
            "deployment": {"environment": settings.environment},
            "kinds": [
                {
                    "kind": kind,
                    "schema_versions": schema_versions,
                    "method": service.methods.get((kind, schema_versions[-1]), {}),
                }
                for kind, schema_versions in kinds().items()
            ],
            "limits": {
                "max_body_bytes": settings.max_body_bytes,
                "max_voters": settings.max_voters,
                "max_items": settings.max_items,
                "max_cells": settings.max_cells,
                "max_work": settings.max_work,
                "job_timeout_seconds": settings.job_timeout_seconds,
                "result_ttl_seconds": settings.result_ttl_seconds,
                "max_retained_jobs": settings.max_retained_jobs,
                "max_result_bytes": settings.max_result_bytes,
            },
            "capacity": {
                "slots": manager.slots,
                "running": manager.running,
                "accepting": service.phase == "ready" and manager.accepting,
            },
        }
    )


async def submit(request: Request) -> Response:
    service = _service(request)
    _authenticate(service, request)
    if service.phase == "starting":
        raise ApiError.not_ready()
    if service.phase == "shutting_down":
        raise ApiError.shutting_down()
    media = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if media != "application/json":
        raise ApiError.media_type()
    raw = await _body(request, service.settings.max_body_bytes)
    adapter = _adapter(_decoded(raw))
    task = adapter.prepare(raw, service.settings.limits)
    try:
        submitted = service.manager.submit(task)
    except ConflictError:
        logs.event("job_conflict", kind=task.kind)
        raise ApiError.conflict() from None
    except BusyError:
        logs.event("job_busy", kind=task.kind)
        raise ApiError.busy(BUSY_RETRY_AFTER_SECONDS) from None
    except ShuttingDownError:
        raise ApiError.shutting_down() from None
    job = submitted.job
    document = job.document(service.versions.of_job())
    if submitted.replayed:
        logs.event("job_replayed", job_id=job.id, kind=job.kind, state=job.state)
        return _json(document, 200)
    logs.event("job_started", job_id=job.id, kind=job.kind, schema_version=job.schema_version)
    return _json(document, 202, {"Location": f"/v1/jobs/{job.id}"})


def _job_id(request: Request) -> str:
    job_id = str(request.path_params["job_id"])
    if not _JOB_ID.fullmatch(job_id):
        raise ApiError.job_not_found()
    return job_id


async def get_job(request: Request) -> Response:
    service = _service(request)
    _authenticate(service, request)
    job = service.manager.get(_job_id(request))
    if job is None:
        raise ApiError.job_not_found()
    return _json(job.document(service.versions.of_job()))


async def delete_job(request: Request) -> Response:
    service = _service(request)
    _authenticate(service, request)
    job_id = _job_id(request)
    if not await service.manager.delete(job_id):
        raise ApiError.job_not_found()
    logs.event("job_deleted", job_id=job_id)
    return Response(status_code=204)


# Errors and access log ----------------------------------------------------------------


async def _api_error(request: Request, error: Exception) -> Response:
    assert isinstance(error, ApiError)
    return _json(error.body(), error.status, dict(error.headers))


async def _http_error(request: Request, error: Exception) -> Response:
    assert isinstance(error, HTTPException)
    if error.status_code == 405:
        failure = ApiError.method_not_allowed()
    elif error.status_code == 404:
        failure = ApiError.not_found()
    else:
        failure = ApiError(error.status_code, "http_error", "The request was refused.")
    return _json(failure.body(), failure.status, dict(failure.headers))


async def _unexpected(request: Request, error: Exception) -> Response:
    logs.event("unexpected_error", logging.ERROR, exception=type(error).__name__)
    failure = ApiError.internal()
    return _json(failure.body(), failure.status)


class AccessLog:
    """Log each request by method, route template, status and duration; never its content."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in _PROBES:
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        status = 500

        async def sending(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, sending)
        finally:
            route = scope.get("route")
            logs.event(
                "request",
                method=scope.get("method"),
                route=getattr(route, "path", "unmatched"),
                status=status,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )


def create_app(service: Service) -> Starlette:
    """Return the ASGI application of contract v1 around ``service``."""

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        await service.start()
        logs.event(
            "service_ready",
            service_version=service.versions.service,
            contract_version=service.versions.contract,
            engine_version=service.versions.engine,
            environment=service.settings.environment,
        )
        try:
            yield
        finally:
            await service.stop()
            logs.event("service_stopped")

    handlers: dict[Any, Callable[[Request, Exception], Awaitable[Response]]] = {
        ApiError: _api_error,
        HTTPException: _http_error,
        Exception: _unexpected,
    }
    app = Starlette(
        routes=[
            Route("/healthz", health, methods=["GET"]),
            Route("/readyz", readiness, methods=["GET"]),
            Route("/v1/service", service_info, methods=["GET"]),
            Route("/v1/jobs", submit, methods=["POST"]),
            Route("/v1/jobs/{job_id}", get_job, methods=["GET"]),
            Route("/v1/jobs/{job_id}", delete_job, methods=["DELETE"]),
        ],
        exception_handlers=handlers,
        lifespan=lifespan,
    )
    app.state.service = service
    app.add_middleware(AccessLog)
    return app
