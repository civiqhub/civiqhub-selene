# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the tests of the service. TEST ONLY: never imported by the service.

:class:`FakeRunner` stands in for the computation process where a test needs to hold a
job running or to choose its outcome. It is not a computation: the service itself only
ever runs :class:`selene_api.runner.ProcessRunner`.
"""

import asyncio
import contextlib
import dataclasses
import pathlib
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from selene_api.app import Service, create_app
from selene_api.config import Settings
from selene_api.runner import ProcessRunner, Runner, RunOutcome
from selene_api.security import TokenStore
from selene_api.versions import BuildInfo, Versions

TOKEN = "test-only-token-" + "0123456789abcdef" * 2
"""A synthetic machine token of 48 characters."""

AUTH = {"Authorization": f"Bearer {TOKEN}"}


def settings_for(directory: pathlib.Path, **overrides: Any) -> Settings:
    """Return loopback settings with a token file in ``directory``."""
    token_file = directory / "token"
    token_file.write_text(TOKEN + "\n", encoding="ascii")
    settings = Settings(token_file=token_file, host="127.0.0.1", port=18443, metrics_port=19464)
    return dataclasses.replace(settings, **overrides)


def service_for(settings: Settings, runner: Runner | None = None) -> Service:
    """Assemble a service around ``runner``, by default the real process runner."""
    runner = runner or ProcessRunner(max_result_bytes=settings.max_result_bytes)
    versions = Versions.current(BuildInfo())
    return Service.build(settings, TokenStore(settings.token_file), runner, versions)


@contextlib.asynccontextmanager
async def asgi_client(service: Service) -> AsyncIterator[httpx.AsyncClient]:
    """Run the application in process, with its start-up and shutdown, and a client."""
    app = create_app(service)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://selene.test") as client:
            yield client


class FakeRunner:
    """A runner whose computations wait for :meth:`release`, then return ``outcome``."""

    def __init__(self, outcome: RunOutcome | None = None) -> None:
        self.outcome = outcome or RunOutcome(result={"schema": "fake", "available": True})
        self.started: list[Mapping[str, Any]] = []
        self.cancelled = 0
        self.timeouts: list[float] = []
        self._gate = asyncio.Event()
        self.hold = False

    def release(self) -> None:
        """Let every held computation finish."""
        self._gate.set()

    async def run(self, task: Mapping[str, Any], *, timeout: float) -> RunOutcome:
        self.started.append(task)
        self.timeouts.append(timeout)
        try:
            if self.hold:
                await self._gate.wait()
            else:
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return self.outcome

    async def describe(self, *, timeout: float) -> dict[str, Any]:
        from civiqhub_selene import __version__

        return {
            "engine_version": __version__,
            "kinds": [
                {
                    "kind": "coclustering.single",
                    "schema_version": 1,
                    "method": {"name": "latent_block_model", "version": "fake"},
                }
            ],
        }


async def settle(predicate: Any, *, timeout: float = 30.0, interval: float = 0.01) -> None:
    """Wait until ``predicate()`` is true."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("the condition did not hold in time")
        await asyncio.sleep(interval)
