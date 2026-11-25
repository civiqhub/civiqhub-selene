# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Start the service: one Uvicorn server process, configured by :class:`Settings`.

One process and one event loop serve HTTP; computations run in child processes. There is
no reload and no multiplication of HTTP workers: the jobs live in this process's memory.
"""

import uvicorn

from selene_api import logs
from selene_api.app import Service, create_app
from selene_api.config import Settings
from selene_api.runner import ProcessRunner
from selene_api.security import TokenStore
from selene_api.versions import BuildInfo, Versions

__all__ = ["build_service", "serve", "uvicorn_config"]


def build_service(settings: Settings) -> Service:
    """Assemble the service from checked settings."""
    tokens = TokenStore(settings.token_file)
    runner = ProcessRunner(max_result_bytes=settings.max_result_bytes)
    versions = Versions.current(BuildInfo.read(settings.build_info_file))
    return Service.build(settings, tokens, runner, versions)


def uvicorn_config(settings: Settings, service: Service) -> uvicorn.Config:
    """Return the Uvicorn configuration of the service."""
    return uvicorn.Config(
        create_app(service),
        host=settings.host,
        port=settings.port,
        ssl_certfile=None if settings.tls_cert_file is None else str(settings.tls_cert_file),
        ssl_keyfile=None if settings.tls_key_file is None else str(settings.tls_key_file),
        loop="asyncio",
        http="h11",
        lifespan="on",
        workers=None,
        reload=False,
        log_config=None,
        access_log=False,
        server_header=False,
        proxy_headers=False,
        timeout_keep_alive=5,
        timeout_graceful_shutdown=settings.shutdown_grace_seconds,
    )


def serve(settings: Settings) -> None:
    """Run the service until it receives SIGTERM or SIGINT."""
    logs.configure(settings.log_level)
    service = build_service(settings)
    logs.event(
        "service_starting",
        service_version=service.versions.service,
        contract_version=service.versions.contract,
        engine_version=service.versions.engine,
        environment=settings.environment,
        tls=settings.tls,
        port=settings.port,
    )
    uvicorn.Server(uvicorn_config(settings, service)).run()
