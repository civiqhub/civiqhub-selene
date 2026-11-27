# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Start the service: one Uvicorn server process, configured by :class:`Settings`.

One process and one event loop serve HTTP; computations run in child processes. There is
no reload and no multiplication of HTTP workers: the jobs live in this process's memory.

- HTTPS when a certificate is configured, TLS 1.2 at least; the certificate is reloaded
  into the live context when its files change.
- Metrics on their own port, from a background thread.
- On SIGTERM or SIGINT the service stops receiving computations at once (``/readyz``
  answers 503), Uvicorn closes the connections within the grace period, then the running
  computation and its process group are stopped and every job is forgotten.
"""

import ssl
from types import FrameType

import uvicorn

from selene_api import logs
from selene_api.app import Service, create_app
from selene_api.config import Settings
from selene_api.metrics import Metrics
from selene_api.runner import ProcessRunner
from selene_api.security import CertificateReloader, TokenStore
from selene_api.versions import BuildInfo, Versions

__all__ = ["SeleneServer", "build_service", "serve", "uvicorn_config"]

MAX_CONNECTIONS = 64
"""Connections served at a time; beyond, Uvicorn answers 503 at once."""


def build_service(settings: Settings, metrics: Metrics | None = None) -> Service:
    """Assemble the service from checked settings."""
    metrics = metrics or Metrics()
    tokens = TokenStore(settings.token_file, token=settings.token)
    runner = ProcessRunner(
        max_result_bytes=settings.max_result_bytes,
        memory_bytes=settings.worker_memory_bytes,
        cpu_seconds=settings.worker_cpu_seconds,
    )
    runner.on_stop = lambda reason: metrics.worker_stops.labels(reason).inc()
    certificates = None
    if settings.tls_cert_file is not None and settings.tls_key_file is not None:
        certificates = CertificateReloader(settings.tls_cert_file, settings.tls_key_file)
    versions = Versions.current(BuildInfo.read(settings.build_info_file))
    return Service.build(
        settings, tokens, runner, versions, metrics=metrics, certificates=certificates
    )


def uvicorn_config(settings: Settings, service: Service) -> uvicorn.Config:
    """Return the Uvicorn configuration of the service, loaded, with its TLS context."""
    config = uvicorn.Config(
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
        limit_concurrency=MAX_CONNECTIONS,
        timeout_graceful_shutdown=settings.shutdown_grace_seconds,
    )
    config.load()
    if config.ssl is not None:
        config.ssl.minimum_version = ssl.TLSVersion.TLSv1_2
        if service.certificates is not None:
            service.certificates.attach(config.ssl)
    return config


class SeleneServer(uvicorn.Server):
    """Uvicorn's server, which tells the service to drain as soon as a signal arrives."""

    def __init__(self, config: uvicorn.Config, service: Service) -> None:
        super().__init__(config)
        self.service = service

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        # A signal handler: set a flag only. Logging here could wait on a lock that the
        # interrupted code holds.
        self.service.draining()
        super().handle_exit(sig, frame)


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
        tls_terminated_by_proxy=settings.tls_terminated_by_proxy,
        port=settings.port,
        metrics_port=settings.metrics_port,
    )
    config = uvicorn_config(settings, service)
    service.metrics.serve(settings.host, settings.metrics_port)
    try:
        SeleneServer(config, service).run()
    finally:
        service.metrics.close()
