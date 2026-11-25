# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Metrics of the service, in the Prometheus text format, on their own port.

They are served on ``SELENE_METRICS_PORT``, never on the application port, for the
collector of the cluster only. Every label has a closed set of values: kind, state,
reason, outcome, route template, method and status. No label carries a job, a project,
a voter, a key or a token.

Signals an operator reads without looking at any data:

- the server is alive and ready: ``selene_ready``;
- a computation runs: ``selene_compute_busy`` (a busy service is healthy);
- submissions refused because busy: ``selene_submissions_total{outcome="busy"}``;
- failed computations and why: ``selene_jobs_finished_total{state="failed",reason=...}``;
- stopped processes: ``selene_worker_stops_total{reason=...}``;
- memory of the last computation: ``selene_worker_peak_rss_bytes``, to compare with
  ``selene_worker_memory_limit_bytes``;
- a token file without valid token: ``selene_token_usable == 0``;
- versions: ``selene_build_info``.
"""

import threading
from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    ProcessCollector,
    start_http_server,
)

__all__ = ["DURATION_BUCKETS", "Metrics"]

DURATION_BUCKETS = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0)
"""Buckets of the computation duration, in seconds."""


class Metrics:
    """The metrics of one instance, in a registry of their own."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        registry = self.registry
        ProcessCollector(registry=registry)
        self.build = Info("selene_build", "Versions of this instance.", registry=registry)
        self.ready = Gauge(
            "selene_ready", "1 when the instance receives requests.", registry=registry
        )
        self.slots = Gauge("selene_compute_slots", "Computations run at a time.", registry=registry)
        self.busy = Gauge("selene_compute_busy", "1 while a computation runs.", registry=registry)
        self.retained = Gauge(
            "selene_jobs_retained", "Jobs held in memory, running or finished.", registry=registry
        )
        self.submissions = Counter(
            "selene_submissions",
            "Submissions by outcome: accepted, replayed, conflict, busy, refused, not_ready.",
            ["kind", "outcome"],
            registry=registry,
        )
        self.finished = Counter(
            "selene_jobs_finished",
            "Finished computations by state and reason.",
            ["kind", "state", "reason"],
            registry=registry,
        )
        self.duration = Histogram(
            "selene_job_duration_seconds",
            "Duration of a computation, from its start to its end.",
            ["kind", "state"],
            buckets=DURATION_BUCKETS,
            registry=registry,
        )
        self.worker_peak_rss = Gauge(
            "selene_worker_peak_rss_bytes",
            "Peak resident memory of the last computation process that answered.",
            registry=registry,
        )
        self.worker_memory_limit = Gauge(
            "selene_worker_memory_limit_bytes",
            "Address-space limit of a computation process.",
            registry=registry,
        )
        self.worker_stops = Counter(
            "selene_worker_stops",
            "Computation processes stopped by the server: timeout, result_too_large, cancelled.",
            ["reason"],
            registry=registry,
        )
        self.requests = Counter(
            "selene_http_requests",
            "HTTP requests by route template, method and status.",
            ["route", "method", "status"],
            registry=registry,
        )
        self.token_usable = Gauge(
            "selene_token_usable", "1 when the token file holds a valid token.", registry=registry
        )
        self.tls_reloads = Counter(
            "selene_tls_reloads", "Reloads of the server certificate.", registry=registry
        )
        self.tls_reload_failures = Counter(
            "selene_tls_reload_failures",
            "Certificate files that could not be loaded; the previous certificate stays.",
            registry=registry,
        )
        self._server: Any = None
        self._thread: threading.Thread | None = None

    def serve(self, host: str, port: int) -> None:
        """Serve the metrics on ``host:port`` from a background thread."""
        self._server, self._thread = start_http_server(port, addr=host, registry=self.registry)

    def close(self) -> None:
        """Stop serving the metrics."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
