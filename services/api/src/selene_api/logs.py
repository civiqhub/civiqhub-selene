# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Structured logs: one JSON object per line, with a closed set of fields.

A log line carries an event name and technical fields only: method, route template,
status, duration, job identifier, kind, state, error code, versions. Never a header, a
token, a request body, a vote, a pseudonym or an exception message: an exception is
logged by its class name. :func:`event` refuses any other field.
"""

import json
import logging
import sys
import time
from typing import Any

__all__ = ["ALLOWED_FIELDS", "JsonFormatter", "configure", "event"]

LOGGER = logging.getLogger("selene_api")

ALLOWED_FIELDS = frozenset(
    {
        "method",
        "route",
        "status",
        "duration_ms",
        "job_id",
        "kind",
        "schema_version",
        "state",
        "code",
        "replayed",
        "exception",
        "service_version",
        "contract_version",
        "engine_version",
        "environment",
        "tls",
        "port",
        "metrics_port",
        "peak_rss_bytes",
        "reloads",
    }
)


class JsonFormatter(logging.Formatter):
    """Write a record as one JSON line: time, level, event and its allowed fields."""

    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage() if record.name == LOGGER.name else "library",
        }
        fields = getattr(record, "selene", None)
        if isinstance(fields, dict):
            line.update(fields)
        if record.exc_info and record.exc_info[0] is not None:
            line["exception"] = record.exc_info[0].__name__
        return json.dumps(line, separators=(",", ":"), default=str)


def configure(level: str) -> None:
    """Send every log record to the standard output as JSON lines.

    Records of libraries keep their level and logger name, never their message: a
    library message may quote a request.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.WARNING)
    LOGGER.setLevel(level.upper())
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers[:] = []
        logging.getLogger(name).propagate = True


def event(name: str, level: int = logging.INFO, **fields: Any) -> None:
    """Log one event with allowed fields only.

    Raises:
        ValueError: If a field is not allowed; a log line never carries data.
    """
    unknown = set(fields) - ALLOWED_FIELDS
    if unknown:
        raise ValueError(f"field not allowed in a log line: {sorted(unknown)[0]}")
    LOGGER.log(level, name, extra={"selene": fields})
