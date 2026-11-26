# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Entry point of the computation process: ``python -m selene_api.worker``.

The server starts one such process per job and reads nothing else from it than one JSON
document on its standard output:

- input, on standard input: ``{"kind", "schema_version", "payload"}``, the normalised
  payload of an adapter;
- output: ``{"outcome": "result", "result": {...}}`` or
  ``{"outcome": "error", "code": "engine_error", "exception": "<class name>"}``.

With ``--describe``, it reads nothing and writes the identity of the engine and of each
adapter method; the server runs it once at start-up to check that a computation process
can start, before it declares itself ready.

The numerical libraries are limited to one thread before they are imported. The engine
message of an exception is never written: it may name an identifier of the input.
"""

import os

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

import json  # noqa: E402 - the thread limits come before any numerical import
import platform  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
from typing import Any, BinaryIO  # noqa: E402

__all__ = ["main"]


def _peak_rss_bytes() -> int:
    # ru_maxrss is in kibibytes on Linux.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _describe() -> dict[str, Any]:
    import numpy

    import civiqhub_selene
    from selene_api.adapters import ADAPTERS

    return {
        "engine_version": civiqhub_selene.__version__,
        "numpy_version": numpy.__version__,
        "python_version": platform.python_version(),
        "kinds": [
            {"kind": kind, "schema_version": version, "method": adapter.describe()}
            for (kind, version), adapter in sorted(ADAPTERS.items())
        ],
    }


def _compute(document: dict[str, Any]) -> dict[str, Any]:
    from civiqhub_selene.errors import SeleneError
    from selene_api.adapters import adapter_for

    adapter = adapter_for(document["kind"], document["schema_version"])
    if adapter is None:
        return {"outcome": "error", "code": "engine_error", "exception": "UnsupportedKind"}
    try:
        result = adapter.compute(document["payload"])
    except SeleneError as error:
        return {"outcome": "error", "code": "engine_error", "exception": type(error).__name__}
    return {"outcome": "result", "result": result, "peak_rss_bytes": _peak_rss_bytes()}


def main(argv: list[str] | None = None) -> int:
    """Run one computation, or describe the engine, and write one JSON document."""
    arguments = sys.argv[1:] if argv is None else argv
    output: BinaryIO = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    # Anything a library prints goes to the error stream, never into the protocol.
    sys.stdout = sys.stderr
    if arguments == ["--describe"]:
        answer = _describe()
    elif not arguments:
        answer = _compute(json.loads(sys.stdin.buffer.read()))
    else:
        return 2
    output.write(json.dumps(answer, allow_nan=False).encode("utf-8"))
    output.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
