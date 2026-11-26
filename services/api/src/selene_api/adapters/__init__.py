# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Engine adapters, one per ``(kind, schema_version)`` that the service accepts.

The list is closed and written here: a request never chooses a module, a class or an
import. Adding a payload schema version adds an adapter next to the previous one.
"""

from collections.abc import Mapping

from selene_api.adapters.base import Adapter, InstanceLimits, PreparedTask
from selene_api.adapters.coclustering_single import CoclusteringSingleV1

__all__ = ["ADAPTERS", "Adapter", "InstanceLimits", "PreparedTask", "adapter_for", "kinds"]

ADAPTERS: Mapping[tuple[str, int], Adapter] = {
    (adapter.kind, adapter.schema_version): adapter for adapter in (CoclusteringSingleV1(),)
}


def adapter_for(kind: str, schema_version: int) -> Adapter | None:
    """Return the adapter of a kind and payload schema version, or ``None``."""
    return ADAPTERS.get((kind, schema_version))


def kinds() -> dict[str, list[int]]:
    """Return each accepted kind with its payload schema versions, in increasing order."""
    accepted: dict[str, list[int]] = {}
    for kind, version in sorted(ADAPTERS):
        accepted.setdefault(kind, []).append(version)
    return accepted
