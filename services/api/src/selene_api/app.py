# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""HTTP application of the service. Stub, no routes yet.

This module will define the HTTP/JSON application described by ``openapi.yaml``:

- **Authentication.** Every call is authenticated as a service caller. The private
  network is never treated as authentication. The service validates the scope it
  receives but does not resolve the permissions of the calling product.
- **Validation and limits.** Request size, concurrency and time limits are enforced
  before any computation; errors follow the documented error schema.
- **Health.** Liveness and readiness endpoints for the orchestrator.
- **Observability.** Volumes, latencies, errors, saturation and versions, never texts,
  responses, identities or secrets; metrics are not exposed on the application route.

Short computations answer synchronously; long ones go through ``jobs``.
"""
