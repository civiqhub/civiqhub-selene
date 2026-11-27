# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""HTTP service around the SELENE engine: contract v1 of ``services/api/openapi.yaml``.

The service imports ``civiqhub_selene`` and adds transport, authentication, limits and
volatile job tracking around it. It never reimplements a computation: a change to a
method is made in the engine and reaches the local and service modes together.

Layers, each behind a narrow interface:

- :mod:`selene_api.app` and :mod:`selene_api.schemas`: HTTP routes and transfer objects;
- :mod:`selene_api.jobs`: admission of one computation, states, idempotency, retention;
- :mod:`selene_api.runner` and :mod:`selene_api.worker`: one supervised child process
  per computation;
- :mod:`selene_api.adapters`: one adapter per kind and payload schema version, the only
  place that calls the engine.

Start it with ``python -m selene_api``; :mod:`selene_api.config` lists its settings.
"""
