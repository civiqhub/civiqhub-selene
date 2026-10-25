# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""JSON schemas of the service and their adaptation to engine types. Stub, empty for now.

This module will translate request and response bodies to and from the types of
``civiqhub_selene.types``, and nothing more: no computation and no default that the
engine does not define. It mirrors ``openapi.yaml``; both change in the same pull
request, and the contract tests check that they agree.

The HTTP schema is versioned separately from the engine, so that the transport can
evolve without changing a method, and the reverse.
"""
