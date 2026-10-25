# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Input and output types of SELENE computations. Contract only, no types defined yet.

This module will hold the types that cross the engine boundary. It describes what a
computation receives and returns, not the CiviqHub product model: projects, proposals,
revisions and users belong to the platform and to the ``civiqhub`` SDK, which adapts
them to these types before calling the engine.

Contract that every type added here follows:

- **Standard library only.** Frozen dataclasses, enums and protocols; no dependency on
  the SDK, the CiviqHub backend, a network client or a storage layer.
- **Opaque identifiers.** Entities are referenced by identifiers the engine never
  interprets; no personal data is required to run a computation.
- **Explicit absence.** A missing value is represented as missing, never silently
  replaced by a default or a neutral value.
- **Explicit parameters.** Every tunable of a method is a named, typed field with a
  documented default; nothing is read from the environment or from global state.
- **Provenance.** Every result carries enough to identify how it was produced: input
  fingerprint and schema version, engine version, method and its parameters, and the
  model, weights revision and seed whenever they apply.
- **Stability.** Changes to these types follow the versioning policy in CHANGELOG.md;
  the HTTP schemas of the service adapt to them, not the reverse.
"""
