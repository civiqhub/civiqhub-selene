# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Property-based tests: invariants that hold for every valid input.

Examples: bounds of a statistic, invariance under relabelling of opaque identifiers,
and independence of the ordinal and semantic readings. Shared strategies go here.

Inputs are generated with Hypothesis, a test-only dependency. The profile below has no
deadline, so that a slow CI runner does not turn a correct property into a failure, and
derandomises the search, so that a run is reproducible.
"""

from hypothesis import settings

settings.register_profile("selene", deadline=None, derandomize=True, max_examples=60)
settings.load_profile("selene")
