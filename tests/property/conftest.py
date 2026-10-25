# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Property-based tests: invariants that hold for every valid input.

Examples: bounds of a statistic, invariance under relabelling of opaque identifiers,
and independence of the ordinal and semantic readings. A property-testing library is
added to the ``test`` dependency group with the first property test; its profiles and
seeds are configured here so that failures can be replayed.
"""
