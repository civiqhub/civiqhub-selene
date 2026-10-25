# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Joint view of the ordinal and semantic readings. Reserved namespace, empty for now.

This subpackage will combine the outputs of ``ordinal`` and ``semantic`` into an
annotated map, without feeding one reading back into the other: changing responses
leaves semantic positions unchanged, and changing texts leaves ordinal annotations
unchanged.
"""
