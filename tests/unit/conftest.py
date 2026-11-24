# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Unit tests: one behaviour per test, on small hand-written inputs.

Every method ships with tests on known cases (unanimity, extreme splits, small samples,
empty input and exact threshold values). Shared fixtures of this suite go here.
"""

import os

# Figures are drawn without a display: the Agg backend renders them in memory.
os.environ.setdefault("MPLBACKEND", "Agg")
