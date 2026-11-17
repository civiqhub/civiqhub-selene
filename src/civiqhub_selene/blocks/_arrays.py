# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Array type aliases shared by the co-clustering modules."""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["BoolArray", "FloatArray", "IntArray"]

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
