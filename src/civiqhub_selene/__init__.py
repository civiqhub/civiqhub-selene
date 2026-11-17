# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""SELENE, the open scientific engine of CiviqHub.

The package root holds the contract every method shares, on the standard library
alone: the errors, the result types, the base class of estimators, the registry and
the contract kit. Methods live in subpackages (``ordinal``, ``blocks``) and are added
together with their first tested implementation, never as placeholders.
"""

from civiqhub_selene.errors import (
    ContractError,
    EstimatorDefinitionError,
    InvalidInputError,
    InvalidParameterError,
    MissingExtraError,
    SeleneError,
    UnavailableError,
    UnknownEstimatorError,
)

__all__ = [
    "ContractError",
    "EstimatorDefinitionError",
    "InvalidInputError",
    "InvalidParameterError",
    "MissingExtraError",
    "SeleneError",
    "UnavailableError",
    "UnknownEstimatorError",
    "__version__",
]

__version__ = "0.0.1"
