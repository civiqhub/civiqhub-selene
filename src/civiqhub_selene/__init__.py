# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""SELENE, the open scientific engine of CiviqHub.

The package root holds the contract every method shares, on the standard library
alone: the errors, the result types, the base class of estimators and the registry.
The contract kit lives in ``civiqhub_selene.testing``. Methods live in subpackages
(``ordinal``, ``blocks``) and are added together with their first tested
implementation, never as placeholders.

Importing ``civiqhub_selene`` loads neither NumPy nor the ``civiqhub`` SDK.
"""

#: Version of the engine, in PEP 440 notation. Every result records it in its provenance.
__version__ = "0.3.0"

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
from civiqhub_selene.estimator import Estimator
from civiqhub_selene.registry import EstimatorSpec, Registry, default_registry
from civiqhub_selene.types import (
    Interval,
    Method,
    Provenance,
    Randomness,
    Result,
    StandardError,
    Unavailable,
    UnavailableReason,
    Uncertainty,
)

__all__ = [
    "ContractError",
    "Estimator",
    "EstimatorDefinitionError",
    "EstimatorSpec",
    "Interval",
    "InvalidInputError",
    "InvalidParameterError",
    "Method",
    "MissingExtraError",
    "Provenance",
    "Randomness",
    "Registry",
    "Result",
    "SeleneError",
    "StandardError",
    "Unavailable",
    "UnavailableError",
    "UnavailableReason",
    "Uncertainty",
    "UnknownEstimatorError",
    "__version__",
    "default_registry",
]
