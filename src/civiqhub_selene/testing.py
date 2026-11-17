# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Contract kit: prove in ten lines that an estimator honours the contract.

Write a pytest class with three methods; it inherits seven tests::

    from civiqhub_selene.testing import EstimatorContract

    class TestMean(EstimatorContract):
        def make_estimator(self):
            return Mean()

        def make_data(self):
            return Numbers((1.0, 2.0))  # an input on which a value exists

        def make_empty_data(self):
            return Numbers(())  # an input with no answer, or None if there is none

A test of the kit that fails raises :class:`~civiqhub_selene.errors.ContractError`,
whose message names the estimator and the rule it breaks. The kit runs under pytest
but the package does not depend on it: pytest is imported only when a test needs it.
"""

import dataclasses
import hashlib
import math
import re
from typing import Any, NoReturn

from civiqhub_selene.errors import (
    ContractError,
    EstimatorDefinitionError,
    InvalidInputError,
    UnavailableError,
)
from civiqhub_selene.estimator import Estimator
from civiqhub_selene.types import Provenance, Result, canonical, settings_of

__all__ = ["EstimatorContract", "fingerprint_of"]

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")


def fingerprint_of(value: object) -> str:
    """Return a stable fingerprint of a value: 16 hexadecimal characters.

    ``value.fingerprint()`` when the object has one, otherwise the SHA-256 of its
    canonical JSON.

    Raises:
        ValueError: If the value holds NaN or an infinity, which have no canonical form.
    """
    fingerprint = getattr(value, "fingerprint", None)
    if callable(fingerprint):
        return str(fingerprint())
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()[:16]


class EstimatorContract:
    """Base class of a pytest class that checks one estimator against the contract.

    Write :meth:`make_estimator`, :meth:`make_data` and :meth:`make_empty_data`; the
    seven ``test_*`` methods are inherited. Name the subclass ``Test...`` so that
    pytest collects it.
    """

    def make_estimator(self) -> Estimator[Any, Any]:
        """Return an estimator ready to use."""
        raise NotImplementedError(f"{type(self).__name__} must write make_estimator()")

    def make_data(self) -> Any:
        """Return an input on which a value exists."""
        raise NotImplementedError(f"{type(self).__name__} must write make_data()")

    def make_empty_data(self) -> Any:
        """Return an input with no possible answer, or ``None`` if the method always answers."""
        raise NotImplementedError(f"{type(self).__name__} must write make_empty_data()")

    def test_identity_is_declared(self) -> None:
        """The class declares its name, its version and a reference."""
        estimator = self.make_estimator()
        try:
            method = type(estimator).method()
        except EstimatorDefinitionError as error:
            _fail(estimator, "identity is declared", str(error))
        if not _NAME.fullmatch(method.name):
            _fail(estimator, "identity is declared", f"name {method.name!r} is not a slug")
        if not _VERSION.fullmatch(method.version):
            _fail(estimator, "identity is declared", f"version {method.version!r} is not X.Y.Z")
        if not method.reference.strip():
            _fail(estimator, "identity is declared", "reference is empty")

    def test_estimator_is_a_frozen_value(self) -> None:
        """The estimator is a frozen dataclass, equal to a copy of itself, and hashable."""
        estimator = self.make_estimator()
        rule = "estimator is a frozen value"
        parameters = getattr(type(estimator), "__dataclass_params__", None)
        if not dataclasses.is_dataclass(estimator) or parameters is None:
            _fail(estimator, rule, "the class is not a dataclass")
        if not parameters.frozen:
            _fail(estimator, rule, "the dataclass is not frozen")
        fields = {
            field.name: getattr(estimator, field.name)
            for field in dataclasses.fields(estimator)
            if field.init
        }
        copy = type(estimator)(**fields)
        if copy != estimator:
            _fail(estimator, rule, "a copy built from the same fields is not equal")
        try:
            same_hash = hash(copy) == hash(estimator)
        except TypeError as error:
            _fail(estimator, rule, f"the estimator is not hashable: {error}")
        if not same_hash:
            _fail(estimator, rule, "a copy built from the same fields has another hash")

    def test_returns_a_result(self) -> None:
        """``estimate`` returns an available ``Result`` on the data of ``make_data``."""
        estimator = self.make_estimator()
        _available(estimator, self.make_data(), "returns a result")

    def test_deterministic(self) -> None:
        """Two calls with the same seed give the same fingerprints."""
        estimator = self.make_estimator()
        rule = "deterministic"
        data = self.make_data()
        first = _available(estimator, data, rule)
        second = _available(estimator, data, rule)
        if first.fingerprint() != second.fingerprint():
            _fail(estimator, rule, "two calls with seed=0 give two result fingerprints")
        if _value_fingerprint(estimator, rule, first) != _value_fingerprint(
            estimator, rule, second
        ):
            _fail(estimator, rule, "two calls with seed=0 give two different values")

    def test_provenance_is_complete(self) -> None:
        """The provenance names the engine, the input, the settings and the packages."""
        estimator = self.make_estimator()
        rule = "provenance is complete"
        result = _available(estimator, self.make_data(), rule)
        _check_provenance(estimator, rule, result.provenance)

    def test_absence_is_explicit(self) -> None:
        """An input with no answer gives an ``Unavailable`` with a detail, never ``None``."""
        estimator = self.make_estimator()
        rule = "absence is explicit"
        empty = self.make_empty_data()
        if empty is None:
            import pytest

            pytest.skip(f"{type(estimator).__name__} always answers: make_empty_data() is None")
        result = estimator.estimate(empty)
        if not isinstance(result, Result):
            _fail(estimator, rule, f"estimate returned a {type(result).__name__}")
        unavailable = result.unavailable
        if unavailable is None:
            _fail(
                estimator,
                rule,
                f"an input with no answer gave the value {result.outcome!r} "
                "instead of an Unavailable",
            )
        if not unavailable.detail.strip():
            _fail(estimator, rule, "the Unavailable has an empty detail")
        _check_provenance(estimator, rule, result.provenance)
        try:
            _ = result.value
        except UnavailableError:
            return
        _fail(estimator, rule, "reading the value of an absent result did not raise")

    def test_rejects_foreign_input(self) -> None:
        """An input of another type raises ``InvalidInputError``."""
        estimator = self.make_estimator()
        rule = "rejects foreign input"
        try:
            estimator.estimate(object())
        except InvalidInputError:
            return
        except Exception as error:
            _fail(
                estimator,
                rule,
                f"estimate(object()) raised {type(error).__name__} instead of InvalidInputError",
            )
        _fail(estimator, rule, "estimate(object()) returned instead of raising")


def _fail(estimator: object, rule: str, detail: str) -> NoReturn:
    raise ContractError(f"{type(estimator).__name__} breaks the rule '{rule}': {detail}")


def _available(estimator: Estimator[Any, Any], data: object, rule: str) -> Result[Any]:
    """Estimate with seed 0 and return the result, which must hold a real value."""
    result = estimator.estimate(data, seed=0)
    if not isinstance(result, Result):
        _fail(estimator, rule, f"estimate returned a {type(result).__name__}, not a Result")
    if not result.is_available:
        _fail(estimator, rule, f"make_data() gives an input with no value: {result.unavailable}")
    value = result.outcome
    if value is None:
        _fail(estimator, rule, "the value is None; an absence is an Unavailable")
    if isinstance(value, float) and math.isnan(value):
        _fail(estimator, rule, "the value is NaN; an absence is an Unavailable")
    return result


def _value_fingerprint(estimator: object, rule: str, result: Result[Any]) -> str:
    try:
        return fingerprint_of(result.value)
    except ValueError as error:
        _fail(estimator, rule, f"the value has no canonical form: {error}")


def _check_provenance(estimator: Estimator[Any, Any], rule: str, provenance: Provenance) -> None:
    cls = type(estimator)
    if not provenance.engine_version:
        _fail(estimator, rule, "engine_version is empty")
    if not provenance.input_fingerprint:
        _fail(estimator, rule, "input_fingerprint is empty")
    if provenance.parameters != settings_of(estimator):
        _fail(estimator, rule, "parameters are not the settings of the estimator")
    if cls.stochastic and provenance.randomness is None:
        _fail(estimator, rule, "a stochastic method records no randomness")
    if not cls.stochastic and provenance.randomness is not None:
        _fail(estimator, rule, "a method without randomness records a seed")
    recorded = dict(provenance.dependencies)
    for package in ("python", *cls.dependencies):
        if package not in recorded:
            _fail(estimator, rule, f"the dependency {package!r} is not recorded")
