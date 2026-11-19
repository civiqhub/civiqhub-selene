# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Base class of every method of the engine. Standard library only.

An estimator is a fully specified computation: every setting is a field of a frozen
dataclass and nothing is ambient. :class:`Estimator` writes once what every method
guarantees: the input is checked, the provenance is recorded, an absence is explicit,
and the outcome is wrapped in a :class:`~civiqhub_selene.types.Result`. A concrete
method only writes its computation::

    @dataclass(frozen=True, kw_only=True)
    class Mean(Estimator[Numbers, float]):
        name = "mean"
        version = "1.0.0"
        reference = "Arithmetic mean."
        input_type = Numbers

        def _compute(self, data: Numbers, *, seed: int, computation: str) -> float:
            return sum(data.values) / len(data.values)

    result = Mean().estimate(numbers)

The method is named ``estimate`` on purpose: ``fit`` would promise the scikit-learn
usage, where the method returns the estimator itself.
"""

import platform
import re
from abc import ABC, abstractmethod
from importlib import metadata
from typing import ClassVar, Generic, Protocol, TypeVar, final

from civiqhub_selene.errors import (
    EstimatorDefinitionError,
    InvalidInputError,
    InvalidParameterError,
)
from civiqhub_selene.types import (
    Method,
    Provenance,
    Randomness,
    Result,
    Unavailable,
    Uncertainty,
    settings_of,
)

__all__ = ["Estimator", "Fingerprinted"]

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")


class Fingerprinted(Protocol):
    """Anything with a stable fingerprint. The only requirement on an input."""

    def fingerprint(self) -> str:
        """Return a stable fingerprint of the content."""
        ...


DataT_contra = TypeVar("DataT_contra", bound=Fingerprinted, contravariant=True)
ValueT_co = TypeVar("ValueT_co", covariant=True)


def _installed_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


class Estimator(ABC, Generic[DataT_contra, ValueT_co]):
    """A fully specified computation that reads one input and returns a ``Result``.

    Concrete subclasses are frozen dataclasses declared with ``kw_only=True``: every
    setting is a named, typed field, and two estimators with the same fields are the
    same computation. :meth:`estimate` is written once, here, and is never overridden;
    a subclass writes :meth:`_compute`, and :meth:`_why_unavailable` and
    :meth:`_uncertainty` when they apply.

    Determinism: same data, same fields, same seed give the same value and the same
    fingerprint. Any randomness goes through the seed received.

    Class attributes a concrete subclass declares:

    Attributes:
        name: Stable identifier of the method, lower case and underscores.
        version: ``X.Y.Z``. It changes as soon as the numbers returned can change.
        reference: Citation of the source of the method.
        input_type: The class of the input the method reads.
        stochastic: Whether the method uses randomness. Defaults to ``False``.
        generator: Name of the bit generator a stochastic method uses.
        dependencies: Names of the third-party packages the method uses.
    """

    name: ClassVar[str]
    version: ClassVar[str]
    reference: ClassVar[str]
    input_type: ClassVar[type]
    stochastic: ClassVar[bool] = False
    generator: ClassVar[str] = "PCG64"
    dependencies: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        for sealed in ("estimate", "_provenance"):
            if sealed in cls.__dict__:
                raise EstimatorDefinitionError(
                    f"{cls.__name__} redefines {sealed}(), which is written once in Estimator",
                    hint="Write the computation in _compute(); leave estimate() as it is.",
                )

    @classmethod
    def method(cls) -> Method:
        """Return the identity of the method, after checking it is declared.

        Raises:
            EstimatorDefinitionError: If ``name``, ``version`` or ``input_type`` is
                missing or malformed. The message names the class and the value.
        """
        name = getattr(cls, "name", None)
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise EstimatorDefinitionError(
                f"{cls.__name__}.name must be a lower-case identifier such as "
                f"'ordinal_dissensus', got {name!r}"
            )
        version = getattr(cls, "version", None)
        if not isinstance(version, str) or not _VERSION.fullmatch(version):
            raise EstimatorDefinitionError(
                f"{cls.__name__}.version must read X.Y.Z, got {version!r}"
            )
        input_type = getattr(cls, "input_type", None)
        if not isinstance(input_type, type):
            raise EstimatorDefinitionError(
                f"{cls.__name__}.input_type must be the class of the input, got {input_type!r}"
            )
        reference = getattr(cls, "reference", "")
        if not isinstance(reference, str):
            raise EstimatorDefinitionError(
                f"{cls.__name__}.reference must be a citation in text, got {reference!r}"
            )
        return Method(name, version, reference)

    @final
    def estimate(self, data: DataT_contra, *, seed: int = 0) -> Result[ValueT_co]:
        """Run the method on ``data`` and return its result.

        In this order: the identity of the method is checked, the input is checked,
        the provenance is written, the subclass says whether an answer exists, the
        value is computed and wrapped.

        Args:
            data: The input, an instance of ``input_type``.
            seed: Seed of every random draw, a non-negative integer. A method without
                randomness ignores it and leaves it out of the provenance.

        Raises:
            EstimatorDefinitionError: If the class does not declare its identity.
            InvalidInputError: If ``data`` is not an instance of ``input_type``.
            InvalidParameterError: If ``seed`` is not a non-negative integer.
        """
        method = type(self).method()
        input_type = type(self).input_type
        if not isinstance(data, input_type):
            raise InvalidInputError(
                f"{method.name} reads a {input_type.__name__}, got {type(data).__name__}"
            )
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise InvalidParameterError(f"seed must be a non-negative integer, got {seed!r}")
        provenance = self._provenance(method, data, seed)
        unavailable = self._why_unavailable(data)
        if unavailable is not None:
            return Result(unavailable, provenance)
        value = self._compute(data, seed=seed, computation=provenance.fingerprint())
        if isinstance(value, Unavailable):
            return Result(value, provenance)
        return Result(value, provenance, self._uncertainty(data, value))

    @final
    def _provenance(self, method: Method, data: DataT_contra, seed: int) -> Provenance:
        from civiqhub_selene import __version__

        cls = type(self)
        return Provenance(
            method=method,
            engine_version=__version__,
            parameters=settings_of(self),
            input_fingerprint=data.fingerprint(),
            dependencies=(
                ("python", platform.python_version()),
                *((package, _installed_version(package)) for package in cls.dependencies),
            ),
            randomness=Randomness(cls.generator, str(seed)) if cls.stochastic else None,
            lineage=tuple(getattr(data, "lineage", ())),
        )

    def _why_unavailable(self, data: DataT_contra) -> Unavailable | None:
        """Say why no answer exists on ``data``, or return ``None`` when one does.

        Called before :meth:`_compute`, which is skipped when an absence is returned.
        """
        return None

    @abstractmethod
    def _compute(
        self, data: DataT_contra, *, seed: int, computation: str
    ) -> ValueT_co | Unavailable:
        """Compute the value.

        Args:
            data: The input, already checked against ``input_type``.
            seed: Seed of every random draw.
            computation: Fingerprint the result will carry, for values that cite the
                computation they come from.

        Returns:
            The value, or an :class:`~civiqhub_selene.types.Unavailable` when the
            absence only shows during the computation. Never ``None``, never NaN.
        """

    def _uncertainty(
        self,
        data: DataT_contra,
        value: ValueT_co,  # type: ignore[misc]  # hook fed only by estimate()
    ) -> Uncertainty | None:
        """Return the uncertainty of ``value``, or ``None`` when the method defines none.

        Not called when :meth:`_compute` returns an absence.
        """
        return None
