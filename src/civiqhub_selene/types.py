# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Input and output types of SELENE computations.

This module holds the types that cross the engine boundary. It describes what a
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

import dataclasses
import hashlib
import html
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Generic, TypeAlias, TypeVar, final

from civiqhub_selene.errors import InvalidParameterError, UnavailableError

__all__ = [
    "ADDED_SETTING",
    "RESULT_SCHEMA",
    "Displayable",
    "Interval",
    "Method",
    "Provenance",
    "Randomness",
    "Result",
    "Settings",
    "StandardError",
    "Unavailable",
    "UnavailableReason",
    "Uncertainty",
    "canonical",
    "settings_of",
]

RESULT_SCHEMA = "civiqhub_selene.result/0"
"""Schema identifier written by :meth:`Result.to_dict`."""

T_co = TypeVar("T_co", covariant=True)

Settings: TypeAlias = tuple[tuple[str, Any], ...]
"""Effective settings of a computation: ``(name, value)`` pairs sorted by name.

A setting that is itself a dataclass becomes the tuple
``(("type", ClassName), *its own pairs)``.
"""

_FINGERPRINT_LENGTH = 16


def canonical(value: object) -> str:
    """Serialise ``value`` to canonical JSON. The only serialisation used before hashing.

    Keys are sorted, NaN and infinities are refused (``ValueError``), and an object
    JSON does not know is written as its ``repr``.
    """
    return json.dumps(value, sort_keys=True, allow_nan=False, default=repr)


def _fingerprint(value: object) -> str:
    """First 16 hexadecimal characters of the SHA-256 of the canonical JSON of ``value``."""
    digest = hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()
    return digest[:_FINGERPRINT_LENGTH]


def _is_dataclass_instance(obj: object) -> bool:
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def _setting(value: object) -> object:
    """Normalise one setting value: nested dataclasses become tagged settings."""
    if _is_dataclass_instance(value):
        return (("type", type(value).__name__), *settings_of(value))
    if isinstance(value, (tuple, list)):
        return tuple(_setting(item) for item in value)
    return value


ADDED_SETTING = "civiqhub_selene.added_setting"
"""Metadata key of a setting added to a method after results were published.

A dataclass field whose metadata maps this key to ``True`` is left out of
:func:`settings_of` while it holds its default value. Its default must reproduce the
computation of the method before the setting existed: the settings, and so the
fingerprint of a computation at that default, then stay those published before. A value
other than the default enters the settings as any other.
"""


def _is_left_out(field: dataclasses.Field[Any], value: object) -> bool:
    """Tell an added setting at its default, which :func:`settings_of` leaves out."""
    if not field.metadata.get(ADDED_SETTING, False):
        return False
    default = field.default
    return bool(type(value) is type(default) and value == default)


def settings_of(obj: object) -> Settings:
    """Return the settings of a dataclass instance as ``(name, value)`` pairs sorted by name.

    Only the fields a caller can set (``init=True``) are settings. A field that is
    itself a dataclass is nested as ``(("type", ClassName), *its own pairs)``, and so
    are dataclasses held in a tuple or a list. A setting added after results were
    published (field metadata :data:`ADDED_SETTING`) is left out while it holds its
    default, so that the fingerprints published before it stay valid. Returns ``()``
    when ``obj`` is not a dataclass instance.
    """
    if not _is_dataclass_instance(obj):
        return ()
    pairs = [
        (field.name, _setting(getattr(obj, field.name)))
        for field in dataclasses.fields(obj)  # type: ignore[arg-type]
        if field.init and not _is_left_out(field, getattr(obj, field.name))
    ]
    return tuple(sorted(pairs, key=lambda pair: pair[0]))


def _is_settings(value: object) -> bool:
    """Tell a nested settings tuple, tagged by its leading ``("type", name)`` pair."""
    return (
        isinstance(value, tuple)
        and len(value) > 0
        and all(
            isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], str)
            for pair in value
        )
        and value[0][0] == "type"
    )


def _plain(value: object) -> Any:
    """Turn a setting value into plain JSON-ready data."""
    if _is_settings(value):
        return {name: _plain(item) for name, item in value}  # type: ignore[attr-defined]
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return json.loads(canonical(value))


class Displayable:
    """Stateless mixin that gives an object its three displays from one description.

    A subclass writes :meth:`_describe`: the first line is the identity of the object
    on one line, the following lines summarise it in plain words. The mixin derives
    ``repr`` (the first line), ``str`` and :meth:`summary` (every line), and the HTML
    display of notebooks (the same text, escaped, in a ``pre`` element).

    A dataclass that uses the mixin is declared with ``repr=False``, otherwise the
    generated ``__repr__`` replaces this one.
    """

    __slots__ = ()

    def _describe(self) -> list[str]:
        """Return the identity line, then the summary lines."""
        raise NotImplementedError(f"{type(self).__name__} must write _describe()")

    def __repr__(self) -> str:
        return self._describe()[0]

    def __str__(self) -> str:
        return "\n".join(self._describe())

    def summary(self) -> str:
        """Return the full description, the same text as ``str``."""
        return str(self)

    def _repr_html_(self) -> str:
        return f"<pre>{html.escape(str(self))}</pre>"


@dataclass(frozen=True)
class Method:
    """Identity of a method.

    Attributes:
        name: Stable identifier, lower case and underscores, for example
            ``ordinal_dissensus``.
        version: ``X.Y.Z``. It changes as soon as the numbers returned can change.
        reference: Citation of the source of the method.
    """

    name: str
    version: str
    reference: str = ""

    def __str__(self) -> str:
        return f"{self.name} {self.version}"


@dataclass(frozen=True)
class Randomness:
    """Pseudo-random generator used by a computation, and its seed.

    Attributes:
        generator: Name of the bit generator, for example ``"PCG64"``.
        seed: The seed, kept as text so that an integer of any size survives JSON.
    """

    generator: str
    seed: str


@dataclass(frozen=True)
class Provenance:
    """What it takes to redo a result and to cite it.

    Attributes:
        method: The method.
        engine_version: ``civiqhub_selene.__version__`` when the computation ran.
        parameters: The effective settings, as :data:`Settings`.
        input_fingerprint: Fingerprint of the input.
        dependencies: ``("python", version)`` then one pair per third-party package used.
        randomness: ``None`` for a method without randomness.
        lineage: Fingerprints of the computations the input comes from.
    """

    method: Method
    engine_version: str
    parameters: Settings
    input_fingerprint: str
    dependencies: tuple[tuple[str, str], ...] = ()
    randomness: Randomness | None = None
    lineage: tuple[str, ...] = ()

    @property
    def seed(self) -> int | None:
        """The seed as an integer, or ``None`` for a method without randomness."""
        return None if self.randomness is None else int(self.randomness.seed)

    def fingerprint(self) -> str:
        """Return the identity of the computation: 16 hexadecimal characters.

        SHA-256 of the canonical JSON of the method name and version, the settings,
        the randomness, the input fingerprint and the lineage. ``engine_version`` and
        ``dependencies`` are left out: the same request on a newer NumPy is the same
        request.
        """
        randomness = (
            None if self.randomness is None else (self.randomness.generator, self.randomness.seed)
        )
        return _fingerprint(
            (
                self.method.name,
                self.method.version,
                self.parameters,
                randomness,
                self.input_fingerprint,
                self.lineage,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return plain data, ready for JSON."""
        return {
            "method": {
                "name": self.method.name,
                "version": self.method.version,
                "reference": self.method.reference,
            },
            "engine_version": self.engine_version,
            "parameters": {name: _plain(value) for name, value in self.parameters},
            "input_fingerprint": self.input_fingerprint,
            "dependencies": dict(self.dependencies),
            "randomness": (
                None
                if self.randomness is None
                else {"generator": self.randomness.generator, "seed": self.randomness.seed}
            ),
            "lineage": list(self.lineage),
            "fingerprint": self.fingerprint(),
        }


class UnavailableReason(StrEnum):
    """Why a method has no value to return."""

    NO_VOTES = "no_votes"
    NOT_ENOUGH_DATA = "not_enough_data"
    UNSUPPORTED_INPUT = "unsupported_input"


@dataclass(frozen=True)
class Unavailable:
    """An explained absence of value.

    Attributes:
        reason: The kind of absence.
        detail: One English sentence that names the numbers at stake.
    """

    reason: UnavailableReason
    detail: str

    def __str__(self) -> str:
        return self.detail


class Uncertainty(ABC):
    """Uncertainty of a value. ``None`` in a result means "not defined", never "zero"."""

    @property
    @abstractmethod
    def half_width(self) -> float:
        """Half the width of the uncertainty around the value."""


@dataclass(frozen=True)
class StandardError(Uncertainty):
    """Standard error of an estimate.

    Attributes:
        value: The standard error, in the unit of the estimate.
    """

    value: float

    @property
    def half_width(self) -> float:
        return self.value


@dataclass(frozen=True)
class Interval(Uncertainty):
    """Interval around an estimate.

    Attributes:
        lower: Lower bound.
        upper: Upper bound, at least ``lower``.
        confidence: Confidence level, strictly between 0 and 1.

    Raises:
        InvalidParameterError: If ``lower > upper`` or ``confidence`` is not in ``(0, 1)``.
    """

    lower: float
    upper: float
    confidence: float

    def __post_init__(self) -> None:
        if not self.lower <= self.upper:
            raise InvalidParameterError(
                f"Interval needs lower <= upper, got lower={self.lower!r} and upper={self.upper!r}"
            )
        if not 0 < self.confidence < 1:
            raise InvalidParameterError(
                "Interval needs a confidence strictly between 0 and 1, "
                f"got confidence={self.confidence!r}"
            )

    @property
    def half_width(self) -> float:
        return (self.upper - self.lower) / 2


@final
@dataclass(frozen=True, eq=False, repr=False)
class Result(Displayable, Generic[T_co]):
    """The only type a method of the engine returns.

    It carries a value or an explained absence, the provenance, and the uncertainty
    when the method defines one. It decides nothing: it describes.

    Two results of the same computation are equal: equality and hash follow
    :meth:`fingerprint`. A result is neither true nor false; test ``is_available``.

    Attributes:
        outcome: The value, or the :class:`Unavailable` that explains its absence.
        provenance: How the result was produced.
        uncertainty: The uncertainty of the value, or ``None`` when the method does
            not define one. ``None`` is not a zero uncertainty.
    """

    outcome: T_co | Unavailable
    provenance: Provenance
    uncertainty: Uncertainty | None = None

    @property
    def is_available(self) -> bool:
        """Whether the result holds a value."""
        return not isinstance(self.outcome, Unavailable)

    @property
    def value(self) -> T_co:
        """The value.

        Raises:
            UnavailableError: If the result has no value. The message gives the reason
                and the detail of the absence.
        """
        if isinstance(self.outcome, Unavailable):
            raise UnavailableError(
                f"{self.method} has no value ({self.outcome.reason.value}): {self.outcome.detail}"
            )
        return self.outcome

    @property
    def unavailable(self) -> Unavailable | None:
        """The explained absence, or ``None`` when there is a value."""
        return self.outcome if isinstance(self.outcome, Unavailable) else None

    @property
    def method(self) -> Method:
        """The method that produced the result."""
        return self.provenance.method

    def fingerprint(self) -> str:
        """Return the identity of the computation, as :meth:`Provenance.fingerprint`."""
        return self.provenance.fingerprint()

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a dictionary under the schema :data:`RESULT_SCHEMA`.

        The value is written through its own ``to_dict()`` when it has one, as it is
        otherwise.
        """
        data: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "available": self.is_available,
            "provenance": self.provenance.to_dict(),
        }
        if isinstance(self.outcome, Unavailable):
            data["unavailable"] = {
                "reason": self.outcome.reason.value,
                "detail": self.outcome.detail,
            }
        else:
            to_dict = getattr(self.outcome, "to_dict", None)
            data["value"] = to_dict() if callable(to_dict) else self.outcome
        if self.uncertainty is not None:
            data["uncertainty"] = {
                "type": type(self.uncertainty).__name__,
                **dict(settings_of(self.uncertainty)),
            }
        return data

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Result):
            return NotImplemented
        return self.fingerprint() == other.fingerprint()

    def __hash__(self) -> int:
        return hash(self.fingerprint())

    def __bool__(self) -> bool:
        raise TypeError("a Result is neither true nor false; test result.is_available")

    def __format__(self, format_spec: str) -> str:
        if isinstance(self.outcome, Unavailable):
            return f"unavailable ({self.outcome.reason.value}, {self.method})"
        return f"{format(self.outcome, format_spec)} ({self.method})"

    def _describe(self) -> list[str]:
        if isinstance(self.outcome, Unavailable):
            return [f"Result({self.outcome.reason.value}, {self.method})", self.outcome.detail]
        return [f"Result({self.method})", *str(self.outcome).splitlines()]
