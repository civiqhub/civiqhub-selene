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
from dataclasses import dataclass
from typing import Any, TypeAlias

__all__ = [
    "Displayable",
    "Method",
    "Provenance",
    "Randomness",
    "Settings",
    "canonical",
    "settings_of",
]

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


def settings_of(obj: object) -> Settings:
    """Return the settings of a dataclass instance as ``(name, value)`` pairs sorted by name.

    Only the fields a caller can set (``init=True``) are settings. A field that is
    itself a dataclass is nested as ``(("type", ClassName), *its own pairs)``, and so
    are dataclasses held in a tuple or a list. Returns ``()`` when ``obj`` is not a
    dataclass instance.
    """
    if not _is_dataclass_instance(obj):
        return ()
    pairs = [
        (field.name, _setting(getattr(obj, field.name)))
        for field in dataclasses.fields(obj)  # type: ignore[arg-type]
        if field.init
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
