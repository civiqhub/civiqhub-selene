# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Registry of estimators, read from entry points. Standard library only.

The registry is the extension point of the engine. A package adds a method by
declaring a class in its metadata; no class of the engine is modified::

    [project.entry-points."civiqhub_selene.estimators"]
    "coclustering.single" = "civiqhub_selene.blocks.single:LatentBlockModel"

A key reads ``family.choice`` in lower case. Facades translate a plain word
(``"single"``, ``"mixed"``) into an estimator through the registry::

    spec = default_registry()["coclustering.single"]
    estimator_class = spec.load()

Reading the registry imports nothing: a class is imported when its entry is loaded.
"""

import functools
import importlib
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, TypeAlias, TypeVar, overload

from civiqhub_selene.errors import (
    EstimatorDefinitionError,
    InvalidParameterError,
    UnknownEstimatorError,
)
from civiqhub_selene.estimator import Estimator

__all__ = ["GROUP", "EstimatorSpec", "Registry", "default_registry"]

GROUP = "civiqhub_selene.estimators"
"""Entry-point group under which estimators are declared."""

_KEY = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
_TARGET = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")

EstimatorClass: TypeAlias = type[Estimator[Any, Any]]
_DefaultT = TypeVar("_DefaultT")


@dataclass(frozen=True)
class EstimatorSpec:
    """One entry of the registry.

    Attributes:
        key: ``family.choice``, lower case, for example ``coclustering.single``.
        target: Where the class lives, written ``module:Class``.

    Raises:
        InvalidParameterError: If the key or the target is malformed.
    """

    key: str
    target: str
    _loaded: EstimatorClass | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not _KEY.fullmatch(self.key):
            raise InvalidParameterError(
                f"an estimator key reads family.choice in lower case, such as "
                f"'coclustering.single', got {self.key!r}"
            )
        if not isinstance(self.target, str) or not _TARGET.fullmatch(self.target):
            raise InvalidParameterError(
                f"the target of {self.key!r} reads module:Class, got {self.target!r}"
            )

    @property
    def family(self) -> str:
        """The first half of the key: the family of methods."""
        return self.key.partition(".")[0]

    @property
    def choice(self) -> str:
        """The second half of the key: the method chosen within its family."""
        return self.key.partition(".")[2]

    def load(self) -> EstimatorClass:
        """Import the module and return the estimator class. The class is kept.

        Raises:
            UnknownEstimatorError: If the module or the class does not exist.
            EstimatorDefinitionError: If the target is not a subclass of ``Estimator``.
            MissingExtraError: Raised by the import itself when an optional dependency
                is missing; it goes through unchanged.
        """
        if self._loaded is not None:
            return self._loaded
        module_name, _, attribute = self.target.partition(":")
        try:
            loaded: Any = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            missing = error.name or ""
            if module_name == missing or module_name.startswith(f"{missing}."):
                raise UnknownEstimatorError(
                    f"the estimator {self.key!r} points to {self.target!r}, "
                    f"but the module {missing!r} does not exist"
                ) from error
            raise
        for part in attribute.split("."):
            try:
                loaded = getattr(loaded, part)
            except AttributeError as error:
                raise UnknownEstimatorError(
                    f"the estimator {self.key!r} points to {self.target!r}, "
                    f"but {module_name!r} has no {attribute!r}"
                ) from error
        return self._keep(loaded)

    def _keep(self, loaded: object) -> EstimatorClass:
        if not (isinstance(loaded, type) and issubclass(loaded, Estimator)):
            raise EstimatorDefinitionError(
                f"the estimator {self.key!r} points to {self.target!r}, "
                f"which is not a subclass of Estimator: {loaded!r}"
            )
        object.__setattr__(self, "_loaded", loaded)
        return loaded


class Registry(Mapping[str, EstimatorSpec]):
    """Immutable mapping from ``family.choice`` keys to :class:`EstimatorSpec`.

    Iteration yields the keys in sorted order. A missing key raises
    :class:`~civiqhub_selene.errors.UnknownEstimatorError`, which lists the choices of
    the same family. ``left | right`` gives a new registry where ``right`` wins on a
    shared key.

    Args:
        specs: The entries. When two entries share a key, the last one wins.
    """

    __slots__ = ("_specs",)

    def __init__(self, specs: Iterable[EstimatorSpec] = ()) -> None:
        latest = {spec.key: spec for spec in specs}
        self._specs: dict[str, EstimatorSpec] = dict(sorted(latest.items()))

    @classmethod
    def from_entry_points(cls, group: str = GROUP) -> "Registry":
        """Read the installed entry points of ``group``. No class is imported.

        Raises:
            InvalidParameterError: If an installed entry point has a malformed key.
            EstimatorDefinitionError: If two installed packages declare the same key
                with different targets.
        """
        specs: dict[str, EstimatorSpec] = {}
        for entry_point in metadata.entry_points(group=group):
            spec = EstimatorSpec(entry_point.name, f"{entry_point.module}:{entry_point.attr}")
            known = specs.get(spec.key)
            if known is not None and known.target != spec.target:
                raise EstimatorDefinitionError(
                    f"two installed packages declare the estimator {spec.key!r}: "
                    f"{known.target!r} and {spec.target!r}",
                    hint=(
                        "Uninstall one of the two packages, or build the registry "
                        "explicitly with Registry.from_classes(...)."
                    ),
                )
            specs[spec.key] = spec
        return cls(specs.values())

    @classmethod
    def from_classes(cls, classes: Mapping[str, EstimatorClass]) -> "Registry":
        """Build a registry from classes at hand, for tests and uses without install.

        Raises:
            InvalidParameterError: If a key is malformed.
            EstimatorDefinitionError: If a value is not a subclass of ``Estimator``.
        """
        specs = []
        for key, estimator_class in classes.items():
            module = getattr(estimator_class, "__module__", "unknown")
            name = getattr(estimator_class, "__name__", "Unknown")
            spec = EstimatorSpec(key, f"{module}:{name}")
            spec._keep(estimator_class)
            specs.append(spec)
        return cls(specs)

    def __getitem__(self, key: str) -> EstimatorSpec:
        try:
            return self._specs[key]
        except (KeyError, TypeError):
            family = key.partition(".")[0] if isinstance(key, str) else ""
            choices = sorted(self.family(family))
            if choices:
                offer = f"the family {family!r} offers: {', '.join(choices)}"
            else:
                families = sorted({spec.family for spec in self._specs.values()})
                offer = f"there is no family {family!r} among: {', '.join(families) or 'none'}"
            raise UnknownEstimatorError(
                f"no estimator is registered under {key!r}; {offer}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._specs

    @overload
    def get(self, key: str, /) -> EstimatorSpec | None: ...
    @overload
    def get(self, key: str, /, default: EstimatorSpec | _DefaultT) -> EstimatorSpec | _DefaultT: ...
    def get(
        self, key: str, /, default: EstimatorSpec | _DefaultT | None = None
    ) -> EstimatorSpec | _DefaultT | None:
        """Return the entry under ``key``, or ``default`` when there is none."""
        return self._specs.get(key, default)

    def family(self, name: str) -> dict[str, EstimatorSpec]:
        """Return the entries of one family, indexed by ``choice``."""
        return {spec.choice: spec for spec in self._specs.values() if spec.family == name}

    def __or__(self, other: "Registry") -> "Registry":
        if not isinstance(other, Registry):
            return NotImplemented
        return Registry([*self._specs.values(), *other._specs.values()])

    def __repr__(self) -> str:
        return f"Registry({', '.join(self._specs)})"


@functools.cache
def default_registry() -> Registry:
    """Return the registry of the installed entry points, read once per process."""
    return Registry.from_entry_points()
