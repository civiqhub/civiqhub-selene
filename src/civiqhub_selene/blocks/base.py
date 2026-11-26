# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Base classes of the co-clustering estimators.

:class:`CoclusteringEstimator` says what an estimator of co-clustering answers and
under which plain word it is chosen. :class:`BlockModel` joins it with
``IterativeEstimator`` and carries the settings every block model shares. A concrete
model then only writes three hooks::

    Estimator                      estimate()     written once
    |- IterativeEstimator          _compute()     written once
    '- CoclusteringEstimator       with_shape()
    BlockModel(IterativeEstimator, CoclusteringEstimator)
    |- LatentBlockModel            "single"
    '- MixedMembershipBlockModel   "mixed"
"""

import dataclasses
import math
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, Self

from civiqhub_selene.blocks.data import VoteMatrix
from civiqhub_selene.blocks.iterative import IterativeEstimator, StateT
from civiqhub_selene.blocks.responses import OrdinalProbit, ResponseModel
from civiqhub_selene.blocks.starts import Initialization, RandomStart
from civiqhub_selene.errors import EstimatorDefinitionError, InvalidParameterError
from civiqhub_selene.estimator import Estimator, ValueT_co
from civiqhub_selene.registry import Registry, default_registry
from civiqhub_selene.types import Unavailable, UnavailableReason

__all__ = [
    "FAMILY",
    "BlockModel",
    "CoclusteringEstimator",
    "availability",
    "estimator_for",
    "memberships",
]

FAMILY = "coclustering"
"""Family of the co-clustering estimators in the registry: keys read ``coclustering.word``."""

_RESPONSE_MODEL_METHODS = (
    "initial_parameters",
    "log_probabilities",
    "update",
    "complexity_penalty",
    "to_table",
)
_INITIALIZATION_METHODS = ("arrangement",)


def availability(votes: VoteMatrix, n_groups: int = 1, n_blocks: int = 1) -> Unavailable | None:
    """Say why no co-clustering of this shape exists on ``votes``, or return ``None``.

    A co-clustering needs at least one vote, at least ``max(2, n_groups)`` voters with
    a vote and at least ``max(2, n_blocks)`` items with a vote.

    See the mathematical model: :doc:`/model/data`.
    """
    if votes.n_votes == 0:
        return Unavailable(UnavailableReason.NO_VOTES, "the matrix holds no vote")
    needed_voters, needed_items = max(2, n_groups), max(2, n_blocks)
    active_voters, active_items = votes.n_active_voters, votes.n_active_items
    if active_voters < needed_voters or active_items < needed_items:
        return Unavailable(
            UnavailableReason.NOT_ENOUGH_DATA,
            f"{n_groups} groups and {n_blocks} blocks need at least {needed_voters} voters "
            f"and {needed_items} items with a vote; the matrix has {active_voters} such "
            f"voters and {active_items} such items",
        )
    return None


class CoclusteringEstimator(Estimator[VoteMatrix, ValueT_co]):
    """An estimator that reads a vote matrix and returns groups, blocks and co-clusters.

    Class attributes a concrete subclass declares, besides those of ``Estimator``:

    Attributes:
        membership: The plain word under which the estimator is chosen, for example
            ``"single"``.
        meaning: One sentence in plain words, for example "each voter belongs to one
            group".
        default_criterion: Key of the criterion that chooses its shape by default.

    Instance attributes a concrete subclass carries:

    Attributes:
        n_groups: Number of groups of voters.
        n_blocks: Number of blocks of items.
    """

    input_type: ClassVar[type] = VoteMatrix
    membership: ClassVar[str]
    meaning: ClassVar[str]
    default_criterion: ClassVar[str]

    n_groups: int
    n_blocks: int

    @abstractmethod
    def with_shape(self, n_groups: int, n_blocks: int) -> Self:
        """Return the same estimator at another shape."""

    def _why_unavailable(self, votes: VoteMatrix) -> Unavailable | None:
        return availability(votes, self.n_groups, self.n_blocks)


def _coclustering_classes(
    registry: Registry | None,
) -> dict[str, type[CoclusteringEstimator[Any]]]:
    """Load the estimator class of each installed choice of the family."""
    registry = default_registry() if registry is None else registry
    classes: dict[str, type[CoclusteringEstimator[Any]]] = {}
    for choice, spec in registry.family(FAMILY).items():
        estimator_class = spec.load()
        if not issubclass(estimator_class, CoclusteringEstimator):
            raise EstimatorDefinitionError(
                f"{spec.key!r} points to {estimator_class.__name__}, "
                "which is not a CoclusteringEstimator"
            )
        classes[choice] = estimator_class
    return classes


def memberships(registry: Registry | None = None) -> dict[str, str]:
    """Return the meaning of each installed choice of co-clustering, by its plain word.

    Args:
        registry: Where to look. Defaults to the installed entry points.
    """
    return {
        choice: getattr(estimator_class, "meaning", "")
        for choice, estimator_class in _coclustering_classes(registry).items()
    }


def estimator_for(membership: str, registry: Registry | None = None) -> CoclusteringEstimator[Any]:
    """Return the default estimator behind a plain word, built without argument.

    Args:
        membership: The plain word, for example ``"single"`` or ``"mixed"``.
        registry: Where to look. Defaults to the installed entry points.

    Raises:
        InvalidParameterError: If the word is unknown. The message lists each
            installed choice with its meaning, one per line.
    """
    classes = _coclustering_classes(registry)
    if not isinstance(membership, str) or membership not in classes:
        choices = "\n".join(
            f"  {choice!r}: {getattr(estimator_class, 'meaning', '')}"
            for choice, estimator_class in classes.items()
        )
        raise InvalidParameterError(
            f"membership {membership!r} is not installed. The choices are:\n"
            f"{choices or '  (none is installed)'}"
        )
    return classes[membership]()


@dataclass(frozen=True, kw_only=True)
class BlockModel(
    IterativeEstimator[VoteMatrix, StateT, ValueT_co],
    CoclusteringEstimator[ValueT_co],
    Generic[StateT, ValueT_co],
):
    """Settings and guarantees every block model shares.

    A block model is a latent structure (groups of voters, blocks of items) and a
    response model for its co-clusters. A concrete model declares its identity and
    writes ``_start``, ``_step`` and ``_finish``; its ``_start`` reads the arrangement
    of the start from ``initialization``.

    See the mathematical model: :doc:`/model/estimation`.

    Attributes:
        n_groups: Number of groups of voters, at least 1. Defaults to 2.
        n_blocks: Number of blocks of items, at least 1. Defaults to 2.
        response_model: The response model of the co-clusters. Defaults to
            ``OrdinalProbit()``.
        n_starts: Number of random starts, at least 1; the best one is kept.
            Defaults to 6.
        max_iterations: Largest number of iterations of a start, at least 1.
            Defaults to 200.
        tolerance: Gain of objective under which a start stops, at least 0.
            Defaults to 1e-6.
        initialization: The object that gives the arrangement each start begins from.
            Defaults to ``RandomStart()``; ``SpectralStart()`` starts from regularised
            spectral co-clustering.

    Raises:
        InvalidParameterError: If a setting is outside its range.
    """

    n_groups: int = 2
    n_blocks: int = 2
    response_model: ResponseModel[Any] = field(default_factory=OrdinalProbit)
    n_starts: int = 6
    max_iterations: int = 200
    tolerance: float = 1e-6
    initialization: Initialization = field(default_factory=RandomStart)

    def __post_init__(self) -> None:
        for name in ("n_groups", "n_blocks", "n_starts", "max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise InvalidParameterError(f"{name} is an integer of at least 1, got {value!r}")
        tolerance = self.tolerance
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not 0 <= tolerance < math.inf
        ):
            raise InvalidParameterError(
                f"tolerance is a finite number of at least 0, got {tolerance!r}"
            )
        missing = [
            name
            for name in _RESPONSE_MODEL_METHODS
            if not callable(getattr(self.response_model, name, None))
        ]
        if missing:
            raise InvalidParameterError(
                f"response_model is a response model such as OrdinalProbit() or Categorical(); "
                f"{self.response_model!r} has no {', '.join(missing)}"
            )
        if not all(
            callable(getattr(self.initialization, name, None)) for name in _INITIALIZATION_METHODS
        ):
            raise InvalidParameterError(
                f"initialization is a start object such as RandomStart() or SpectralStart(); "
                f"{self.initialization!r} has no arrangement"
            )

    def with_shape(self, n_groups: int, n_blocks: int) -> Self:
        """Return the same model at another shape."""
        return dataclasses.replace(self, n_groups=n_groups, n_blocks=n_blocks)
