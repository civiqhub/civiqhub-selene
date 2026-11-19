# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Errors raised by SELENE. Standard library only.

Every error raised by ``civiqhub_selene`` derives from :class:`SeleneError` and from
the standard exception a reader expects, so ``except ValueError`` catches an
:class:`InvalidInputError`. Each class carries a ``hint``: one sentence that says what
to do next. The hint is attached to the exception as a note, so it is printed under the
traceback, and ``str(error)`` stays the message alone.

A message names the faulty object and the value received; the hint names the remedy.
"""

__all__ = [
    "ContractError",
    "EstimatorDefinitionError",
    "InvalidInputError",
    "InvalidParameterError",
    "MissingExtraError",
    "SeleneError",
    "UnavailableError",
    "UnknownEstimatorError",
]


class SeleneError(Exception):
    """Root of every error raised by SELENE.

    A subclass declares its own non-empty ``hint`` in its class body; a subclass
    without one is refused when the class is defined.

    Args:
        message: What went wrong, naming the faulty object and the value received.
        hint: What to do next. Defaults to the ``hint`` of the class.
    """

    hint: str = "Read the message: it names the faulty object and the value received."

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        hint = cls.__dict__.get("hint")
        if not isinstance(hint, str) or not hint.strip():
            raise TypeError(
                f"{cls.__name__} must declare its own non-empty hint in its class body: "
                "one sentence that says what to do next"
            )

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = type(self).hint if hint is None else hint
        self.add_note(self.hint)


class InvalidInputError(SeleneError, ValueError):
    """The data handed over is not what the method reads, or breaks an invariant."""

    hint = (
        "Build the input with its constructors, for example VoteCounts.from_scores(...) "
        "or VoteMatrix.from_records(...)."
    )


class InvalidParameterError(SeleneError, ValueError):
    """A setting is outside its documented range."""

    hint = "Check the documented range of the parameter named in the message."


class UnknownEstimatorError(SeleneError, LookupError):
    """A registry key does not exist."""

    hint = "List what is installed with Registry.from_entry_points()."


class UnavailableError(SeleneError, LookupError):
    """The value of a result that has none was read."""

    hint = "Test result.is_available first; result.unavailable says why there is no value."


class MissingExtraError(SeleneError, ImportError):
    """An optional dependency is not installed."""

    hint = 'Install the extra, for example: pip install "civiqhub-selene[blocks]".'


class EstimatorDefinitionError(SeleneError, TypeError):
    """An estimator class does not declare its identity."""

    hint = (
        "A concrete estimator declares name (a slug), version (X.Y.Z), reference and "
        "input_type as class attributes."
    )


class ContractError(SeleneError, AssertionError):
    """An estimator breaks the contract. Raised by the contract kit."""

    hint = "Run civiqhub_selene.testing.EstimatorContract on the estimator."
