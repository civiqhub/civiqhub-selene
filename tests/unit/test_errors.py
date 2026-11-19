# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import ast
import pathlib
import sys

import pytest

import civiqhub_selene
from civiqhub_selene import errors
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

SUBCLASSES: list[tuple[type[SeleneError], type[Exception]]] = [
    (InvalidInputError, ValueError),
    (InvalidParameterError, ValueError),
    (UnknownEstimatorError, LookupError),
    (UnavailableError, LookupError),
    (MissingExtraError, ImportError),
    (EstimatorDefinitionError, TypeError),
    (ContractError, AssertionError),
]


def test_module_exports_the_eight_classes() -> None:
    assert len(errors.__all__) == 8
    for name in errors.__all__:
        assert getattr(civiqhub_selene, name) is getattr(errors, name)
        assert name in civiqhub_selene.__all__


@pytest.mark.parametrize(("error_class", "standard_class"), SUBCLASSES)
def test_subclass_derives_from_the_root_and_a_standard_exception(
    error_class: type[SeleneError], standard_class: type[Exception]
) -> None:
    assert issubclass(error_class, SeleneError)
    assert issubclass(error_class, standard_class)
    with pytest.raises(standard_class):
        raise error_class("m")


@pytest.mark.parametrize(("error_class", "standard_class"), SUBCLASSES)
def test_hint_defaults_to_the_class_hint_and_is_a_note(
    error_class: type[SeleneError], standard_class: type[Exception]
) -> None:
    error = error_class("m")
    assert error.hint == error_class.hint
    assert error.hint.strip()
    assert error.hint in error.__notes__

    custom = error_class("m", hint="h")
    assert custom.hint == "h"
    assert custom.__notes__ == ["h"]
    assert error_class.hint != "h"


def test_root_accepts_a_hint_and_attaches_it_as_a_note() -> None:
    error = SeleneError("m", hint="h")
    assert error.hint == "h"
    assert "h" in error.__notes__
    assert SeleneError("m").hint == SeleneError.hint


def test_except_value_error_catches_an_invalid_input() -> None:
    with pytest.raises(ValueError, match=r"^m\b") as caught:
        raise InvalidInputError("m")
    assert caught.type is InvalidInputError


def test_subclass_without_its_own_hint_is_refused() -> None:
    with pytest.raises(TypeError, match="BadError"):

        class BadError(SeleneError):
            pass

    with pytest.raises(TypeError, match="InheritedError"):

        class InheritedError(InvalidInputError):
            pass

    with pytest.raises(TypeError, match="BlankError"):

        class BlankError(SeleneError):
            hint = "   "


def test_subclass_with_its_own_hint_is_accepted() -> None:
    class CustomError(InvalidInputError):
        hint = "Do this instead."

    assert CustomError("m").hint == "Do this instead."
    assert isinstance(CustomError("m"), ValueError)


@pytest.mark.parametrize(("error_class", "standard_class"), SUBCLASSES)
def test_str_is_the_message_alone(
    error_class: type[SeleneError], standard_class: type[Exception]
) -> None:
    assert str(error_class("m")) == "m"
    assert error_class("m").message == "m"


def test_hints_are_plain_sentences() -> None:
    for error_class, _ in SUBCLASSES:
        assert error_class.hint.endswith(".")
        assert "http" not in error_class.hint
    assert "civiqhub-selene[blocks]" in MissingExtraError.hint


def test_module_imports_the_standard_library_only() -> None:
    source = pathlib.Path(errors.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= sys.stdlib_module_names
