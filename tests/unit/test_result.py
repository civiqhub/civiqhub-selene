# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
from dataclasses import dataclass

import pytest

from civiqhub_selene.errors import InvalidParameterError, UnavailableError
from civiqhub_selene.types import (
    RESULT_SCHEMA,
    Displayable,
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

METHOD = Method("ordinal_dissensus", "0.1.0", "Research note, 2026")
NO_VOTES = Unavailable(UnavailableReason.NO_VOTES, "no vote")


def make_provenance(seed: int = 0) -> Provenance:
    return Provenance(
        method=METHOD,
        engine_version="0.0.1",
        parameters=(("moderate_position", 0.5),),
        input_fingerprint="0123456789abcdef",
        randomness=Randomness("PCG64", str(seed)),
    )


def test_result_with_a_value() -> None:
    result = Result(outcome=3.0, provenance=make_provenance())
    assert result.value == 3.0
    assert result.is_available is True
    assert result.unavailable is None
    assert result.uncertainty is None
    assert result.method is METHOD


def test_result_without_a_value() -> None:
    result: Result[float] = Result(outcome=NO_VOTES, provenance=make_provenance())
    assert result.is_available is False
    assert result.unavailable is NO_VOTES
    with pytest.raises(UnavailableError, match="no_votes") as caught:
        _ = result.value
    assert isinstance(caught.value, LookupError)
    assert "no vote" in str(caught.value)
    assert "result.is_available" in caught.value.hint


def test_unavailable_prints_its_detail_and_reasons_are_text() -> None:
    assert str(NO_VOTES) == "no vote"
    assert [reason.value for reason in UnavailableReason] == [
        "no_votes",
        "not_enough_data",
        "unsupported_input",
    ]
    assert isinstance(UnavailableReason.NO_VOTES, str)


def test_a_result_is_neither_true_nor_false() -> None:
    result = Result(outcome=3.0, provenance=make_provenance())
    with pytest.raises(TypeError, match=r"result\.is_available"):
        bool(result)
    with pytest.raises(TypeError, match="neither true nor false"):
        _ = not result


def test_results_of_the_same_computation_are_equal() -> None:
    first = Result(outcome=3.0, provenance=make_provenance())
    second = Result(outcome=3.0, provenance=make_provenance())
    other_seed = Result(outcome=3.0, provenance=make_provenance(seed=1))
    assert first == second
    assert hash(first) == hash(second)
    assert first != other_seed
    assert first != 3.0
    assert len({first, second, other_seed}) == 2
    assert first.fingerprint() == make_provenance().fingerprint()


def test_result_is_frozen() -> None:
    result = Result(outcome=3.0, provenance=make_provenance())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.outcome = 4.0  # type: ignore[misc]


def test_to_dict_with_a_plain_value() -> None:
    result = Result(outcome=3.0, provenance=make_provenance())
    data = result.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert data["schema"] == RESULT_SCHEMA == "civiqhub_selene.result/0"
    assert data["available"] is True
    assert data["value"] == 3.0
    assert data["provenance"]["fingerprint"] == result.fingerprint()
    assert "unavailable" not in data
    assert "uncertainty" not in data


def test_to_dict_with_an_absence() -> None:
    result: Result[float] = Result(outcome=NO_VOTES, provenance=make_provenance())
    data = result.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert data["available"] is False
    assert data["unavailable"] == {"reason": "no_votes", "detail": "no vote"}
    assert "value" not in data


def test_to_dict_with_an_interval() -> None:
    result = Result(outcome=0.3, provenance=make_provenance(), uncertainty=Interval(0.2, 0.4, 0.95))
    data = result.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert data["uncertainty"] == {
        "type": "Interval",
        "lower": 0.2,
        "upper": 0.4,
        "confidence": 0.95,
    }
    standard = Result(outcome=0.3, provenance=make_provenance(), uncertainty=StandardError(0.1))
    assert standard.to_dict()["uncertainty"] == {"type": "StandardError", "value": 0.1}


@dataclass(frozen=True, repr=False)
class Reading(Displayable):
    level: int

    def to_dict(self) -> dict[str, int]:
        return {"level": self.level}

    def _describe(self) -> list[str]:
        return [f"Reading({self.level})", "a reading in plain words"]


def test_to_dict_uses_the_to_dict_of_the_value() -> None:
    result = Result(outcome=Reading(2), provenance=make_provenance())
    assert result.to_dict()["value"] == {"level": 2}


def test_format_renders_the_value_then_the_method() -> None:
    result = Result(outcome=0.123456, provenance=make_provenance())
    assert f"{result:.3f}" == "0.123 (ordinal_dissensus 0.1.0)"
    absent: Result[float] = Result(outcome=NO_VOTES, provenance=make_provenance())
    assert f"{absent:.3f}" == "unavailable (no_votes, ordinal_dissensus 0.1.0)"


def test_describe_shows_the_method_then_the_value_or_the_detail() -> None:
    result = Result(outcome=Reading(2), provenance=make_provenance())
    assert repr(result) == "Result(ordinal_dissensus 0.1.0)"
    assert str(result) == ("Result(ordinal_dissensus 0.1.0)\nReading(2)\na reading in plain words")
    absent: Result[float] = Result(outcome=NO_VOTES, provenance=make_provenance())
    assert repr(absent) == "Result(no_votes, ordinal_dissensus 0.1.0)"
    assert str(absent) == "Result(no_votes, ordinal_dissensus 0.1.0)\nno vote"
    assert absent.summary() == str(absent)


def accepts_any_result(result: Result[object]) -> bool:
    return result.is_available


def test_result_is_covariant_in_its_value() -> None:
    result: Result[float] = Result(outcome=3.0, provenance=make_provenance())
    # mypy --strict checks this call: a Result[float] is a Result[object].
    assert accepts_any_result(result)


def test_uncertainties_give_their_half_width() -> None:
    assert StandardError(0.1).half_width == 0.1
    assert Interval(0.2, 0.4, 0.95).half_width == pytest.approx(0.1)
    assert Interval(0.3, 0.3, 0.5).half_width == 0.0
    assert isinstance(StandardError(0.1), Uncertainty)
    with pytest.raises(TypeError, match="abstract"):
        Uncertainty()  # type: ignore[abstract]


@pytest.mark.parametrize(
    ("lower", "upper", "confidence"),
    [
        (0.4, 0.2, 0.95),
        (0.2, 0.4, 0.0),
        (0.2, 0.4, 1.0),
        (0.2, 0.4, 1.5),
        (float("nan"), 0.4, 0.95),
    ],
)
def test_interval_refuses_inverted_bounds_and_confidence_out_of_range(
    lower: float, upper: float, confidence: float
) -> None:
    with pytest.raises(InvalidParameterError, match="Interval needs"):
        Interval(lower, upper, confidence)
