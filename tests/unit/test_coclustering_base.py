# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene.blocks import VoteMatrix
from civiqhub_selene.blocks._numerics import History
from civiqhub_selene.blocks.base import (
    FAMILY,
    BlockModel,
    CoclusteringEstimator,
    availability,
    estimator_for,
    memberships,
)
from civiqhub_selene.blocks.responses import Categorical, OrdinalProbit
from civiqhub_selene.errors import EstimatorDefinitionError, InvalidParameterError
from civiqhub_selene.estimator import Estimator
from civiqhub_selene.registry import Registry
from civiqhub_selene.testing import EstimatorContract
from civiqhub_selene.types import UnavailableReason, settings_of

pytestmark = pytest.mark.blocks

VOTES = VoteMatrix.from_table([[2, 2, -2], [2, 1, -1], [-2, -2, 2], [-1, None, 2]])


@dataclass(frozen=True)
class Split:
    """The state of the test model: where the voters are cut in two, and its objective."""

    cut: int
    objective: float


@dataclass(frozen=True)
class Halves:
    """The value of the test model."""

    cut: int
    shape: tuple[int, int]
    history: History

    def fingerprint(self) -> str:
        return f"halves:{self.cut}:{self.shape}"


@dataclass(frozen=True, kw_only=True)
class HalvesModel(BlockModel[Split, Halves]):
    """A block model written in the tests, without touching a class of the package."""

    name = "halves_model"
    version = "0.1.0"
    reference = "A test model."
    membership = "halves"
    meaning = "the voters are cut in two halves"
    default_criterion = "integrated_likelihood"

    def _start(self, votes: VoteMatrix, rng: np.random.Generator) -> Split:
        return Split(int(rng.integers(0, votes.shape[0] + 1)), -float(votes.shape[0]))

    def _step(self, votes: VoteMatrix, state: Split) -> Split:
        middle = votes.shape[0] // 2
        cut = state.cut + int(np.sign(middle - state.cut))
        return Split(cut, -float(abs(cut - middle)))

    def _finish(
        self, votes: VoteMatrix, state: Split, history: History, computation: str
    ) -> Halves:
        return Halves(state.cut, (self.n_groups, self.n_blocks), history)


@dataclass(frozen=True, kw_only=True)
class ThirdsModel(HalvesModel):
    name = "thirds_model"
    membership = "thirds"
    meaning = "the voters are cut in three thirds"


REGISTRY = Registry.from_classes(
    {"coclustering.halves": HalvesModel, "coclustering.thirds": ThirdsModel}
)


def test_family() -> None:
    assert FAMILY == "coclustering"


def test_default_settings_of_a_block_model() -> None:
    model = HalvesModel()
    assert (model.n_groups, model.n_blocks) == (2, 2)
    assert model.response_model == OrdinalProbit()
    assert (model.n_starts, model.max_iterations, model.tolerance) == (6, 200, 1e-6)
    assert HalvesModel.input_type is VoteMatrix
    assert HalvesModel.stochastic is True
    assert HalvesModel.dependencies == ("numpy",)
    assert isinstance(model, CoclusteringEstimator)
    assert isinstance(model, Estimator)


def test_a_block_model_runs_through_the_shared_pipeline() -> None:
    result = HalvesModel(n_starts=3).estimate(VOTES, seed=1)
    assert result.is_available
    assert result.value.cut == 2
    assert result.value.shape == (2, 2)
    assert result.value.history.converged is True
    assert result.method.name == "halves_model"
    assert result.provenance.input_fingerprint == VOTES.fingerprint()
    assert result.provenance.randomness is not None
    assert "numpy" in dict(result.provenance.dependencies)
    assert result == HalvesModel(n_starts=3).estimate(VOTES, seed=1)


def test_the_response_model_enters_the_provenance_with_its_settings() -> None:
    settings = dict(settings_of(HalvesModel(response_model=Categorical(smoothing=1.0))))
    assert settings["response_model"] == (
        ("type", "Categorical"),
        ("min_probability", 1e-6),
        ("smoothing", 1.0),
    )
    probit = HalvesModel().estimate(VOTES)
    categorical = HalvesModel(response_model=Categorical()).estimate(VOTES)
    assert probit.fingerprint() != categorical.fingerprint()
    assert probit.provenance.to_dict()["parameters"]["response_model"]["type"] == "OrdinalProbit"


def test_with_shape_gives_the_same_model_at_another_shape() -> None:
    model = HalvesModel(n_starts=3, response_model=Categorical())
    other = model.with_shape(4, 3)
    assert (other.n_groups, other.n_blocks) == (4, 3)
    assert other == dataclasses.replace(model, n_groups=4, n_blocks=3)
    assert other.with_shape(2, 2) == model
    assert type(other) is HalvesModel
    assert (model.n_groups, model.n_blocks) == (2, 2)
    # A shape larger than the data is not an error of settings: it gives no result.
    large = model.with_shape(50, 2).estimate(VOTES)
    assert large.unavailable is not None
    assert large.unavailable.reason is UnavailableReason.NOT_ENOUGH_DATA


def test_a_block_model_needs_its_three_hooks() -> None:
    @dataclass(frozen=True, kw_only=True)
    class Hookless(BlockModel[Split, Halves]):
        name = "hookless"
        version = "0.1.0"
        reference = "r"
        membership = "hookless"
        meaning = "nothing"
        default_criterion = "integrated_likelihood"

    with pytest.raises(TypeError, match="abstract"):
        Hookless()  # type: ignore[abstract]
    with pytest.raises(TypeError, match="abstract"):
        BlockModel()  # type: ignore[abstract]


@pytest.mark.parametrize(
    ("settings", "named"),
    [
        ({"n_groups": 0}, "n_groups"),
        ({"n_blocks": 0}, "n_blocks"),
        ({"n_starts": 0}, "n_starts"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"n_groups": 2.0}, "n_groups"),
        ({"n_groups": True}, "n_groups"),
        ({"tolerance": -1e-9}, "tolerance"),
        ({"tolerance": float("inf")}, "tolerance"),
        ({"tolerance": float("nan")}, "tolerance"),
        ({"tolerance": "small"}, "tolerance"),
        ({"response_model": "probit"}, "response_model"),
        ({"response_model": None}, "response_model"),
    ],
)
def test_settings_out_of_range_are_refused(settings: dict[str, Any], named: str) -> None:
    with pytest.raises(InvalidParameterError, match=named):
        HalvesModel(**settings)


def test_a_zero_tolerance_is_allowed() -> None:
    assert HalvesModel(tolerance=0).estimate(VOTES).is_available


# Availability -------------------------------------------------------------------------


def test_availability() -> None:
    assert availability(VOTES) is None
    assert availability(VOTES, 4, 3) is None

    empty = availability(VoteMatrix.from_records([]), 2, 2)
    assert empty is not None
    assert empty.reason is UnavailableReason.NO_VOTES
    assert empty.detail == "the matrix holds no vote"
    abstentions = availability(VoteMatrix.from_table([[None, None], [None, None]]))
    assert abstentions is not None
    assert abstentions.reason is UnavailableReason.NO_VOTES

    two_voters = VoteMatrix.from_table([[2, 1, 0], [-2, -1, 0], [None, None, None]])
    too_many = availability(two_voters, 3, 2)
    assert too_many is not None
    assert too_many.reason is UnavailableReason.NOT_ENOUGH_DATA
    for number in ("3 groups", "2 blocks", "at least 3 voters", "2 such voters", "3 such items"):
        assert number in too_many.detail
    assert availability(two_voters, 2, 3) is None
    assert availability(two_voters, 2, 4) is not None


def test_a_single_cocluster_still_needs_two_voters_and_two_items() -> None:
    one_voter = VoteMatrix.from_table([[2, 1, 0]])
    one_item = VoteMatrix.from_table([[2], [1], [0]])
    for votes in (one_voter, one_item):
        unavailable = availability(votes, 1, 1)
        assert unavailable is not None
        assert unavailable.reason is UnavailableReason.NOT_ENOUGH_DATA
    assert availability(VoteMatrix.from_table([[2, 1], [0, None]]), 1, 1) is None


def test_an_estimator_says_why_it_has_no_answer_without_computing() -> None:
    result = HalvesModel().estimate(VoteMatrix.from_records([]))
    assert result.is_available is False
    assert result.unavailable is not None
    assert result.unavailable.reason is UnavailableReason.NO_VOTES
    assert result.provenance.method.name == "halves_model"
    too_large = HalvesModel(n_groups=5).estimate(VOTES)
    assert too_large.unavailable is not None
    assert too_large.unavailable.reason is UnavailableReason.NOT_ENOUGH_DATA


# Plain words --------------------------------------------------------------------------


def test_memberships_give_the_meaning_of_each_installed_choice() -> None:
    assert memberships(REGISTRY) == {
        "halves": "the voters are cut in two halves",
        "thirds": "the voters are cut in three thirds",
    }
    assert memberships(Registry()) == {}


def test_estimator_for_builds_the_default_estimator_behind_a_word() -> None:
    estimator = estimator_for("halves", REGISTRY)
    assert estimator == HalvesModel()
    assert isinstance(estimator_for("thirds", REGISTRY), ThirdsModel)


def test_an_unknown_word_lists_the_choices_and_their_meaning() -> None:
    with pytest.raises(InvalidParameterError, match="membership 'nope' is not installed") as caught:
        estimator_for("nope", REGISTRY)
    message = str(caught.value)
    assert "'halves': the voters are cut in two halves" in message
    assert "'thirds': the voters are cut in three thirds" in message
    assert message.count("\n") == 2
    with pytest.raises(InvalidParameterError, match="none is installed"):
        estimator_for("single", Registry())
    with pytest.raises(InvalidParameterError, match="membership 3"):
        estimator_for(3, REGISTRY)  # type: ignore[arg-type]


def test_another_family_is_ignored_and_a_foreign_class_is_refused() -> None:
    @dataclass(frozen=True)
    class Numbers:
        def fingerprint(self) -> str:
            return "numbers"

    @dataclass(frozen=True, kw_only=True)
    class Plain(Estimator[Numbers, float]):
        name = "plain"
        version = "0.1.0"
        reference = "r"
        input_type = Numbers

        def _compute(self, data: Numbers, *, seed: int, computation: str) -> float:
            return 0.0

    other_family = Registry.from_classes({"dissensus.ordinal": Plain}) | REGISTRY
    assert set(memberships(other_family)) == {"halves", "thirds"}
    foreign = Registry.from_classes({"coclustering.plain": Plain})
    with pytest.raises(EstimatorDefinitionError, match="not a CoclusteringEstimator"):
        memberships(foreign)


class TestHalvesModel(EstimatorContract):
    def make_estimator(self) -> HalvesModel:
        return HalvesModel(n_starts=2)

    def make_data(self) -> VoteMatrix:
        return VOTES

    def make_empty_data(self) -> VoteMatrix:
        return VoteMatrix.from_records([])
