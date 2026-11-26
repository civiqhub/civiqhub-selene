# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""The pipeline of a block model, end to end, on a model written in the tests.

This is a smoke test of the architecture, not a delivered method: a classification
algorithm in its simplest form, written with three hooks and the public pieces of the
package only. It proves that the same loop runs with either response model, that the
plain word goes through the registry, and that planted groups are found again.
"""

import math
from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene import Result
from civiqhub_selene.blocks import OrdinalScale, Partition, VoteMatrix
from civiqhub_selene.blocks._arrays import IntArray
from civiqhub_selene.blocks._numerics import History, count_states, one_hot
from civiqhub_selene.blocks.base import BlockModel, estimator_for, memberships
from civiqhub_selene.blocks.responses import Categorical, OrdinalProbit
from civiqhub_selene.blocks.results import (
    HasIntegratedLikelihood,
    PredictsVotes,
    SingleCoclustering,
)
from civiqhub_selene.blocks.synthetic import make_votes, single_memberships
from civiqhub_selene.registry import Registry
from civiqhub_selene.testing import EstimatorContract
from civiqhub_selene.types import UnavailableReason

pytestmark = pytest.mark.blocks


@dataclass(frozen=True)
class Assignment:
    rows: IntArray
    cols: IntArray
    parameters: Any
    objective: float


@dataclass(frozen=True, kw_only=True)
class ToyModel(BlockModel[Assignment, SingleCoclustering]):
    name = "toy_single_model"
    version = "0.1.0"
    reference = "Architecture smoke test: a classification algorithm in its simplest form."
    membership = "toy"
    meaning = "each voter belongs to one group (test model)"
    default_criterion = "integrated_likelihood"

    def _start(self, votes: VoteMatrix, rng: np.random.Generator) -> Assignment:
        n_voters, n_items = votes.shape
        shape = (self.n_groups, self.n_blocks)
        return Assignment(
            rng.integers(0, self.n_groups, size=n_voters),
            rng.integers(0, self.n_blocks, size=n_items),
            self.response_model.initial_parameters(shape, votes, rng),
            -math.inf,
        )

    def _step(self, votes: VoteMatrix, state: Assignment) -> Assignment:
        states = votes._states()
        shape = (self.n_groups, self.n_blocks)
        counts = count_states(states, state.rows, state.cols, shape)
        parameters = self.response_model.update(counts, state.parameters)
        logs = self.response_model.log_probabilities(parameters)
        rows = np.einsum("ijs,qjs->iq", states, logs[:, state.cols, :]).argmax(axis=1)
        cols = np.einsum("ijs,ils->jl", states, logs[rows]).argmax(axis=1)
        objective = float(np.einsum("ijs,ijs->", states, logs[rows][:, cols, :]))
        return Assignment(rows, cols, parameters, objective)

    def _finish(
        self, votes: VoteMatrix, state: Assignment, history: History, computation: str
    ) -> SingleCoclustering:
        shape = (self.n_groups, self.n_blocks)
        counts = count_states(votes._states(), state.rows, state.cols, shape)
        return SingleCoclustering(
            computation=computation,
            voters=votes.voters,
            items=votes.items,
            item_scales=votes.item_scales,
            group_index=state.rows,
            block_index=state.cols,
            group_memberships=one_hot(state.rows, self.n_groups),
            block_memberships=one_hot(state.cols, self.n_blocks),
            coclusters=self.response_model.to_table(state.parameters, counts, votes.scale),
            objective=state.objective,
            history=history,
            integrated_likelihood=state.objective
            - self.response_model.complexity_penalty(shape, votes),
        )


def planted() -> tuple[VoteMatrix, Partition[Any], Partition[Any]]:
    votes, truth = make_votes(
        single_memberships((40, 40)),
        single_memberships((6, 6)),
        vote_rate=[[0.85, 0.45], [0.55, 0.90]],
        latent_mean=[[4.7, 1.4], [1.3, 4.6]],
        latent_spread=[[0.6, 0.7], [0.7, 0.6]],
        seed=0,
    )
    groups = Partition.from_voter_labels(votes.voters, truth.roles[:, 0])
    blocks = Partition.from_item_labels(votes.items, np.repeat([0, 1], 6))
    return votes, groups, blocks


@pytest.mark.parametrize("response_model", [OrdinalProbit(), Categorical()])
def test_the_same_loop_finds_planted_groups_with_either_response_model(
    response_model: Any,
) -> None:
    votes, groups, blocks = planted()
    model = ToyModel(n_groups=2, n_blocks=2, response_model=response_model, n_starts=6)
    result: Result[SingleCoclustering] = model.estimate(votes, seed=0)
    value = result.value
    assert value.groups.agreement_with(groups) >= 0.95
    assert value.blocks.agreement_with(blocks) == 1.0
    assert value.history.n_iterations >= 1
    assert np.all(np.diff(value.history.objectives) >= -1e-6)
    assert value.computation == result.fingerprint()
    assert value.groups.computation == result.fingerprint()
    assert isinstance(value, HasIntegratedLikelihood)
    assert isinstance(value, PredictsVotes)
    assert float(value.log_probability(votes)[votes.shown].sum()) == pytest.approx(
        value.objective, rel=1e-9
    )
    # The latent parameters are read only under the ordinal probit model.
    has_latent = value[0, 0].latent_mean is not None
    assert has_latent == isinstance(response_model, OrdinalProbit)


def test_two_configurations_of_one_model_are_two_computations_to_compare() -> None:
    votes, groups, _ = planted()
    probit = ToyModel(response_model=OrdinalProbit()).estimate(votes, seed=0)
    categorical = ToyModel(response_model=Categorical()).estimate(votes, seed=0)
    assert probit.fingerprint() != categorical.fingerprint()
    assert probit == ToyModel(response_model=OrdinalProbit()).estimate(votes, seed=0)
    assert probit.value.fingerprint() == ToyModel().estimate(votes, seed=0).value.fingerprint()
    # Both find the same groups; they are compared through their partitions.
    assert probit.value.groups.agreement_with(categorical.value.groups) >= 0.95
    assert probit.value.groups.agreement_with(groups) >= 0.95
    # The groups found feed any reading of the votes through groups, with their lineage.
    grouped = votes.group_by(probit.value.groups, probit.value.blocks)
    assert grouped.lineage == (probit.fingerprint(),)
    assert grouped.to_table().shape == (2, 2)


def test_the_plain_word_goes_through_the_registry() -> None:
    votes, _, _ = planted()
    registry = Registry.from_classes({"coclustering.toy": ToyModel})
    assert memberships(registry) == {"toy": "each voter belongs to one group (test model)"}
    estimator = estimator_for("toy", registry)
    assert estimator == ToyModel()
    result = estimator.with_shape(2, 2).estimate(votes, seed=0)
    assert result == ToyModel(n_groups=2, n_blocks=2).estimate(votes, seed=0)
    assert result.provenance.to_dict()["parameters"]["response_model"]["type"] == "OrdinalProbit"


class TestToyModel(EstimatorContract):
    def make_estimator(self) -> ToyModel:
        return ToyModel(n_starts=2)

    def make_data(self) -> VoteMatrix:
        return planted()[0]

    def make_empty_data(self) -> VoteMatrix:
        return VoteMatrix.from_records([])


def test_a_block_model_written_for_one_scale_refuses_several() -> None:
    # TEST ONLY. Two items of two answers and one of four.
    yes_no, four = OrdinalScale((0, 1)), OrdinalScale((1, 2, 3, 4))
    votes = VoteMatrix.from_responses(
        [[1, 4, 0], [0, 2, 1], [1, 1, 1], [0, 3, 1]],
        item_scales={"q1": yes_no, "q2": four, "q3": yes_no},
    )
    assert not ToyModel.accepts_several_scales
    result = ToyModel().estimate(votes, seed=0)
    assert not result.is_available
    assert result.unavailable is not None
    assert result.unavailable.reason is UnavailableReason.UNSUPPORTED_INPUT
    assert "ToyModel reads items on one scale" in result.unavailable.detail
