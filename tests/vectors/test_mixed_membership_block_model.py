# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import json
import pathlib
from typing import Any

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene.blocks import (
    MixedMembershipBlockModel,
    OrdinalScale,
    SpectralStart,
    VoteMatrix,
)
from civiqhub_selene.blocks.mixed import _expect, _profiles

pytestmark = pytest.mark.blocks

VECTOR: dict[str, Any] = json.loads(
    pathlib.Path(__file__)
    .with_name("mixed_membership_block_model.json")
    .read_text(encoding="utf-8")
)
VOTES = VoteMatrix.from_arrays(
    np.array(VECTOR["votes"]["levels"]), scale=OrdinalScale(tuple(VECTOR["votes"]["scores"]))
)


def test_the_expectation_step_agrees_with_the_reference_engine() -> None:
    fixed, expected = VECTOR["fixed"], VECTOR["expectation"]
    tolerance = expected["tolerance"]
    theta = np.array(fixed["voter_memberships"])
    expectation = _expect(
        VOTES._states(),
        VOTES.shown,
        theta,
        np.array(fixed["item_memberships"]),
        np.log(np.array(fixed["state_probabilities"])),
        _profiles(3, 2),
    )
    assert expectation.log_likelihood == pytest.approx(expected["log_likelihood"], abs=tolerance)
    penalised = expectation.log_likelihood + 0.2 * float(np.log(theta).sum())
    assert penalised == pytest.approx(
        expected["objective_with_voter_concentration_1.2"], abs=tolerance
    )
    for name in ("role_by_block", "counts", "item_mass"):
        computed = getattr(expectation, name)
        assert computed == pytest.approx(np.array(expected[name]), abs=tolerance), name


def test_reference_run_on_the_stored_votes() -> None:
    run = VECTOR["run"]
    settings = run["settings"]
    assert settings["initialization"] == "SpectralStart()"
    model = MixedMembershipBlockModel(
        n_groups=settings["n_groups"],
        n_blocks=settings["n_blocks"],
        n_starts=settings["n_starts"],
        max_iterations=settings["max_iterations"],
        tolerance=settings["tolerance"],
        voter_concentration=settings["voter_concentration"],
        item_concentration=settings["item_concentration"],
        start_weight=settings["start_weight"],
        initialization=SpectralStart(),
    )
    assert type(model).method().name == VECTOR["method"]["name"]
    assert type(model).method().version == VECTOR["method"]["version"]
    value = model.estimate(VOTES, seed=settings["seed"]).value
    assert value.history.converged
    assert value.objective == pytest.approx(run["objective"], abs=run["tolerance"])
    assert value.log_likelihood == pytest.approx(run["log_likelihood"], abs=run["tolerance"])
    # The reference engine, from its own random starts, reaches the same optimum.
    assert value.objective == pytest.approx(
        run["javascript_best_objective"], abs=run["javascript_tolerance"]
    )
