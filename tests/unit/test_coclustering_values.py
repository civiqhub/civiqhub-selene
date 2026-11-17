# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import itertools
import json
import math
import re
from typing import Any

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene.blocks import (
    Cocluster,
    CoclusterTable,
    ItemBlock,
    OrdinalScale,
    Partition,
    VoteMatrix,
    VoterGroup,
)
from civiqhub_selene.blocks._numerics import History
from civiqhub_selene.blocks.results import (
    Coclustering,
    HasIntegratedLikelihood,
    MixedCoclustering,
    PredictsVotes,
    Selection,
    SingleCoclustering,
)
from civiqhub_selene.errors import InvalidInputError
from civiqhub_selene.estimator import Fingerprinted

pytestmark = pytest.mark.blocks

# 4 voters, 3 items, 2 groups, 2 blocks.
VOTERS = ("v1", "v2", "v3", "v4")
ITEMS = ("i1", "i2", "i3")
GROUP_INDEX = [1, 0, 1, 0]
BLOCK_INDEX = [0, 1, 0]
VOTES = VoteMatrix.from_table(
    [[2, -2, 1], [-2, 2, None], [1, -1, 2], [-1, None, -2]], voters=VOTERS, items=ITEMS
)
TABLE = CoclusterTable(
    OrdinalScale(),
    vote_rate=np.array([[0.8, 0.5], [0.9, 0.6]]),
    vote_shares=np.array(
        [
            [[0.6, 0.3, 0.05, 0.03, 0.02], [0.02, 0.03, 0.05, 0.3, 0.6]],
            [[0.02, 0.03, 0.05, 0.4, 0.5], [0.5, 0.4, 0.05, 0.03, 0.02]],
        ]
    ),
    n_votes=np.array([[3.0, 1.0], [4.0, 2.0]]),
)
HISTORY = History((-40.0, -31.5, -31.2), converged=True)
COMPUTATION = "0123456789abcdef"


def one_hot(index: list[int], n: int) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.eye(n)[index]


def common(**changes: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "computation": COMPUTATION,
        "voters": VOTERS,
        "items": ITEMS,
        "scale": OrdinalScale(),
        "group_index": GROUP_INDEX,
        "block_index": BLOCK_INDEX,
        "group_memberships": one_hot(GROUP_INDEX, 2),
        "block_memberships": one_hot(BLOCK_INDEX, 2),
        "coclusters": TABLE,
        "objective": -31.2,
        "history": HISTORY,
    }
    fields.update(changes)
    return fields


def make_single(**changes: Any) -> SingleCoclustering:
    fields = common(integrated_likelihood=-45.5)
    fields.update(changes)
    return SingleCoclustering(**fields)


def make_mixed(**changes: Any) -> MixedCoclustering:
    profiles = np.array(list(itertools.product(range(2), repeat=2)))
    # Certain profiles: each voter follows its own group on both blocks.
    certain = np.zeros((4, 4))
    for voter, group in enumerate(GROUP_INDEX):
        certain[voter, group * 2 + group] = 1.0
    role_by_block = np.zeros((4, 2, 2))
    for voter, group in enumerate(GROUP_INDEX):
        role_by_block[voter, :, group] = 1.0
    fields = common(
        role_by_block=role_by_block,
        profiles=profiles,
        profile_probabilities=certain,
        log_likelihood=-30.0,
    )
    fields.update(changes)
    return MixedCoclustering(**fields)


# Common value -------------------------------------------------------------------------


def test_groups_and_blocks_are_partitions_that_cite_the_computation() -> None:
    value = make_single()
    assert isinstance(value.groups, Partition)
    assert value.groups.computation == COMPUTATION
    assert value.blocks.computation == COMPUTATION
    assert value.groups.kind is VoterGroup
    assert value.blocks.kind is ItemBlock
    assert value.groups.ids == VOTERS
    assert value.groups[1].voters == ("v1", "v3")
    assert value.blocks[0].items == ("i1", "i3")
    assert value.group_of("v1") == value.groups[1]
    assert value.group_of("v1").index == 1
    assert value.block_of("i2").index == 1
    assert isinstance(value.block_of("i2"), ItemBlock)
    assert value.groups is value.groups
    with pytest.raises(InvalidInputError, match="voter 'v9' is not in the partition"):
        value.group_of("v9")


def test_shape_and_coclusters() -> None:
    value = make_single()
    assert value.n_groups == 2
    assert value.n_blocks == 2
    cocluster = value[0, 1]
    assert isinstance(cocluster, Cocluster)
    assert cocluster == TABLE[0, 1]
    assert cocluster.typical_score == 2
    with pytest.raises(IndexError, match=r"table\[group, block\]"):
        value[2, 0]


def test_shares_of_a_single_membership_are_the_sizes_over_the_count() -> None:
    soft = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    value = make_single(group_memberships=soft)
    assert [group.share for group in value.groups] == [0.5, 0.5]
    assert [block.share for block in value.blocks] == [2 / 3, 1 / 3]
    assert value.groups.memberships is not None
    assert np.array_equal(value.groups.memberships, soft)


def test_display_order_sorts_by_group_and_keeps_the_order_inside_a_group() -> None:
    voters, items = make_single().display_order()
    assert voters.tolist() == [1, 3, 0, 2]
    assert items.tolist() == [0, 2, 1]
    assert voters.dtype == np.int64


@pytest.mark.parametrize("make", [make_single, make_mixed])
def test_to_dict_is_json_ready_even_with_an_infinite_objective(make: Any) -> None:
    value = make(objective=-math.inf, history=History((-math.inf, -31.2), converged=False))
    data = value.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert data["kind"] == type(value).__name__
    assert data["objective"] is None
    assert data["history"] == {"objectives": [None, -31.2], "converged": False}
    assert data["voters"] == list(VOTERS)
    assert data["items"] == list(ITEMS)
    assert data["scores"] == [-2, -1, 0, 1, 2]
    assert data["group_index"] == GROUP_INDEX
    assert data["block_index"] == BLOCK_INDEX
    assert data["group_memberships"] == [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    assert data["coclusters"]["vote_rate"] == [[0.8, 0.5], [0.9, 0.6]]
    assert data["selection"] is None
    assert data["computation"] == COMPUTATION
    assert data["fingerprint"] == value.fingerprint()


def test_to_dict_adds_the_fields_of_each_subclass() -> None:
    single = make_single().to_dict()
    assert single["integrated_likelihood"] == -45.5
    assert "log_likelihood" not in single
    mixed = make_mixed().to_dict()
    assert mixed["log_likelihood"] == -30.0
    assert mixed["profiles"] == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert np.array(mixed["role_by_block"]).shape == (4, 2, 2)
    assert np.array(mixed["profile_probabilities"]).shape == (4, 4)
    assert "integrated_likelihood" not in mixed
    assert make_single(integrated_likelihood=-math.inf).to_dict()["integrated_likelihood"] is None


def test_fingerprint_follows_what_was_found_not_how() -> None:
    value = make_single()
    readable: Fingerprinted = value
    assert re.fullmatch(r"[0-9a-f]{16}", readable.fingerprint())
    assert value.fingerprint() == make_single().fingerprint()
    assert value.fingerprint() != make_single(group_index=[1, 0, 0, 0]).fingerprint()
    assert value.fingerprint() != make_single(voters=("v1", "v2", "v3", "v5")).fingerprint()
    soft = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    assert value.fingerprint() != make_single(group_memberships=soft).fingerprint()
    # The history, the objective and the computation do not enter it.
    other_history = make_single(history=History((-1.0,), converged=False), objective=-1.0)
    assert value.fingerprint() == other_history.fingerprint()
    assert value.fingerprint() == make_single(computation="ffffffffffffffff").fingerprint()


def test_arrays_are_copied_read_only_and_the_value_is_frozen() -> None:
    index = np.array(GROUP_INDEX)
    value = make_single(group_index=index)
    index[0] = 0
    assert value.group_index.tolist() == GROUP_INDEX
    for array in (value.group_index, value.block_index, value.group_memberships):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        value.objective = 0.0  # type: ignore[misc]
    with pytest.raises(TypeError, match="positional"):
        SingleCoclustering(COMPUTATION, VOTERS)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"group_index": [1, 0, 1]}, "group_index has the shape (4,)"),
        ({"block_index": [0, 1]}, "block_index has the shape (3,)"),
        ({"group_index": [2, 0, 1, 0]}, "group_index holds indices from 0 to 1"),
        ({"group_memberships": np.ones((4, 3)) / 3}, "group_memberships has the shape (4, 2)"),
        ({"block_memberships": np.ones((2, 2)) / 2}, "block_memberships has the shape (3, 2)"),
        ({"coclusters": "table"}, "coclusters are a CoclusterTable"),
        ({"scale": OrdinalScale((1, 2, 3, 4, 5))}, "scale"),
    ],
)
def test_inconsistent_values_are_refused(changes: dict[str, Any], named: str) -> None:
    with pytest.raises(InvalidInputError, match=re.escape(named)):
        make_single(**changes)


def test_description() -> None:
    lines = str(make_single()).splitlines()
    assert lines[0] == "SingleCoclustering(4 voters in 2 groups x 3 items in 2 blocks)"
    assert lines[1] == "groups of 2, 2 voters; blocks of 2, 1 items"
    assert lines[2] == "group 0 x block 0: vote rate 0.80, typical score -2, 3 votes"
    assert len(lines) == 6
    assert repr(make_mixed()) == "MixedCoclustering(4 voters in 2 groups x 3 items in 2 blocks)"
    base = Coclustering(**common())
    assert repr(base) == "Coclustering(4 voters in 2 groups x 3 items in 2 blocks)"


# SingleCoclustering -------------------------------------------------------------------


def test_single_log_probability_sums_to_the_hand_computed_log_likelihood() -> None:
    value = make_single()
    logs = value.log_probability(VOTES)
    assert logs.shape == (4, 3)
    expected = 0.0
    for voter, group in enumerate(GROUP_INDEX):
        for item, block in enumerate(BLOCK_INDEX):
            cocluster = TABLE[group, block]
            level = int(VOTES.levels[voter, item])
            if level == 0:
                probability = 1 - cocluster.vote_rate
            else:
                probability = cocluster.vote_rate * cocluster.vote_shares[level - 1]
            assert logs[voter, item] == pytest.approx(math.log(probability))
            if VOTES.shown[voter, item]:
                expected += math.log(probability)
    assert float(logs[VOTES.shown].sum()) == pytest.approx(expected)


def test_log_probability_reads_the_votes_without_shown() -> None:
    value = make_single()
    hidden = VOTES.hold_out(np.eye(4, 3, dtype=bool))
    assert np.array_equal(value.log_probability(hidden), value.log_probability(VOTES))


def test_log_probability_needs_the_matrix_of_the_coclustering() -> None:
    value = make_single()
    with pytest.raises(InvalidInputError, match="same voters, items and scale"):
        value.log_probability(VOTES[["v1", "v2"], :])
    with pytest.raises(InvalidInputError, match="votes are a VoteMatrix"):
        value.log_probability("votes")  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError, match="same voters, items and scale"):
        make_mixed().log_probability(VOTES[:, ["i1", "i2"]])


# MixedCoclustering --------------------------------------------------------------------


def test_mixed_with_certain_profiles_predicts_like_single() -> None:
    mixed, single = make_mixed(), make_single()
    assert mixed.log_probability(VOTES) == pytest.approx(single.log_probability(VOTES), abs=1e-9)
    assert mixed.mixedness.tolist() == [0.0, 0.0, 0.0, 0.0]
    changes = mixed.role_changes()
    assert changes.tolist() == [[2, 0], [0, 2]]
    assert changes.sum() == 4
    assert changes.dtype == np.int64


def test_mixed_log_probability_follows_its_formula_on_spread_memberships() -> None:
    rng = np.random.default_rng(3)
    profile_probabilities = rng.dirichlet(np.ones(4), size=4)
    block_memberships = rng.dirichlet(np.ones(2), size=3)
    mixed = make_mixed(
        profile_probabilities=profile_probabilities, block_memberships=block_memberships
    )
    logs = mixed.log_probability(VOTES)
    probabilities = np.exp(TABLE.log_probabilities())
    for voter in range(4):
        for item in range(3):
            state = int(VOTES.levels[voter, item])
            total = 0.0
            for profile, weight in zip(mixed.profiles, profile_probabilities[voter], strict=True):
                total += weight * sum(
                    block_memberships[item, block] * probabilities[profile[block], block, state]
                    for block in range(2)
                )
            assert logs[voter, item] == pytest.approx(math.log(total), abs=1e-12)


def test_mixedness_and_role_changes_of_voters_who_change_role() -> None:
    role_by_block = np.zeros((4, 2, 2))
    role_by_block[:, 0] = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]]
    role_by_block[:, 1] = [[0.2, 0.8], [0.1, 0.9], [0.7, 0.3], [0.6, 0.4]]
    memberships = np.array([[0.5, 0.5], [0.1, 0.9], [0.7, 0.3], [0.45, 0.55]])
    mixed = make_mixed(role_by_block=role_by_block, group_memberships=memberships)
    assert mixed.mixedness.tolist() == pytest.approx([0.5, 0.1, 0.3, 0.45])
    assert mixed.role_changes().tolist() == [[1, 1], [1, 1]]
    assert mixed.role_changes(1, 0).tolist() == [[1, 1], [1, 1]]
    assert mixed.role_changes(0, 0).tolist() == [[2, 0], [0, 2]]
    with pytest.raises(InvalidInputError, match="2 blocks, got the block 2"):
        mixed.role_changes(0, 2)
    # Shares of a mixed membership are the mean of the memberships.
    assert [group.share for group in mixed.groups] == pytest.approx([0.4375, 0.5625])
    assert "histogram" in (MixedCoclustering.mixedness.__doc__ or "")


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"role_by_block": np.zeros((4, 2, 3))}, "role_by_block has the shape (4, 2, 2)"),
        ({"profiles": np.zeros((3, 2), dtype=int)}, "profiles has the shape (4, 2)"),
        ({"profile_probabilities": np.zeros((4, 3))}, "profile_probabilities has the shape (4, 4)"),
        ({"profiles": np.full((4, 2), 2)}, "profiles hold groups from 0 to 1"),
    ],
)
def test_inconsistent_mixed_values_are_refused(changes: dict[str, Any], named: str) -> None:
    with pytest.raises(InvalidInputError, match=re.escape(named)):
        make_mixed(**changes)


# Protocols ----------------------------------------------------------------------------


def test_protocols_tell_what_a_value_can_do() -> None:
    single, mixed = make_single(), make_mixed()
    assert isinstance(single, HasIntegratedLikelihood)
    assert isinstance(single, PredictsVotes)
    assert isinstance(mixed, PredictsVotes)
    assert not isinstance(mixed, HasIntegratedLikelihood)
    base = Coclustering(**common())
    assert not isinstance(base, HasIntegratedLikelihood)
    assert not isinstance(base, PredictsVotes)
    assert single.integrated_likelihood == -45.5


# Selection ----------------------------------------------------------------------------


def make_selection(**changes: Any) -> Selection:
    fields: dict[str, Any] = {
        "criterion": "integrated_likelihood",
        "n_groups": (1, 2, 3),
        "n_blocks": (1, 2),
        "scores": [[-900.0, -850.0], [-820.5, -790.25], [-800.0, -math.inf]],
        "best": (2, 2),
    }
    fields.update(changes)
    return Selection(**fields)


def test_selection_is_the_trace_of_a_choice_of_shape() -> None:
    selection = make_selection(alternatives=[(3, 1)])
    assert selection.criterion == "integrated_likelihood"
    assert selection.scores.shape == (3, 2)
    assert selection.best == (2, 2)
    assert selection.alternatives == ((3, 1),)
    assert selection.standard_errors is None
    data = selection.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert data["scores"][2] == [-800.0, None]
    assert data["best"] == [2, 2]
    assert data["alternatives"] == [[3, 1]]
    assert data["standard_errors"] is None
    with pytest.raises(ValueError, match="read-only"):
        selection.scores[0, 0] = 0.0


def test_selection_with_standard_errors_and_its_description() -> None:
    selection = make_selection(standard_errors=np.full((3, 2), 0.5))
    assert selection.to_dict()["standard_errors"] == [[0.5, 0.5]] * 3
    lines = str(selection).splitlines()
    assert lines[0] == "Selection(integrated_likelihood: 2 groups x 2 blocks among 6 shapes)"
    assert lines[1].split() == ["groups", "\\", "blocks", "1", "2"]
    assert lines[3].split() == ["2", "-820.50", "-790.25*"]
    assert len(lines) == 5


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"scores": [[1.0, 2.0]]}, "scores has the shape (3, 2)"),
        ({"standard_errors": [1.0]}, "standard_errors has the shape (3, 2)"),
        ({"best": (4, 2)}, "best shape (4, 2) is not among the shapes compared"),
        ({"best": (2, 3)}, "best shape (2, 3) is not among the shapes compared"),
    ],
)
def test_inconsistent_selections_are_refused(changes: dict[str, Any], named: str) -> None:
    with pytest.raises(InvalidInputError, match=re.escape(named)):
        make_selection(**changes)


def test_a_value_carries_the_trace_of_its_selection() -> None:
    selection = make_selection()
    value = make_single(selection=selection)
    assert value.selection is selection
    assert value.to_dict()["selection"] == selection.to_dict()
    assert value.fingerprint() == make_single().fingerprint()
    copy = dataclasses.replace(make_single(), selection=selection)
    assert copy.selection is selection
