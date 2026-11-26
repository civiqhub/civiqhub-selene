# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import ast
import itertools
import math
import pathlib

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene.blocks import _numerics, responses
from civiqhub_selene.blocks._numerics import (
    History,
    count_states,
    fit_latent_normal,
    interval_mass,
    iterate,
    level_probabilities,
    logsumexp,
    norm_cdf,
    norm_pdf,
    one_hot,
    truncated_moments,
)

pytestmark = pytest.mark.blocks

THRESHOLDS = [1.5, 2.5, 3.5, 4.5]


# Counting -----------------------------------------------------------------------------


def test_one_hot() -> None:
    indicator = one_hot(np.array([0, 2, 1]), 3)
    assert indicator.tolist() == [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    assert indicator.dtype == np.float64
    assert one_hot(np.array([], dtype=np.int64), 2).shape == (0, 2)


def test_count_states_matches_a_direct_count() -> None:
    rng = np.random.default_rng(0)
    states = one_hot(rng.integers(0, 4, size=30), 4).reshape(6, 5, 4)
    rows = np.array([0, 0, 1, 1, 2, 2])
    cols = np.array([0, 1, 1, 0, 1])
    counts = count_states(states, rows, cols, (3, 2))
    assert counts.shape == (3, 2, 4)
    expected = np.zeros((3, 2, 4))
    for i in range(6):
        for j in range(5):
            expected[rows[i], cols[j]] += states[i, j]
    assert np.array_equal(counts, expected)
    assert counts.sum() == 30


def test_logsumexp_is_stable() -> None:
    values = np.array([[1000.0, 1000.0], [-1000.0, -1000.0], [0.0, -np.inf]])
    result = logsumexp(values, axis=1)
    assert result.tolist() == pytest.approx([1000.0 + np.log(2), -1000.0 + np.log(2), 0.0])
    assert logsumexp(np.array([-np.inf, -np.inf]), axis=0) == -np.inf
    small = np.log(np.array([[0.1, 0.2, 0.7]]))
    assert logsumexp(small, axis=1).tolist() == pytest.approx([0.0], abs=1e-12)
    assert logsumexp(np.zeros((2, 3, 4)), axis=-1).shape == (2, 3)


# Normal law ---------------------------------------------------------------------------


def test_normal_distribution_function_and_density() -> None:
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(np.array([-np.inf, np.inf])).tolist() == [0.0, 1.0]
    assert float(norm_cdf(1.959963984540054)) == pytest.approx(0.975, abs=1e-12)
    # Accurate in the lower tail, where 1 - cdf(-x) would lose every digit.
    assert float(norm_cdf(-10.0)) == pytest.approx(7.619853024160527e-24, rel=1e-10)
    assert float(norm_pdf(0.0)) == pytest.approx(1 / math.sqrt(2 * math.pi))
    assert norm_pdf(np.array([-np.inf, np.inf])).tolist() == [0.0, 0.0]
    assert norm_cdf(np.zeros((2, 3))).shape == (2, 3)


def test_interval_mass_stays_positive_and_accurate_far_in_both_tails() -> None:
    # A level more than 8 standard deviations away from the mean, on both sides.
    upper = interval_mass(np.array([8.5, 9.5]))
    lower = interval_mass(np.array([-9.5, -8.5]))
    expected = 0.5 * (math.erfc(8.5 / math.sqrt(2)) - math.erfc(9.5 / math.sqrt(2)))
    assert expected > 0
    assert float(upper[0]) == pytest.approx(expected, rel=1e-10)
    assert float(lower[0]) == pytest.approx(expected, rel=1e-10)
    # The naive difference of distribution functions is zero there.
    assert float(norm_cdf(9.5) - norm_cdf(8.5)) == 0.0


def test_level_probabilities_reference_values() -> None:
    probabilities = level_probabilities(3.0, 1.0, THRESHOLDS)
    assert probabilities == pytest.approx(
        [0.0668072, 0.2417303, 0.3829249, 0.2417303, 0.0668072], abs=1e-7
    )


def test_level_probabilities_sum_to_one() -> None:
    rng = np.random.default_rng(4)
    means = rng.uniform(-2, 8, size=20)
    spreads = rng.uniform(0.1, 4, size=20)
    probabilities = level_probabilities(means, spreads, THRESHOLDS)
    assert probabilities.shape == (20, 5)
    assert probabilities.sum(axis=-1) == pytest.approx(np.ones(20), abs=1e-12)
    assert (probabilities >= 0).all()
    grid = level_probabilities(np.full((2, 3), 3.0), np.ones((2, 3)), [2.0, 4.0])
    assert grid.shape == (2, 3, 3)


def integrate(mean: float, spread: float, low: float, high: float) -> tuple[float, float]:
    """First two moments of a normal law truncated to (low, high), by a midpoint sum."""
    low = max(low, mean - 14 * spread)
    high = min(high, mean + 14 * spread)
    edges = np.linspace(low, high, 400_001)
    points = (edges[:-1] + edges[1:]) / 2
    density = np.exp(-0.5 * ((points - mean) / spread) ** 2)
    mass = density.sum()
    return float((points * density).sum() / mass), float((points**2 * density).sum() / mass)


@pytest.mark.parametrize(
    ("mean", "spread"), [(3.0, 1.0), (4.2, 0.8), (1.4, 0.7), (2.2, 1.6), (3.6, 0.45)]
)
def test_truncated_moments_match_a_direct_numerical_integration(mean: float, spread: float) -> None:
    first, second = truncated_moments(mean, spread, THRESHOLDS)
    bounds = [-math.inf, *THRESHOLDS, math.inf]
    probabilities = level_probabilities(mean, spread, THRESHOLDS)
    for level in range(5):
        if probabilities[level] < 1e-9:
            continue
        expected_first, expected_second = integrate(mean, spread, bounds[level], bounds[level + 1])
        assert float(first[level]) == pytest.approx(expected_first, abs=1e-6)
        assert float(second[level]) == pytest.approx(expected_second, abs=1e-6)
    # The moments recombine into the moments of the whole law.
    assert float((probabilities * first).sum()) == pytest.approx(mean, abs=1e-9)
    assert float((probabilities * second).sum()) == pytest.approx(mean**2 + spread**2, abs=1e-8)


def test_truncated_moments_stay_inside_their_level_and_finite_in_the_tails() -> None:
    first, second = truncated_moments(np.array([30.0, -30.0]), np.array([0.3, 0.3]), THRESHOLDS)
    bounds = np.array([-np.inf, *THRESHOLDS, np.inf])
    assert np.isfinite(first).all()
    assert np.isfinite(second).all()
    assert (first >= bounds[:-1]).all()
    assert (first <= bounds[1:]).all()
    assert (second >= first**2).all()


# Loop ---------------------------------------------------------------------------------


def test_iterate_stops_converged_when_the_gap_goes_under_the_tolerance() -> None:
    state, history = iterate(
        lambda x: x / 2,
        16.0,
        gap=lambda old, new: old - new,
        value=lambda x: -x,
        tolerance=1.5,
        max_iterations=50,
    )
    # 16 -> 8 -> 4 -> 2 -> 1: the gap 2 - 1 = 1 is the first one under 1.5.
    assert state == 1.0
    assert history == History((-8.0, -4.0, -2.0, -1.0), converged=True)
    assert history.n_iterations == 4


def test_iterate_stops_unconverged_at_the_last_step() -> None:
    state, history = iterate(
        lambda x: x + 1,
        0,
        gap=lambda old, new: 1.0,
        value=float,
        tolerance=1e-6,
        max_iterations=3,
    )
    assert state == 3
    assert history.objectives == (1.0, 2.0, 3.0)
    assert history.converged is False
    assert history.n_iterations == 3
    state, history = iterate(
        lambda x: x + 1, 0, gap=lambda old, new: 1.0, value=float, tolerance=1.0, max_iterations=0
    )
    assert state == 0
    assert history == History((), converged=False)


# Latent normal opinion ----------------------------------------------------------------


def test_fit_latent_normal_recovers_a_planted_opinion() -> None:
    rng = np.random.default_rng(11)
    opinions = rng.normal(4.2, 0.8, size=5000)
    levels = 1 + (opinions[:, np.newaxis] > np.array(THRESHOLDS)).sum(axis=1)
    counts = np.bincount(levels, minlength=6)[1:].astype(np.float64)
    assert counts.sum() == 5000
    mean, spread = fit_latent_normal(
        counts, np.array(3.0), np.array(1.0), THRESHOLDS, min_spread=0.3, max_iterations=500
    )
    assert float(mean) == pytest.approx(4.2, abs=0.05)
    assert float(spread) == pytest.approx(0.8, abs=0.05)


def test_a_unanimous_cocluster_has_the_minimal_spread_and_no_nan() -> None:
    counts = np.array([[0.0, 0.0, 0.0, 0.0, 40.0], [40.0, 0.0, 0.0, 0.0, 0.0]])
    mean, spread = fit_latent_normal(
        counts, np.full(2, 3.0), np.ones(2), THRESHOLDS, min_spread=0.3, max_iterations=500
    )
    assert spread.tolist() == [0.3, 0.3]
    assert np.isfinite(mean).all()
    assert mean[0] > 4.5
    assert mean[1] < 1.5


def test_a_cocluster_without_a_vote_keeps_its_parameters() -> None:
    counts = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [3.0, 9.0, 14.0, 6.0, 2.0]])
    mean, spread = fit_latent_normal(
        counts,
        np.array([2.25, 3.0]),
        np.array([0.75, 1.0]),
        THRESHOLDS,
        min_spread=0.3,
        max_iterations=50,
    )
    assert float(mean[0]) == 2.25
    assert float(spread[0]) == 0.75
    assert float(mean[1]) == pytest.approx(2.848198, abs=1e-5)


def test_fit_latent_normal_takes_at_most_the_steps_allowed_and_any_shape() -> None:
    counts = np.array([3.0, 9.0, 14.0, 6.0, 2.0])
    one_step = fit_latent_normal(
        counts, np.array(3.0), np.array(1.0), THRESHOLDS, min_spread=0.3, max_iterations=1
    )
    many = fit_latent_normal(
        counts, np.array(3.0), np.array(1.0), THRESHOLDS, min_spread=0.3, max_iterations=200
    )
    assert float(one_step[0]) != pytest.approx(float(many[0]), abs=1e-6)
    grid = np.broadcast_to(counts, (2, 3, 5)).copy()
    mean, spread = fit_latent_normal(
        grid, np.full((2, 3), 3.0), np.ones((2, 3)), THRESHOLDS, min_spread=0.3, max_iterations=200
    )
    assert mean.shape == spread.shape == (2, 3)
    assert mean == pytest.approx(np.full((2, 3), float(many[0])))
    empty = fit_latent_normal(
        np.zeros((0, 5)), np.zeros(0), np.ones(0), THRESHOLDS, min_spread=0.3, max_iterations=5
    )
    assert empty[0].shape == (0,)


# Randomness ---------------------------------------------------------------------------


def test_no_function_draws_a_random_number_except_initial_parameters() -> None:
    numerics_source = pathlib.Path(_numerics.__file__).read_text(encoding="utf-8")
    assert "random" not in numerics_source.replace("draws a random number", "")
    tree = ast.parse(pathlib.Path(responses.__file__).read_text(encoding="utf-8"))
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        uses_generator = any(
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "rng"
            for node in ast.walk(function)
        )
        if uses_generator:
            assert function.name == "initial_parameters", function.name
    source = pathlib.Path(responses.__file__).read_text(encoding="utf-8")
    assert "default_rng" not in source
    assert "np.random.seed" not in source


# The inner thresholds of a scale ------------------------------------------------------


def test_a_threshold_step_keeps_the_order_and_never_lowers_the_objective() -> None:
    from civiqhub_selene.blocks._numerics import (
        fit_thresholds,
        level_probabilities,
        threshold_objective,
    )

    rng = np.random.default_rng(0)
    planted = np.array([1 / 6, 0.25, 0.45, 0.7, 5 / 6])
    mean = rng.uniform(0.2, 0.8, size=(3, 2))
    spread = np.full((3, 2), 0.15)
    counts = 2000 * level_probabilities(mean, spread, planted)
    grid = np.array([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
    values = [threshold_objective(counts, mean, spread, grid)]
    for _ in range(8):
        grid = fit_thresholds(counts, mean, spread, grid, max_steps=1)
        assert (np.diff(grid) > 0).all()
        assert grid[0] == 1 / 6
        assert grid[-1] == 5 / 6
        values.append(threshold_objective(counts, mean, spread, grid))
    assert all(after >= before for before, after in itertools.pairwise(values))
    assert grid == pytest.approx(planted, abs=1e-6)
    # Two thresholds have no inner one: nothing moves.
    two = np.array([0.3, 0.6])
    assert np.array_equal(fit_thresholds(counts[..., :3], mean, spread, two, max_steps=5), two)
