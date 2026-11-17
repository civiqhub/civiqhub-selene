# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import json
import pathlib
from typing import Any

import pytest

pytest.importorskip("numpy")

import numpy as np

from civiqhub_selene.blocks._numerics import (
    fit_latent_normal,
    level_probabilities,
    truncated_moments,
)

pytestmark = pytest.mark.blocks

VECTORS: dict[str, Any] = json.loads(
    pathlib.Path(__file__).with_name("blocks_numerics.json").read_text(encoding="utf-8")
)
THRESHOLDS = VECTORS["thresholds"]


@pytest.mark.parametrize("vector", VECTORS["level_probabilities"])
def test_level_probabilities(vector: dict[str, Any]) -> None:
    computed = level_probabilities(vector["mean"], vector["spread"], THRESHOLDS)
    assert computed == pytest.approx(vector["expected"], abs=vector["tolerance"])


@pytest.mark.parametrize("vector", VECTORS["truncated_moments"])
def test_truncated_moments(vector: dict[str, Any]) -> None:
    first, second = truncated_moments(vector["mean"], vector["spread"], THRESHOLDS)
    assert first == pytest.approx(vector["first"], abs=vector["tolerance"])
    assert second == pytest.approx(vector["second"], abs=vector["tolerance"])


@pytest.mark.parametrize("vector", VECTORS["fit_latent_normal"])
def test_fit_latent_normal(vector: dict[str, Any]) -> None:
    settings = VECTORS["fit_latent_normal_settings"]
    mean, spread = fit_latent_normal(
        np.array(vector["counts"], dtype=np.float64),
        np.array(settings["start_mean"]),
        np.array(settings["start_spread"]),
        THRESHOLDS,
        min_spread=settings["min_spread"],
        tolerance=settings["tolerance"],
        max_iterations=settings["max_iterations"],
    )
    assert float(mean) == pytest.approx(vector["mean"], abs=settings["comparison_tolerance"])
    assert float(spread) == pytest.approx(vector["spread"], abs=settings["comparison_tolerance"])
