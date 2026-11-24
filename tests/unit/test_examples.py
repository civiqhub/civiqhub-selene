# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Every notebook of ``examples/`` runs from top to bottom.

The code cells of a notebook are run in order, in one namespace, from the folder of the
examples, as a reader runs them; figures are drawn with the Agg backend and closed instead
of being shown. Jupyter is not needed: the notebooks are read as JSON.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")
pytest.importorskip("networkx")
pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
NOTEBOOKS = sorted(EXAMPLES.glob("*.ipynb"))

pytestmark = [
    pytest.mark.blocks,
    pytest.mark.viz,
    # The source distribution ships the tests without the examples.
    pytest.mark.skipif(not NOTEBOOKS, reason="this checkout has no examples"),
]


def code_cells(path: Path) -> list[str]:
    notebook: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_the_examples_are_found() -> None:
    assert [path.name for path in NOTEBOOKS] == [
        "01_spectral_start.ipynb",
        "02_olbm.ipynb",
        "03_mm_olbm.ipynb",
        "04_olbm_vs_mm_olbm.ipynb",
    ]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[path.stem for path in NOTEBOOKS])
def test_a_notebook_runs_from_top_to_bottom(
    path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(EXAMPLES)
    monkeypatch.syspath_prepend(str(EXAMPLES))
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: plt.close("all"))
    namespace: dict[str, Any] = {"__name__": "__main__"}
    try:
        for number, source in enumerate(code_cells(path), start=1):
            try:
                exec(compile(source, f"{path.name}, code cell {number}", "exec"), namespace)
            except Exception as error:
                raise AssertionError(
                    f"{path.name}, code cell {number} failed: {error!r}"
                ) from error
    finally:
        plt.close("all")
        sys.modules.pop("loading", None)
    assert capsys.readouterr().out


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[path.stem for path in NOTEBOOKS])
def test_a_notebook_is_saved_with_its_outputs_and_without_errors(path: Path) -> None:
    notebook: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["outputs"] for cell in cells[1:]), "run the notebook before saving it"
    assert not any(output["output_type"] == "error" for cell in cells for output in cell["outputs"])
    assert all("execution" not in cell["metadata"] for cell in cells), (
        "strip the execution timestamps: see examples/README.md"
    )
