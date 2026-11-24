# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Run the notebooks of the examples from the top and save them with their outputs.

Needs the ``examples`` dependency group::

    python -m pip install -e ".[viz]" --group examples
    python examples/run_notebooks.py                 # every notebook
    python examples/run_notebooks.py 02_olbm.ipynb   # some of them

Each notebook runs in a fresh kernel, from this folder. The timestamps that Jupyter can
record in the cells are left out, so that a diff only shows what changed.
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent


def run(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        record_timing=False,
        resources={"metadata": {"path": str(HERE)}},
    ).execute()
    for cell in notebook.cells:
        cell.metadata.pop("execution", None)
    nbformat.write(notebook, path)
    print(f"ran {path.name}")


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(path.name for path in HERE.glob("*.ipynb"))
    for name in names:
        run(HERE / name)
