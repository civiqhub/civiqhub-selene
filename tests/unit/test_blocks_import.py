# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

import importlib
import subprocess
import sys

import pytest

import civiqhub_selene

WITHOUT_NUMPY = (
    "import sys\n"
    "sys.modules['numpy'] = None  # makes 'import numpy' fail, as if it were not installed\n"
    "import civiqhub_selene\n"
    "import civiqhub_selene.ordinal\n"
    "from civiqhub_selene.errors import MissingExtraError\n"
    "try:\n"
    "    import civiqhub_selene.blocks\n"
    "except MissingExtraError as error:\n"
    "    assert isinstance(error, ImportError)\n"
    "    assert 'civiqhub-selene[blocks]' in error.hint, error.hint\n"
    "    assert 'NumPy' in str(error), str(error)\n"
    "    assert isinstance(error.__cause__, ImportError)\n"
    "else:\n"
    "    raise AssertionError('civiqhub_selene.blocks was imported without NumPy')\n"
)


def test_without_numpy_the_core_imports_and_blocks_says_what_to_install() -> None:
    subprocess.run([sys.executable, "-c", WITHOUT_NUMPY], check=True)


@pytest.mark.blocks
def test_with_the_extra_blocks_imports() -> None:
    pytest.importorskip("numpy")
    blocks = importlib.import_module("civiqhub_selene.blocks")
    arrays = importlib.import_module("civiqhub_selene.blocks._arrays")
    assert blocks.__name__ == "civiqhub_selene.blocks"
    assert {"FloatArray", "IntArray", "BoolArray"} <= set(arrays.__all__)


def test_blocks_is_not_a_public_name_of_the_root() -> None:
    assert "blocks" not in civiqhub_selene.__all__


def test_without_numpy_loading_a_block_model_says_what_to_install() -> None:
    script = (
        "import sys\n"
        "sys.modules['numpy'] = None\n"
        "from civiqhub_selene import MissingExtraError, default_registry\n"
        "spec = default_registry()['coclustering.single']\n"
        "try:\n"
        "    spec.load()\n"
        "except MissingExtraError as error:\n"
        "    assert 'civiqhub-selene[blocks]' in error.hint, error.hint\n"
        "else:\n"
        "    raise AssertionError('the block model was loaded without NumPy')\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
