# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Every Python block of the documentation runs, and prints what the page shows.

The rules, also written in ``docs/README.md``:

- a block fenced with three backticks and ``python`` is run; the blocks of a page share
  one namespace and run in the order of the page;
- a ``text`` block that directly follows a Python block is what that block prints;
- a Python block preceded by the comment ``<!-- not run: reason -->`` is skipped;
- a page whose blocks need NumPy carries the ``blocks`` marker; a page whose blocks need
  ``civiqhub_selene.viz`` is skipped when NetworkX or Matplotlib is missing.
"""

import contextlib
import io
import re
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

_FENCE = re.compile(r"^ *(?P<ticks>`{3,})(?P<language>.*)$")
_NOT_RUN = "<!-- not run:"
_NEEDS_NUMPY = re.compile(r"\b(numpy|civiqhub_selene\.blocks|civiqhub_selene import viz)\b")
_NEEDS_VIZ = re.compile(
    r"\b(civiqhub_selene\.viz|civiqhub_selene import viz|networkx|matplotlib)\b"
)


@dataclass(frozen=True)
class Block:
    """One Python block of a page.

    Attributes:
        line: Line of the opening fence, from 1.
        code: The code, without the indentation of the fence.
        output: The content of the ``text`` block that directly follows, if there is one.
        run: False when the block is preceded by the "not run" comment.
    """

    line: int
    code: str
    output: str | None
    run: bool


def blocks_of(text: str) -> list[Block]:
    """Return the Python blocks of a Markdown page, in order."""
    lines = text.splitlines()
    blocks: list[Block] = []
    previous = ""  # the last thing seen: a line of prose, "python" after a Python block
    index = 0
    while index < len(lines):
        fence = _FENCE.match(lines[index])
        if fence is None:
            if lines[index].strip():
                previous = lines[index].strip()
            index += 1
            continue
        # A fence closes on a line made of at least as many backticks as it opened with.
        end = index + 1
        while end < len(lines) and not re.fullmatch(f" *{fence['ticks']}`*", lines[end]):
            end += 1
        body = textwrap.dedent("\n".join(lines[index + 1 : end])) + "\n"
        language = fence["language"].strip()
        if language == "python":
            blocks.append(Block(index + 1, body, None, not previous.startswith(_NOT_RUN)))
            previous = "python"
        elif language == "text" and previous == "python":
            blocks[-1] = replace(blocks[-1], output=body)
            previous = "text"
        else:
            previous = language
        index = end + 1
    return blocks


def needs_numpy(blocks: list[Block]) -> bool:
    return any(_NEEDS_NUMPY.search(block.code) for block in blocks if block.run)


def needs_viz(blocks: list[Block]) -> bool:
    return any(_NEEDS_VIZ.search(block.code) for block in blocks if block.run)


def pages() -> list[Path]:
    """Return the README and every page under ``docs``, when the checkout holds them."""
    found = [ROOT / "README.md", *sorted(DOCS.rglob("*.md"))]
    return [page for page in found if page.is_file() and "_build" not in page.parts]


def _cases() -> list[Any]:
    cases = []
    for page in pages():
        blocks = blocks_of(page.read_text(encoding="utf-8"))
        if any(block.run for block in blocks):
            marks = [pytest.mark.blocks] if needs_numpy(blocks) else []
            cases.append(pytest.param(page, id=page.relative_to(ROOT).as_posix(), marks=marks))
    return cases


@pytest.mark.parametrize("page", _cases())
def test_the_python_blocks_of_a_page_run_and_print_what_the_page_shows(page: Path) -> None:
    blocks = blocks_of(page.read_text(encoding="utf-8"))
    if needs_numpy(blocks):
        pytest.importorskip("numpy")
    if needs_viz(blocks):
        pytest.importorskip("networkx")
        pytest.importorskip("matplotlib")
    namespace: dict[str, object] = {"__name__": "__docs__"}
    for block in blocks:
        if not block.run:
            continue
        where = f"{page.relative_to(ROOT).as_posix()}:{block.line}"
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            exec(compile(block.code, where, "exec"), namespace)
        if block.output is not None:
            assert printed.getvalue() == block.output, (
                f"the Python block at {where} does not print the text block that follows it; "
                "paste what the block prints, or remove the text block"
            )


def test_the_documentation_pages_are_found() -> None:
    names = {page.relative_to(ROOT).as_posix() for page in pages()}
    assert "README.md" in names
    if DOCS.is_dir():  # the source distribution ships the tests without the pages
        assert "docs/getting-started.md" in names


SAMPLE = """\
Some prose.

```python
x = 1
print(x)
```

```text
1
```

<!-- not run: a skeleton -->

```python
undefined_name
```

- In a list:

  ```python
  y = x + 1
  ```

  ```bash
  echo "not Python"
  ```

  ```text
  not an output: it follows a bash block
  ```

````markdown
```python
this is a block shown as an example, inside a longer fence
```
````
"""


def test_blocks_are_read_with_their_output_and_their_skip_comment() -> None:
    first, skipped, indented = blocks_of(SAMPLE)
    assert first == Block(line=3, code="x = 1\nprint(x)\n", output="1\n", run=True)
    assert skipped.run is False
    assert skipped.output is None
    assert indented == Block(line=20, code="y = x + 1\n", output=None, run=True)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("import numpy as np\n", True),
        ("from civiqhub_selene.blocks import VoteMatrix\n", True),
        ("from civiqhub_selene.ordinal import VoteCounts\n", False),
        ("import civiqhub_selene\n", False),
    ],
)
def test_a_page_needs_numpy_when_a_block_that_runs_names_it(code: str, expected: bool) -> None:
    assert needs_numpy([Block(1, code, None, True)]) is expected
    assert needs_numpy([Block(1, code, None, False)]) is False


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("from civiqhub_selene import viz\n", True),
        ("from civiqhub_selene.viz import draw_roles\n", True),
        ("import matplotlib.pyplot as plt\n", True),
        ("from civiqhub_selene.blocks import VoteMatrix\n", False),
    ],
)
def test_a_page_needs_viz_when_a_block_that_runs_names_it(code: str, expected: bool) -> None:
    assert needs_viz([Block(1, code, None, True)]) is expected
    assert needs_numpy([Block(1, "from civiqhub_selene import viz\n", None, True)]) is True
