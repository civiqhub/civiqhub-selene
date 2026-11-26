# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""The section "Mathematical model" of the site keeps one meaning per symbol.

``docs/model/notation.md`` defines every symbol of the section in its tables. These
tests read them: a symbol is defined once, and the Greek and calligraphic letters that
the other pages of the section write are all defined there.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "docs" / "model"
NOTATION = MODEL / "notation.md"

# The source distribution ships the tests without the pages of the site.
pytestmark = pytest.mark.skipif(not NOTATION.is_file(), reason="this checkout has no docs")

_ROW = re.compile(r"^\|(?P<symbol>[^|]*)\|(?P<rest>.*)\|\s*$")
_MATH = re.compile(r"\$\$(.+?)\$\$|\$([^$\n]+)\$", re.DOTALL)
_LETTER = re.compile(
    r"\\(?:var)?(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi"
    r"|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma"
    r"|Upsilon|Phi|Psi|Omega)(?![a-zA-Z])|\\mathcal\{(\w)\}"
)


def _symbols() -> list[str]:
    """Return each symbol the tables of the notation define, in TeX, as written."""
    symbols: list[str] = []
    for line in NOTATION.read_text(encoding="utf-8").splitlines():
        match = _ROW.match(line)
        if match is None or set(match["symbol"].strip()) <= {"-", " "}:
            continue
        cell = match["symbol"].strip()
        if cell == "Symbol":
            continue
        symbols.extend(group.strip() for group in re.findall(r"\$([^$]+)\$", cell))
    return symbols


def _letters(tex: str) -> set[str]:
    return {first or f"mathcal{{{second}}}" for first, second in _LETTER.findall(tex)}


def test_each_symbol_of_the_notation_is_defined_once() -> None:
    symbols = _symbols()
    assert len(symbols) > 30
    repeated = sorted({symbol for symbol in symbols if symbols.count(symbol) > 1})
    assert repeated == []


def test_each_letter_of_the_section_is_defined_in_the_notation() -> None:
    defined = set().union(*(_letters(symbol) for symbol in _symbols()))
    for page in sorted(MODEL.glob("*.md")):
        if page == NOTATION:
            continue
        text = page.read_text(encoding="utf-8")
        used = set().union(*(_letters(a or b) for a, b in _MATH.findall(text)))
        assert used <= defined, f"{page.name} writes letters the notation does not define"


_ROLE = re.compile(r"\{py:(\w+)\}`([^`]*)`")


def test_each_role_of_the_section_names_one_object() -> None:
    # A role closed by a stray brace still builds, as text: catch it here.
    for page in sorted(MODEL.glob("*.md")):
        text = page.read_text(encoding="utf-8")
        for kind, target in _ROLE.findall(text):
            assert re.fullmatch(r"~?[A-Za-z_][\w.]*", target), (
                f"{page.name}: the role {{py:{kind}}} has the target {target!r}"
            )
