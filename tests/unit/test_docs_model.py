# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""The section "Mathematical model" of the site keeps one meaning per symbol.

``docs/model/notation.md`` defines every symbol of the section in its tables. These
tests read them: a symbol is defined once, and the Greek and calligraphic letters that
the other pages of the section write are all defined there. ``docs/model/api-to-formulas.md``
links each public name to its formulas: every exported name it must list is on it, and
every numbered equation of the section is cited there. The glossary holds the words of the
architecture, and the references every work an estimator names. The tests read the files
only and do not import the engine.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "docs" / "model"
NOTATION = MODEL / "notation.md"
API = MODEL / "api-to-formulas.md"
GLOSSARY = MODEL / "glossary.md"
REFERENCES = MODEL / "references.md"
ARCHITECTURE = ROOT / "docs" / "architecture.md"
PACKAGE = ROOT / "src" / "civiqhub_selene"

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


def _exports(module: str) -> list[str]:
    """Return the ``__all__`` of a module of the engine, read from its file."""
    path = PACKAGE.parent.joinpath(*module.split("."))
    path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            names = ast.literal_eval(node.value)
            assert isinstance(names, list)
            return [str(name) for name in names]
    raise AssertionError(f"{module} has no __all__")


def _listed(module: str) -> set[str]:
    """Return the names of ``module`` that a role of the page of the API names."""
    found: set[str] = set()
    for _, target in _ROLE.findall(API.read_text(encoding="utf-8")):
        dotted = target.lstrip("~")
        if dotted.startswith(f"{module}."):
            found.add(dotted.rsplit(".", 1)[1])
    return found


# Each name these modules export has a row; of the others, the names that carry a formula.
_LISTED_IN_FULL = [
    "civiqhub_selene.blocks",
    "civiqhub_selene.blocks.responses",
    "civiqhub_selene.blocks.results",
    "civiqhub_selene.blocks.spectral",
    "civiqhub_selene.blocks.synthetic",
    "civiqhub_selene.ordinal",
]
_LISTED_IN_PART = {
    "civiqhub_selene.viz": {"group_graph", "plot_mixedness"},
    "civiqhub_selene.blocks.results": {
        "objective",
        "integrated_likelihood",
        "log_likelihood",
        "group_memberships",
        "block_memberships",
        "role_by_block",
        "profile_probabilities",
        "mixedness",
    },
}


def test_the_exports_are_read_from_the_code() -> None:
    assert "LatentBlockModel" in _exports("civiqhub_selene.blocks")
    assert _exports("civiqhub_selene.ordinal") == ["SCORES", "VoteCounts"]


@pytest.mark.parametrize("module", _LISTED_IN_FULL)
def test_each_exported_name_has_a_row_on_the_page_of_the_api(module: str) -> None:
    missing = sorted(set(_exports(module)) - _listed(module))
    assert missing == [], f"docs/model/api-to-formulas.md has no row for {missing}"


@pytest.mark.parametrize("module", sorted(_LISTED_IN_PART))
def test_the_names_that_carry_a_formula_have_a_row(module: str) -> None:
    missing = sorted(_LISTED_IN_PART[module] - _listed(module))
    assert missing == [], f"docs/model/api-to-formulas.md has no row for {missing}"


def test_each_numbered_equation_of_the_section_is_cited_on_the_page_of_the_api() -> None:
    labels: set[str] = set()
    for page in sorted(MODEL.glob("*.md")):
        if page.name != "formula-test.md":
            labels.update(
                re.findall(r"^\$\$ \((eq-[\w-]+)\)", page.read_text(encoding="utf-8"), re.M)
            )
    assert len(labels) > 20
    cited = set(re.findall(r"\{eq\}`(eq-[\w-]+)`", API.read_text(encoding="utf-8")))
    assert sorted(labels - cited) == []


def _first_cells(path: Path, heading: str) -> list[str]:
    """Return the first cell of each row of the table under ``heading`` in a page."""
    text = path.read_text(encoding="utf-8").split(f"\n{heading}\n", 1)[1]
    cells = []
    for line in text.split("\n## ", 1)[0].splitlines():
        match = _ROW.match(line)
        if match is not None and not set(match["symbol"].strip()) <= {"-", " "}:
            cells.append(match["symbol"].strip())
    return cells[1:]


def test_the_glossary_holds_each_word_of_the_architecture() -> None:
    words = {w for cell in _first_cells(ARCHITECTURE, "## Words") for w in cell.split(", ")}
    assert {"score", "co-cluster", "fingerprint"} <= words
    glossary = {w for cell in _first_cells(GLOSSARY, "## Words") for w in cell.split(", ")}
    assert sorted(words - glossary) == []


def _references_of_the_estimators() -> list[tuple[str, str]]:
    """Return the first author and the year of each work a ``reference`` attribute names."""
    works = []
    for path in sorted((PACKAGE / "blocks").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "reference" for t in node.targets)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                for part in node.value.value.split(";"):
                    match = re.match(r"\s*([A-Z][a-z]+)\b.*\b(\d{4})\s*$", part)
                    assert match is not None, f"{path.name}: unexpected reference {part!r}"
                    works.append((match[1], match[2]))
    return works


def test_each_work_an_estimator_names_has_an_entry_in_the_references() -> None:
    works = _references_of_the_estimators()
    assert ("Govaert", "2003") in works
    assert ("Airoldi", "2008") in works
    headings = re.findall(r"^### (.+)$", REFERENCES.read_text(encoding="utf-8"), re.M)
    for author, year in works:
        assert any(h.startswith(author) and h.endswith(f"({year})") for h in headings), (
            f"docs/model/references.md has no entry for {author} ({year})"
        )
