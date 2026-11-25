# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

r"""Render the formulas of the site with the copy of KaTeX that the site ships.

Sphinx writes each formula into the page as TeX, between the delimiters that the
auto-render script of KaTeX reads: ``\( ... \)`` inline, ``\[ ... \]`` on its own lines.
In the browser, ``_static/katex-render.js`` hands each formula to auto-render, one
element at a time: a dollar sign or a delimiter elsewhere in the page is never read as
TeX. Formulas written in Markdown (``$ ... $``, ``$$ ... $$``, amsmath environments) and
in docstrings (the ``:math:`` role and the ``.. math::`` directive) take the same way.

KaTeX is read from ``_static/katex/`` (see its ``README.md``), on the pages that hold a
formula only, and from no other server. The macros of ``katex_macros`` (``conf.py``)
are written into those pages and shared by every formula.

Equations are numbered by Sphinx, so that ``{eq}`` references and their numbers agree.
KaTeX numbers some environments by itself (``align``, ``equation``, ``gather``);
``_static/katex-theme.css`` hides those numbers, and a numbered environment shows the
number Sphinx gives it.

KaTeX reads the formulas in the browser: a formula that KaTeX cannot parse does not fail
the build, it shows as an error in the page.
"""

import json
from typing import Any, cast

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.config import Config
from sphinx.errors import ConfigError
from sphinx.locale import _
from sphinx.util.math import get_node_equation_number
from sphinx.writers.html5 import HTML5Translator

NAME = "katex_local"

#: Commands that make KaTeX link to, or load, a resource. KaTeX refuses them as long as
#: ``trust`` is false (``katex-render.js``); a macro may not name them either.
UNTRUSTED_COMMANDS = (
    r"\href",
    r"\url",
    r"\includegraphics",
    r"\htmlClass",
    r"\htmlId",
    r"\htmlStyle",
    r"\htmlData",
)


def visit_inline_math(self: HTML5Translator, node: nodes.math) -> None:
    """Write an inline formula: ``\\( TeX \\)`` in a ``span``."""
    self.body.append(self.starttag(node, "span", "", CLASS="math notranslate nohighlight"))
    self.body.append(rf"\({self.encode(node.astext())}\)</span>")
    raise nodes.SkipNode


def display_tex(node: nodes.math_block) -> str:
    """Return the TeX of a formula on its own lines, as KaTeX reads it in display mode.

    A formula written without wrapping (an amsmath environment, or ``:nowrap:``) is read
    as it is. Otherwise, as Sphinx does for MathJax, an empty line separates equations,
    which are aligned on their ``&``, and an equation broken by ``\\\\`` is split.
    """
    text = str(node.astext())
    if node.get("no-wrap", node.get("nowrap", False)):
        return text
    parts = [part for part in text.split("\n\n") if part.strip()]
    parts = [rf"\begin{{split}}{part}\end{{split}}" if r"\\" in part else part for part in parts]
    if len(parts) > 1:
        return r"\begin{aligned}" + r"\\".join(parts) + r"\end{aligned}"
    return parts[0] if parts else ""


def visit_display_math(self: HTML5Translator, node: nodes.math_block) -> None:
    """Write a formula on its own lines: its number, then ``\\[ TeX \\]`` in a ``div``."""
    self.body.append(self.starttag(node, "div", CLASS="math notranslate nohighlight"))
    if node["number"]:
        number = get_node_equation_number(self, node)
        self.body.append(f'<span class="eqno">({number})')
        self.add_permalink_ref(node, _("Link to this equation"))
        self.body.append("</span>")
    self.body.append(rf"\[{self.encode(display_tex(node))}\]</div>" + "\n")
    raise nodes.SkipNode


def check_macros(app: Sphinx, config: Config) -> None:
    """Refuse a macro that is not a TeX command, or that would link to a resource."""
    macros = config.katex_macros
    for name, body in macros.items():
        if not (isinstance(name, str) and name.startswith("\\") and isinstance(body, str)):
            raise ConfigError(f"katex_macros: {name!r} must map a TeX command to a string")
        if any(command in body for command in UNTRUSTED_COMMANDS):
            raise ConfigError(f"katex_macros: {name} links to a resource: {body!r}")


def macros_document(macros: dict[str, str]) -> str:
    """Return the macros as JSON that can stand inside a ``script`` element."""
    return json.dumps(macros, sort_keys=True).replace("</", "<\\/")


def add_katex(
    app: Sphinx, pagename: str, templatename: str, context: dict[str, Any], doctree: Any
) -> None:
    """Add KaTeX to a page that holds a formula, and to no other page."""
    if app.builder.format != "html":
        return
    builder = cast(StandaloneHTMLBuilder, app.builder)
    if builder.math_renderer_name != NAME:
        return
    if app.registry.html_assets_policy != "always" and not context.get("has_maths_elements"):
        return
    builder.add_css_file("katex/katex.min.css")
    builder.add_css_file("katex-theme.css")
    builder.add_js_file(
        "",
        type="application/json",
        id="selene-katex-macros",
        body=macros_document(app.config.katex_macros),
    )
    builder.add_js_file("katex/katex.min.js", defer="defer")
    builder.add_js_file("katex/contrib/auto-render.min.js", defer="defer")
    builder.add_js_file("katex-render.js", defer="defer")


def setup(app: Sphinx) -> dict[str, bool]:
    app.add_html_math_renderer(
        NAME,
        inline_renderers=(visit_inline_math, None),
        block_renderers=(visit_display_math, None),
    )
    app.add_config_value("katex_macros", {}, "html", types=frozenset({dict}))
    app.connect("config-inited", check_macros)
    app.connect("html-page-context", add_katex)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
