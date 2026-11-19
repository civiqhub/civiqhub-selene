# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Sphinx configuration of the documentation site.

How the site is organised and how to build it: see ``docs/README.md``.
"""

import os
import sys
from pathlib import Path

import civiqhub_selene

# The local extension lives next to this file.
sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "civiqhub-selene"
author = "CiviqHub SAS"
project_copyright = "2026 CiviqHub SAS"
# The site documents the package that is installed in the environment of the build.
release = civiqhub_selene.__version__
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.mermaid",
    "selene_docs",
]

# docs/README.md is read on GitHub; it explains the site and is not a page of it.
exclude_patterns = ["_build", "_ext", "_templates", "README.md"]

# -- Pages (Markdown, read by MyST) ---------------------------------------------------

source_suffix = {".md": "markdown"}
myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "dollarmath"]
# Headings get the anchors GitHub gives them, so that "page.md#a-heading" works in both.
myst_heading_anchors = 3
# A ```mermaid fence is a diagram, on the site as on GitHub.
myst_fence_as_directive = ["mermaid"]

# -- Reference (generated from the docstrings) ----------------------------------------

autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_preserve_defaults = True
python_use_unqualified_type_names = True
napoleon_use_rtype = False

# -- Checks ---------------------------------------------------------------------------

# Every reference to a name of the engine must resolve; the build runs with warnings as
# errors. The site is built without network access, so it does not link to the
# documentation of other packages: their names are declared here, by top-level package.
nitpicky = True
nitpick_ignore_regex = [
    ("py:.*", r"(abc|collections|enum|fractions|numpy)\..*"),
    # Private modules, private module-level names and type variables are not documented.
    ("py:.*", r"civiqhub_selene(\.[a-z]\w*)*\._\w+(\..*)?"),
    ("py:.*", r"civiqhub_selene\..+\.\w*T(_co|_contra)?"),
]

# -- HTML -----------------------------------------------------------------------------

# The published site holds one folder per version, built by
# .github/scripts/build-docs-site.py, which sets SELENE_DOCS_CHANNEL to "dev" for the
# development branch and to "release" for a release.
is_development = os.environ.get("SELENE_DOCS_CHANNEL", "") == "dev" or "dev" in release

html_theme = "furo"
html_title = f"SELENE {release}"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["version-switcher.css"]
html_js_files = ["version-switcher.js"]
# The sidebar of the theme, with the version selector under the name of the site.
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/version-switcher.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
        "sidebar/variant-selector.html",
    ]
}
html_theme_options = {
    "source_repository": "https://github.com/civiqhub/civiqhub-selene/",
    "source_branch": "develop" if is_development else "main",
    "source_directory": "docs/",
}
if is_development:
    html_theme_options["announcement"] = (
        f"This is the documentation of the development branch ({release}). "
        "It describes work that is not released yet."
    )
