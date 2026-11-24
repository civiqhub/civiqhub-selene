# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Sphinx configuration of the documentation site.

How the site is organised and how to build it: see ``docs/README.md``.
"""

import os
import shutil
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
    "sphinxext.opengraph",
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
    ("py:.*", r"(abc|collections|enum|fractions|matplotlib|networkx|numpy)\..*"),
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
html_favicon = "_static/civiqhub-symbol-light.svg"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["version-switcher.css", "civiqhub.css"]
# theme-migrate.js is also loaded in the head from _templates/page.html, so that
# Furo reads localStorage after the one-time migration of the former light lock.
html_js_files = [
    ("theme-migrate.js", {"priority": 200}),
    "search-shortcut.js",
    "sidebar-wrap.js",
    "version-switcher.js",
]
# Dark highlighting: carbon palette, no blue hue. Style lives in docs/_ext/.
pygments_dark_style = "civiqhub_carbon.CiviqHubCarbonDark"
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
    "light_logo": "civiqhub-logo-light.svg",
    "dark_logo": "civiqhub-logo-dark.svg",
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#2C448B",
        "color-brand-content": "#2C448B",
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#FAFAFA",
        "color-foreground-primary": "#020617",
        "color-foreground-secondary": "#64748b",
        "color-background-border": "#e2e8f0",
        "color-code-background": "#f1f5f9",
        "color-sidebar-background": "#FAFAFA",
        "color-sidebar-background-border": "#e2e8f0",
        "color-sidebar-brand-text": "#020617",
        "color-sidebar-caption-text": "#020617",
        "color-sidebar-link-text": "#64748b",
        "color-sidebar-link-text--top-level": "#64748b",
        "color-sidebar-item-background--current": "#f1f5f9",
        "color-sidebar-item-background--hover": "#f1f5f9",
        "color-sidebar-search-background": "#ffffff",
        "color-sidebar-search-border": "#e2e8f0",
        "color-sidebar-search-text": "#020617",
        "color-sidebar-search-foreground": "#64748b",
        "font-stack": (
            '"Bricolage Grotesque", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif'
        ),
        "font-stack--headings": (
            '"Bricolage Grotesque", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif'
        ),
    },
    "dark_css_variables": {
        "color-background-primary": "#171717",
        "color-background-secondary": "#000000",
        "color-background-hover": "#404040",
        "color-background-hover--transparent": "#40404000",
        "color-background-border": "#404040",
        "color-background-item": "#404040",
        "color-foreground-primary": "#FAFAFA",
        "color-foreground-secondary": "#e5e5e5",
        "color-foreground-muted": "#868686",
        "color-foreground-border": "#868686",
        "color-code-background": "#000000",
        "color-sidebar-background": "#000000",
        "color-sidebar-background-border": "#404040",
        "color-sidebar-brand-text": "#FAFAFA",
        "color-sidebar-caption-text": "#FAFAFA",
        "color-sidebar-link-text": "#FAFAFA",
        "color-sidebar-link-text--top-level": "#FAFAFA",
        "color-sidebar-item-background--current": "#404040",
        "color-sidebar-item-background--hover": "#404040",
        "color-sidebar-search-background": "#000000",
        "color-sidebar-search-border": "#404040",
        "color-sidebar-search-text": "#FAFAFA",
        "color-sidebar-search-foreground": "#FAFAFA",
        "color-admonition-title": "#868686",
        "color-admonition-title-background": "rgba(134,134,134,0.2)",
        "color-topic-title": "#868686",
        "color-topic-title-background": "rgba(134,134,134,0.2)",
        "font-stack": (
            '"Bricolage Grotesque", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif'
        ),
        "font-stack--headings": (
            '"Bricolage Grotesque", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif'
        ),
        "color-card-background": "#000000",
        "color-card-border": "#404040",
        "color-card-marginals-background": "#404040",
        "color-admonition-background": "#000000",
        "color-announcement-background": "#000000",
        "color-announcement-text": "#FAFAFA",
        "color-inline-code-background": "#404040",
        "color-brand-primary": "#FAFAFA",
        "color-brand-content": "#FAFAFA",
        "color-brand-visited": "#FAFAFA",
        "color-link": "#FAFAFA",
        "color-link--hover": "#FAFAFA",
        "color-link--visited": "#FAFAFA",
        "color-link-underline": "#868686",
        "color-link-underline--hover": "#FAFAFA",
        "color-highlighted-background": "#404040",
        "color-guilabel-background": "#404040",
        "color-guilabel-border": "#868686",
        "color-api-changed": "#e5e5e5",
        "color-api-changed-border": "#868686",
        "color-admonition-title--note": "#868686",
        "color-admonition-title-background--note": "rgba(134,134,134,0.2)",
        "color-admonition-title--seealso": "#868686",
        "color-admonition-title-background--seealso": "rgba(134,134,134,0.2)",
        "color-admonition-title--important": "#e5e5e5",
        "color-admonition-title-background--important": "rgba(229,229,229,0.2)",
        "color-admonition-title--warning": "#F2A202",
        "color-admonition-title-background--warning": "rgba(242,162,2,0.2)",
        "color-admonition-title--caution": "#F2A202",
        "color-admonition-title-background--caution": "rgba(242,162,2,0.2)",
        "color-admonition-title--danger": "#ff2f1c",
        "color-admonition-title-background--danger": "rgba(255,47,28,0.2)",
        "color-admonition-title--attention": "#ff2f1c",
        "color-admonition-title-background--attention": "rgba(255,47,28,0.2)",
        "color-admonition-title--error": "#ff2f1c",
        "color-admonition-title-background--error": "rgba(255,47,28,0.2)",
        "color-admonition-title--tip": "#73b876",
        "color-admonition-title-background--tip": "rgba(115,184,118,0.2)",
        "color-admonition-title--hint": "#73b876",
        "color-admonition-title-background--hint": "rgba(115,184,118,0.2)",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/civiqhub/civiqhub-selene",
            "html": Path(__file__)
            .with_name("_static")
            .joinpath("github.svg")
            .read_text(encoding="utf-8"),
            "class": "",
        },
    ],
}
if is_development:
    html_theme_options["announcement"] = (
        f"This is the documentation of the development branch ({release}). "
        "It describes work that is not released yet."
    )

# Open Graph. The publication script may override html_baseurl with -D.
# Social cards stay off: the build has no network and must not pull matplotlib.
# The public origin is selene.civiqhub.com (from 21/11/2026). The docs workflow
# still passes the former GitHub Pages address; rewrite it so og:url and
# canonical tags stay on the public domain.
_PUBLIC_DOCS_ORIGIN = "https://selene.civiqhub.com/"
_LEGACY_PAGES_ORIGIN = "https://civiqhub.github.io/civiqhub-selene/"
ogp_site_url = _PUBLIC_DOCS_ORIGIN
ogp_site_name = "SELENE"
ogp_social_cards = {"enable": False}


def setup(app):
    """Bind Open Graph URLs and keep review screenshots out of the published site."""

    def _bind_opengraph(app, config):
        base = config.html_baseurl or config.ogp_site_url or _PUBLIC_DOCS_ORIGIN
        if base.startswith(_LEGACY_PAGES_ORIGIN):
            base = _PUBLIC_DOCS_ORIGIN + base[len(_LEGACY_PAGES_ORIGIN) :]
        config.html_baseurl = base
        config.ogp_site_url = base

    def _drop_review_screenshots(app, exception):
        if exception is not None:
            return
        shots = Path(app.outdir) / "_static" / "screenshots"
        if shots.is_dir():
            shutil.rmtree(shots)

    app.connect("config-inited", _bind_opengraph)
    app.connect("build-finished", _drop_review_screenshots)
