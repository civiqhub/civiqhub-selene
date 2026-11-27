# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""The published site: which versions it holds, in which order, and what lists them.

These tests cover the decisions of ``.github/scripts/build-docs-site.py``. The build of
each version (a clone, an environment, Sphinx) is exercised by the ``documentation`` job
of the CI, which runs the script itself. They also cover
``.github/scripts/check-no-external-resources.py``, which that job runs on the site it
builds, on small pages written here.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github" / "scripts" / "build-docs-site.py"
CHECK = ROOT / ".github" / "scripts" / "check-no-external-resources.py"

# The source distribution ships the tests without the scripts of the CI.
pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason="this checkout has no CI scripts")


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up by name while the class is being created.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


@pytest.fixture(scope="module")
def site() -> ModuleType:
    return _load("build_docs_site", SCRIPT)


@pytest.fixture(scope="module")
def checker() -> ModuleType:
    return _load("check_no_external_resources", CHECK)


def test_the_site_holds_the_latest_release_and_the_development_branch_only(
    site: ModuleType,
) -> None:
    versions = site.select_versions(
        site.Source("origin/develop", "0.3.0.dev0"),
        [
            site.Source("origin/main", "0.2.0"),
            site.Source("v0.2.0", "0.2.0"),
            site.Source("v0.10.0", "0.10.0"),
            site.Source("v0.1.0", "0.1.0"),
        ],
    )
    assert [version.slug for version in versions] == ["0.10.0", "dev"]
    assert [version.label for version in versions] == [
        "0.10.0 (latest release)",
        "dev (development branch)",
    ]
    assert [version.channel for version in versions] == ["release", "dev"]
    assert [version.path for version in versions] == ["0.10.0/", "dev/"]
    assert len(json.loads(site.versions_document(versions))["versions"]) == 2


def test_the_release_branch_wins_over_the_tag_of_the_same_version(site: ModuleType) -> None:
    versions = site.select_versions(
        None, [site.Source("origin/main", "0.1.0"), site.Source("v0.1.0", "0.1.0")]
    )
    assert [(version.slug, version.ref) for version in versions] == [("0.1.0", "origin/main")]


@pytest.mark.parametrize("version", ["0.1.0.dev0", "0.1.0rc1", "1.0", "v0.1.0", ""])
def test_a_version_that_is_not_a_release_number_is_left_out(site: ModuleType, version: str) -> None:
    assert site.select_versions(None, [site.Source("origin/main", version)]) == []


def test_before_the_first_release_the_site_holds_the_development_branch_alone(
    site: ModuleType,
) -> None:
    versions = site.select_versions(site.Source("origin/develop", "0.1.0.dev0"), [])
    assert [version.slug for version in versions] == ["dev"]
    assert json.loads(site.versions_document(versions))["default"] == "dev"


def test_the_versions_document_lists_each_folder_and_names_the_default(site: ModuleType) -> None:
    versions = site.select_versions(
        site.Source("origin/develop", "0.2.0.dev0"), [site.Source("origin/main", "0.1.0")]
    )
    document = site.versions_document(versions)
    assert document.endswith("\n")
    assert json.loads(document) == {
        "default": "0.1.0",
        "versions": [
            {"slug": "0.1.0", "label": "0.1.0 (latest release)", "path": "0.1.0/"},
            {"slug": "dev", "label": "dev (development branch)", "path": "dev/"},
        ],
    }


def test_an_earlier_release_sends_to_the_same_page_of_the_latest(site: ModuleType) -> None:
    (version,) = site.select_versions(None, [site.Source("origin/main", "0.2.0")])
    page = site.retired_version_page(version, "https://example.org/docs/")
    assert "0.2.0/" in page
    assert "window.location.replace" in page
    assert '<a href="https://example.org/docs/0.2.0/">' in page
    assert site.retired_version_page(version, "https://example.org/docs") == page
    assert '<a href="0.2.0/">' in site.retired_version_page(version, "")


def test_the_home_page_sends_to_the_default_version(site: ModuleType) -> None:
    (version,) = site.select_versions(None, [site.Source("origin/main", "0.1.0")])
    page = site.redirection_page(version, "https://example.org/docs/")
    assert '<meta http-equiv="refresh" content="0; url=0.1.0/">' in page
    assert '<link rel="canonical" href="https://example.org/docs/0.1.0/">' in page
    assert '<a href="0.1.0/">' in page
    assert "canonical" not in site.redirection_page(version, "")


def test_a_reference_without_documentation_is_left_out(site: ModuleType) -> None:
    # The first release holds no docs/conf.py; a name git does not know holds nothing.
    assert site.source_at("v0.0.1") is None
    assert site.source_at("a-reference-that-does-not-exist") is None


def test_the_workflows_run_the_script_that_is_tested_here() -> None:
    workflows = ROOT / ".github" / "workflows"
    command = "python .github/scripts/build-docs-site.py"
    assert command in (workflows / "docs.yml").read_text(encoding="utf-8")
    assert command in (workflows / "ci.yml").read_text(encoding="utf-8")


def test_the_reference_is_built_with_the_extras_each_commit_declares(
    site: ModuleType, tmp_path: Path
) -> None:
    assert site.documented_extras(ROOT) == ".[blocks,viz]"
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\n[project.optional-dependencies]\nblocks = ["numpy"]\n',
        encoding="utf-8",
    )
    assert site.documented_extras(tmp_path) == ".[blocks]"
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")
    assert site.documented_extras(tmp_path) == "."


def test_the_build_runs_sphinx_without_network_only_where_the_system_allows_it(
    site: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(["unshare"], 1)

    def missing(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise FileNotFoundError("unshare")

    def allowed(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(["unshare"], 0)

    monkeypatch.setattr(site.subprocess, "run", refused)
    assert site.network_cut() == []
    monkeypatch.setattr(site.subprocess, "run", missing)
    assert site.network_cut() == []
    monkeypatch.setattr(site.subprocess, "run", allowed)
    assert site.network_cut() == ["unshare", "--user", "--map-root-user", "--net"]


# -- No resource from another server -----------------------------------------------------

CLEAN_PAGE = """<!doctype html>
<html><head>
<link rel="canonical" href="https://selene.civiqhub.com/dev/page.html">
<meta property="og:url" content="https://selene.civiqhub.com/dev/page.html">
<meta http-equiv="refresh" content="0; url=dev/">
<link rel="stylesheet" href="_static/katex/katex.min.css">
<script defer src="_static/katex/katex.min.js"></script>
<script type="application/json" id="selene-katex-macros">{}</script>
<script type="module">import mermaid
from "https://cdn.jsdelivr.net/npm/mermaid@11.12.1/dist/mermaid.esm.min.mjs";
const style = "opacity: 1 !important";</script>
<style>@font-face { src: url(fonts/a.woff2) format("woff2"); }</style>
</head><body>
<a href="https://github.com/civiqhub/civiqhub-selene">Source</a>
<img src="_static/logo.svg" srcset="_static/logo.svg 1x, data:image/png;base64,AAAA 2x">
<span style="background: url('data:image/svg+xml;utf8,<svg/>')">x</span>
</body></html>
"""


def test_a_page_that_loads_nothing_from_another_server_passes(checker: ModuleType) -> None:
    findings, allowed = checker.check_html("dev/page.html", CLEAN_PAGE)
    assert findings == []
    # Mermaid is the one exception, and it is reported.
    assert [str(finding) for finding in allowed] == [
        "dev/page.html: <script> import "
        "https://cdn.jsdelivr.net/npm/mermaid@11.12.1/dist/mermaid.esm.min.mjs"
    ]


@pytest.mark.parametrize(
    ("markup", "where", "address"),
    [
        ('<script src="https://cdn.jsdelivr.net/npm/x.js"></script>', "<script src>", "x.js"),
        (
            '<link rel="stylesheet" href="//fonts.example.org/a.css">',
            "<link rel=stylesheet>",
            "a.css",
        ),
        (
            '<link rel="preload" href="https://example.org/f.woff2">',
            "<link rel=preload>",
            "f.woff2",
        ),
        ('<iframe src="https://example.org/embed"></iframe>', "<iframe src>", "embed"),
        ('<img srcset="a.png 1x, https://example.org/b.png 2x">', "<img srcset>", "b.png"),
        (
            '<script type="module">import("https://example.org/m.mjs")</script>',
            "<script> import",
            "m.mjs",
        ),
        ("<style>@import 'https://example.org/s.css';</style>", "<style> @import", "s.css"),
        (
            '<div style="background:url(https://example.org/i.png)"></div>',
            "<div style> url()",
            "i.png",
        ),
        (
            '<link rel="canonical" href="https://example.org/page.html">',
            "<link rel=canonical>",
            "page",
        ),
        (
            '<meta property="og:image" content="https://example.org/i.png">',
            "<meta og:image>",
            "i.png",
        ),
    ],
)
def test_a_page_that_loads_from_another_server_fails_and_names_the_address(
    checker: ModuleType, markup: str, where: str, address: str
) -> None:
    findings, allowed = checker.check_html("dev/bad.html", f"<html><head>{markup}</head></html>")
    assert allowed == []
    assert len(findings) == 1
    assert findings[0].path == "dev/bad.html"
    assert findings[0].where == where
    assert address in findings[0].address
    assert str(findings[0]).startswith(f"dev/bad.html: {where} ")


@pytest.mark.parametrize(
    "address",
    [
        "https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.esm.min.mjs",
        "https://cdn.jsdelivr.net/npm/@mermaid-js/layout-elk@0.1.4/dist/"
        "mermaid-layout-elk.esm.min.mjs",
        "https://cdn.jsdelivr.net/npm/mermaid@11.12.1/dist/mermaid.esm.min.mjs?x=1",
    ],
)
def test_the_mermaid_exception_covers_its_pinned_main_module_only(
    checker: ModuleType, address: str
) -> None:
    findings, allowed = checker.check_html(
        "a.html", f'<script type="module">import m from "{address}";</script>'
    )
    assert allowed == []
    assert [finding.address for finding in findings] == [address]


def test_a_style_sheet_that_loads_from_another_server_fails(checker: ModuleType) -> None:
    css = (
        "/* url(https://example.org/in-a-comment.png) */\n"
        '@import url("https://example.org/a.css");\n'
        "@font-face { src: url(fonts/a.woff2), url( 'https://example.org/b.woff2' ); }\n"
        "body { background: url(data:image/png;base64,AAAA); }\n"
    )
    findings = checker.check_css("_static/x.css", css)
    assert [(finding.where, finding.address) for finding in findings] == [
        ("url()", "https://example.org/a.css"),
        ("url()", "https://example.org/b.woff2"),
    ]


def test_the_script_reads_a_folder_and_exits_with_1_on_a_finding(
    checker: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "dev" / "_static").mkdir(parents=True)
    (tmp_path / "dev" / "page.html").write_text(CLEAN_PAGE, encoding="utf-8")
    (tmp_path / "dev" / "_static" / "site.css").write_text("a { color: red; }", encoding="utf-8")
    assert checker.main([str(tmp_path)]) == 0
    assert "No resource from another server in 2 HTML and CSS file(s)." in capsys.readouterr().out

    bad = '<html><head><script src="https://cdn.jsdelivr.net/npm/x.js"></script></head></html>'
    (tmp_path / "dev" / "bad.html").write_text(bad, encoding="utf-8")
    assert checker.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "dev/bad.html: <script src> https://cdn.jsdelivr.net/npm/x.js" in out
    assert "1 reference(s) to another server in 3 file(s)." in out
    assert checker.main([str(tmp_path / "missing")]) == 2


def test_the_documentation_job_checks_the_site_it_builds() -> None:
    job = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    build = "python .github/scripts/build-docs-site.py --output _site --development HEAD --offline"
    check = "python .github/scripts/check-no-external-resources.py _site"
    assert build in job
    assert check in job
    assert job.index(build) < job.index(check)
