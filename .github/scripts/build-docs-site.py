# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Build the documentation site with one folder per version.

Usage: python .github/scripts/build-docs-site.py --output _site --base-url https://example.org/

The site that is published holds:

- ``X.Y.Z/``: the documentation of the latest release, the one ``origin/main`` holds;
  earlier releases are not published (Decision of 2026-11-24: the site holds the
  latest release and the development branch, nothing else);
- ``dev/``: the documentation of the development branch, ``origin/develop``;
- ``versions.json``: the list that the version selector of every page reads;
- ``index.html``: a redirection to the latest release, or to ``dev`` before the first one;
- ``404.html``: a page GitHub Pages serves for a missing address; it sends a retired
  release to the same page of the latest one.

Each version is built from its own commit, in its own environment, with the tools that
this commit pins: a version is always rebuilt as it was. A commit without
``docs/conf.py`` has no documentation and is left out, and so is a reference that does
not exist in the clone.

How the site is organised: see ``docs/README.md``.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE = re.compile(r"\d+\.\d+\.\d+")
DEVELOPMENT_SLUG = "dev"


@dataclass(frozen=True)
class Source:
    """A commit that holds documentation.

    Attributes:
        ref: What git calls the commit: ``origin/main``, ``v0.1.0``.
        version: The version of the package at this commit, read from ``pyproject.toml``.
    """

    ref: str
    version: str


@dataclass(frozen=True)
class Version:
    """One folder of the site.

    Attributes:
        slug: Name of the folder: a release number, or ``dev``.
        label: What the version selector shows.
        ref: The commit the folder is built from.
        channel: ``release`` or ``dev``; the pages of ``dev`` carry a banner.
    """

    slug: str
    label: str
    ref: str
    channel: str

    @property
    def path(self) -> str:
        return f"{self.slug}/"


def select_versions(development: Source | None, releases: Sequence[Source]) -> list[Version]:
    """Return the folders of the site: the latest release, then ``dev``.

    Args:
        development: The development branch, when it holds documentation.
        releases: The commits that may hold a release, by decreasing priority: the
            release branch first, then the tags. When two commits carry the same
            version, the first one is kept. A version that is not ``X.Y.Z`` is not a
            release and is left out. Only the highest release is published.
    """
    kept: dict[tuple[int, ...], Source] = {}
    for source in releases:
        if RELEASE.fullmatch(source.version):
            kept.setdefault(tuple(int(part) for part in source.version.split(".")), source)
    versions = []
    if kept:
        source = kept[max(kept)]
        label = f"{source.version} (latest release)"
        versions.append(Version(source.version, label, source.ref, "release"))
    if development is not None:
        label = f"{DEVELOPMENT_SLUG} (development branch)"
        versions.append(Version(DEVELOPMENT_SLUG, label, development.ref, "dev"))
    return versions


def versions_document(versions: Sequence[Version]) -> str:
    """Return the content of ``versions.json``. The first version is the default one."""
    listed = [
        {"slug": version.slug, "label": version.label, "path": version.path} for version in versions
    ]
    return json.dumps({"default": versions[0].slug, "versions": listed}, indent=2) + "\n"


def retired_version_page(version: Version, base_url: str) -> str:
    """Return the GitHub Pages 404 page: send a retired release to the latest one."""
    root = base_url.rstrip("/") + "/" if base_url else ""
    href = f"{root}{version.path}"
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        "<title>This page does not exist</title>\n"
        "<script>\n"
        "const match = window.location.pathname.match(/^(.*?\\/)(\\d+\\.\\d+\\.\\d+)\\/(.*)$/);\n"
        f'if (match && match[2] !== "{version.slug}") {{\n'
        "  window.location.replace("
        f'match[1] + "{version.path}" + match[3] + window.location.hash'
        ");\n"
        "}\n"
        "</script>\n"
        "</head>\n"
        "<body>\n"
        "<p>This page does not exist. "
        f'See the <a href="{href}">latest release</a>.</p>\n'
        "</body>\n"
        "</html>\n"
    )


def redirection_page(version: Version, base_url: str) -> str:
    """Return the home page of the site: a redirection to its default version."""
    canonical = f'<link rel="canonical" href="{base_url}{version.path}">\n' if base_url else ""
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        "<title>SELENE documentation</title>\n"
        f'<meta http-equiv="refresh" content="0; url={version.path}">\n'
        f"{canonical}"
        "</head>\n"
        "<body>\n"
        f'<p>The documentation is at <a href="{version.path}">{version.path}</a>.</p>\n'
        "</body>\n"
        "</html>\n"
    )


def git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False
    )


def source_at(ref: str) -> Source | None:
    """Return the commit of a reference, or None when it holds no documentation."""
    if git("cat-file", "-e", f"{ref}:docs/conf.py").returncode != 0:
        return None
    project = tomllib.loads(git("show", f"{ref}:pyproject.toml").stdout)["project"]
    return Source(ref, str(project["version"]))


def discover(development_ref: str, main_ref: str) -> list[Version]:
    """Read the clone and return the folders to build."""
    tags = git("tag", "--list", "v*", "--sort=-version:refname").stdout.split()
    releases = [source for ref in [main_ref, *tags] if (source := source_at(ref)) is not None]
    return select_versions(source_at(development_ref), releases)


def run(command: Sequence[str | Path], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run([str(part) for part in command], cwd=cwd, env=env, check=True)


#: Runs a command in a network namespace of its own, with no interface but loopback.
NO_NETWORK = ("unshare", "--user", "--map-root-user", "--net")


def network_cut() -> list[str]:
    """Return the prefix that runs a command without network, or ``[]``.

    The prefix is ``[]`` when the system does not allow it: no ``unshare``, or user
    namespaces refused to this user.
    """
    try:
        probe = subprocess.run([*NO_NETWORK, "true"], capture_output=True, check=False)
    except OSError:
        return []
    return list(NO_NETWORK) if probe.returncode == 0 else []


def documented_extras(tree: Path) -> str:
    """Return the requirement of the package with the extras its reference imports.

    ``blocks`` exists since the first version with a site; ``viz`` since its
    subpackage was added. An older commit is installed with the extras it declares.
    """
    with (tree / "pyproject.toml").open("rb") as handle:
        declared = tomllib.load(handle)["project"].get("optional-dependencies", {})
    extras = [extra for extra in ("blocks", "viz") if extra in declared]
    return f".[{','.join(extras)}]" if extras else "."


def build(version: Version, output: Path, base_url: str, *, offline: Sequence[str] = ()) -> None:
    """Build one version from its commit, in a fresh environment, into its folder.

    ``offline`` prefixes the Sphinx command, so that the build runs without network once
    the package and the tools are installed (see ``network_cut``).
    """
    print(f"Building {version.slug} from {version.ref}", flush=True)
    with tempfile.TemporaryDirectory() as work:
        tree = Path(work) / "tree"
        environment = Path(work) / "environment"
        python = environment / ("Scripts" if os.name == "nt" else "bin") / "python"
        target = output / version.slug
        added = git("worktree", "add", "--detach", str(tree), version.ref)
        if added.returncode != 0:
            sys.exit(f"::error::Cannot check out {version.ref}: {added.stderr.strip()}")
        try:
            run([sys.executable, "-m", "venv", environment], cwd=tree)
            run([python, "-m", "pip", "install", "--quiet", "--upgrade", "pip"], cwd=tree)
            # The package and the tools are those of this commit, with the extras that
            # the reference documents.
            run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--quiet",
                    documented_extras(tree),
                    "--group",
                    "docs",
                ],
                cwd=tree,
            )
            address = ["-D", f"html_baseurl={base_url}{version.path}"] if base_url else []
            sphinx: list[str | Path] = [python, "-m", "sphinx", "-W", "-q", "-b", "html"]
            sphinx += [*address, "docs", target]
            run(
                [*offline, *sphinx],
                cwd=tree,
                env={**os.environ, "SELENE_DOCS_CHANNEL": version.channel},
            )
        finally:
            git("worktree", "remove", "--force", str(tree))


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="folder of the site, created")
    parser.add_argument("--base-url", default="", help="address the site is published at")
    parser.add_argument("--development", default="origin/develop", help="development branch")
    parser.add_argument("--main", default="origin/main", help="release branch")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="run Sphinx without network when the system allows it, and warn when it does not",
    )
    arguments = parser.parse_args()

    output: Path = arguments.output.resolve()
    base_url: str = arguments.base_url and arguments.base_url.rstrip("/") + "/"
    versions = discover(arguments.development, arguments.main)
    if not versions:
        sys.exit("::error::No commit holds documentation: nothing to build.")

    offline: list[str] = []
    if arguments.offline:
        offline = network_cut()
        if offline:
            print("Sphinx runs without network: " + " ".join(offline), flush=True)
        else:
            print(
                "::warning::Sphinx runs with network: this system refuses " + " ".join(NO_NETWORK)
            )

    output.mkdir(parents=True, exist_ok=True)
    for version in versions:
        build(version, output, base_url, offline=offline)
    (output / "versions.json").write_text(versions_document(versions), encoding="utf-8")
    (output / "index.html").write_text(redirection_page(versions[0], base_url), encoding="utf-8")
    if versions[0].channel == "release":
        (output / "404.html").write_text(
            retired_version_page(versions[0], base_url), encoding="utf-8"
        )
    print("Built: " + ", ".join(f"{version.slug} ({version.ref})" for version in versions))


if __name__ == "__main__":
    main()
