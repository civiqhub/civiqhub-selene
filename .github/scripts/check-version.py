# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Fail unless the release tag, the package version, CITATION.cff and CHANGELOG.md agree.

Usage: python .github/scripts/check-version.py v0.1.0
"""

import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def read_version(path: str, pattern: str) -> str:
    match = re.search(pattern, (ROOT / path).read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        sys.exit(f"::error::No version found in {path}")
    return match.group(1)


tag = sys.argv[1]
if not tag.startswith("v"):
    sys.exit(f"::error::Release tag {tag} does not start with 'v'.")

expected = tag[1:]
pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
versions = {
    "pyproject.toml": str(pyproject["project"]["version"]),
    "src/civiqhub_selene/__init__.py": read_version(
        "src/civiqhub_selene/__init__.py", r'^__version__ = "([^"]+)"$'
    ),
    "CITATION.cff": read_version("CITATION.cff", r'^version:\s*"?([^"\s]+)"?\s*$'),
    "CHANGELOG.md": read_version("CHANGELOG.md", r"^## \[(\d[^\]]*)\]"),
}

status = 0
for path, version in versions.items():
    if version != expected:
        print(f"::error::{path} says {version}, but the release tag is {tag}.")
        status = 1

if status == 0:
    print(f"Release tag {tag} matches the package, citation and changelog versions.")
sys.exit(status)
