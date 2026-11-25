# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Check that no page of the built site loads a resource from another server.

Usage: python .github/scripts/check-no-external-resources.py _site

The site ships every script, style sheet, font and image it uses (``docs/_static/``).
This script reads every HTML and CSS file under a folder and fails when one of them
refers to another server for something a browser loads with the page:

- in HTML: ``<script src>``, the ``import`` of an inline ``<script>``, a ``<link>`` that
  loads (``stylesheet``, ``preload``, ``modulepreload``, ``icon``, ...), ``<iframe>``,
  ``<img>``, ``<source>``, ``<video>``, ``<audio>``, ``<track>``, ``<embed>``,
  ``<object>``, the SVG ``<image>`` and ``<use>``, and the CSS of ``<style>`` and of
  ``style`` attributes;
- in CSS: ``@import`` and ``url()`` (fonts, images).

Links a reader follows (``<a href>``) are allowed. So are the metadata that name the
public address of the site: the canonical URL and the Open Graph tags, on
``PUBLIC_ORIGIN``. ``EXCEPTIONS`` lists the one resource that is still loaded from
another server, with the decision that allows it.

An address is external when it is absolute (``https://host/...``) or relative to the
scheme (``//host/...``). ``data:`` addresses and relative paths stay on the site.

The script exits with 0 when every file is clean, with 1 when it finds a reference, and
names each file and address. It uses the standard library only.
"""

import argparse
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

#: The public address of the site: its canonical URL and its Open Graph tags name it.
PUBLIC_ORIGIN = "https://selene.civiqhub.com/"

#: Resources still loaded from another server, each with the decision that allows it.
#: Mermaid, which draws the diagrams, comes from jsDelivr: decision of Yanis of
#: 25/11/2026 (SEL-11.4; embedding it, SEL-11.6, is not retained). Its main module only,
#: at a pinned version: no other package, no "latest".
EXCEPTIONS = {
    re.compile(
        r"https://cdn\.jsdelivr\.net/npm/mermaid@\d+\.\d+\.\d+/dist/mermaid\.esm\.min\.mjs"
    ): "Mermaid, decision of Yanis of 25/11/2026",
}

#: ``<link rel>`` values that only name a related page: nothing is loaded.
NAVIGATION_RELS = frozenset(
    {
        "alternate",
        "author",
        "bookmark",
        "canonical",
        "first",
        "help",
        "index",
        "last",
        "license",
        "next",
        "prev",
        "search",
        "top",
        "up",
    }
)

#: Attributes that load a resource, by element.
LOADING_ATTRIBUTES = {
    "audio": ("src",),
    "embed": ("src",),
    "iframe": ("src",),
    "image": ("href", "xlink:href"),
    "img": ("src", "srcset"),
    "input": ("src",),
    "object": ("data",),
    "script": ("src",),
    "source": ("src", "srcset"),
    "track": ("src",),
    "use": ("href", "xlink:href"),
    "video": ("src", "poster"),
}

_EXTERNAL = re.compile(r"^\s*(?:[a-z][a-z0-9+.-]*:)?//", re.IGNORECASE)
_LOCAL_SCHEMES = re.compile(r"^\s*(?:data|blob|about|javascript):", re.IGNORECASE)
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_CSS_URL = re.compile(r"""url\(\s*(?:"([^"]*)"|'([^']*)'|([^)\s]*))\s*\)""", re.IGNORECASE)
_CSS_IMPORT = re.compile(r"""@import\s+(?:"([^"]*)"|'([^']*)')""", re.IGNORECASE)
_JS_IMPORT = re.compile(
    r"""\bimport\s*(?:\(\s*|[\w*{}\s,$]+\s+from\s*)?(?:"([^"]+)"|'([^']+)'|`([^`$]+)`)"""
)


@dataclass(frozen=True)
class Finding:
    """A reference to another server, in a file of the site.

    Attributes:
        path: The file, relative to the folder that was checked.
        where: What refers to the address, such as ``<script src>`` or ``url()``.
        address: The address.
    """

    path: str
    where: str
    address: str

    def __str__(self) -> str:
        return f"{self.path}: {self.where} {self.address}"


def is_external(address: str) -> bool:
    """Return whether an address points to another server."""
    return bool(_EXTERNAL.match(address)) and not _LOCAL_SCHEMES.match(address)


def allowed_by(address: str) -> str | None:
    """Return the decision that allows an external resource, or None."""
    for pattern, decision in EXCEPTIONS.items():
        if pattern.fullmatch(address.strip()):
            return decision
    return None


def is_public_address(address: str) -> bool:
    """Return whether an address is on the public address of the site."""
    return address.strip().startswith(PUBLIC_ORIGIN)


def _first(groups: Sequence[str | None]) -> str:
    return next((group for group in groups if group is not None), "")


def css_addresses(css: str) -> Iterator[tuple[str, str]]:
    """Yield ``(where, address)`` for each address a piece of CSS loads."""
    css = _CSS_COMMENT.sub("", css)
    for match in _CSS_IMPORT.finditer(css):
        yield "@import", _first(match.groups())
    for match in _CSS_URL.finditer(css):
        yield "url()", _first(match.groups())


def script_imports(script: str) -> Iterator[str]:
    """Yield the address of each ``import`` of an inline script."""
    for match in _JS_IMPORT.finditer(script):
        yield _first(match.groups())


def _srcset_addresses(value: str) -> Iterator[str]:
    for candidate in value.split(","):
        if words := candidate.split():
            yield words[0]


class _PageReader(HTMLParser):
    """Collect every address an HTML page loads, and the metadata that name an address."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.loads: list[tuple[str, str]] = []
        self.metadata: list[tuple[str, str]] = []
        # The text of the <script> or <style> being read: the parser may hand it in parts.
        self._inside: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name.lower(): value or "" for name, value in attrs}
        if style := values.get("style"):
            self.loads.extend(
                (f"<{tag} style> {where}", address) for where, address in css_addresses(style)
            )
        if tag == "link" and "href" in values:
            rels = set(values.get("rel", "").lower().split())
            if rels and rels <= NAVIGATION_RELS:
                self.metadata.append((f"<link rel={values['rel']}>", values["href"]))
            else:
                self.loads.append((f"<link rel={values.get('rel', '')}>", values["href"]))
        elif tag == "meta" and "content" in values:
            name = values.get("property") or values.get("name") or ""
            if is_external(values["content"]):
                self.metadata.append((f"<meta {name}>", values["content"]))
        for attribute in LOADING_ATTRIBUTES.get(tag, ()):
            if value := values.get(attribute):
                addresses = _srcset_addresses(value) if attribute == "srcset" else [value]
                self.loads.extend((f"<{tag} {attribute}>", address) for address in addresses)
        if tag in {"script", "style"}:
            self._inside = tag
            self._text = []

    def handle_endtag(self, tag: str) -> None:
        if tag != self._inside:
            return
        text = "".join(self._text)
        if tag == "script":
            self.loads.extend(("<script> import", address) for address in script_imports(text))
        else:
            self.loads.extend(
                (f"<style> {where}", address) for where, address in css_addresses(text)
            )
        self._inside = None

    def handle_data(self, data: str) -> None:
        if self._inside is not None:
            self._text.append(data)


def check_html(path: str, text: str) -> tuple[list[Finding], list[Finding]]:
    """Return the findings of an HTML page, and the external resources an exception allows."""
    reader = _PageReader()
    reader.feed(text)
    reader.close()
    findings, allowed = [], []
    for where, address in reader.loads:
        if not is_external(address):
            continue
        if allowed_by(address):
            allowed.append(Finding(path, where, address))
        else:
            findings.append(Finding(path, where, address))
    for where, address in reader.metadata:
        if is_external(address) and not is_public_address(address):
            findings.append(Finding(path, where, address))
    return findings, allowed


def check_css(path: str, text: str) -> list[Finding]:
    """Return the findings of a style sheet."""
    return [
        Finding(path, where, address)
        for where, address in css_addresses(text)
        if is_external(address)
    ]


def check_site(root: Path) -> tuple[list[Finding], list[Finding], int]:
    """Check every HTML and CSS file under a folder.

    Returns:
        The findings, the external resources an exception allows, and the number of
        files read.
    """
    findings: list[Finding] = []
    allowed: list[Finding] = []
    count = 0
    for file in sorted(root.rglob("*")):
        suffix = file.suffix.lower()
        if not file.is_file() or suffix not in {".html", ".htm", ".css"}:
            continue
        count += 1
        name = file.relative_to(root).as_posix()
        text = file.read_text(encoding="utf-8", errors="replace")
        if suffix == ".css":
            findings.extend(check_css(name, text))
        else:
            found, excepted = check_html(name, text)
            findings.extend(found)
            allowed.extend(excepted)
    return findings, allowed, count


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("site", type=Path, help="folder of the built site")
    options = parser.parse_args(arguments)
    root: Path = options.site
    if not root.is_dir():
        print(f"::error::{root} is not a folder", file=sys.stderr)
        return 2
    findings, allowed, count = check_site(root)
    for finding in allowed:
        decision = allowed_by(finding.address)
        print(f"allowed ({decision}): {finding}")
    for finding in findings:
        print(f"::error file={root / finding.path}::{finding}")
    if findings:
        print(f"{len(findings)} reference(s) to another server in {count} file(s).")
        return 1
    print(f"No resource from another server in {count} HTML and CSS file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
