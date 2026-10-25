# civiqhub-selene

[![CI](https://github.com/civiqhub/civiqhub-selene/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/civiqhub/civiqhub-selene/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/civiqhub-selene.svg)](https://pypi.org/project/civiqhub-selene/)

SELENE is the open-source scientific engine of [CiviqHub](https://civiqhub.com) for analysing collective deliberation, usable as a Python library or behind a containerised service.

**Status: early scaffold, not functional yet.** Version 0.0.1 contains the repository structure, governance documents and tooling; the package exposes nothing but `__version__`.

## Install

```bash
pip install civiqhub-selene
```

The distribution is named `civiqhub-selene`; the package is imported as `civiqhub_selene`:

```python
import civiqhub_selene
```

## Layout

| Path | Content |
| --- | --- |
| [`src/civiqhub_selene/`](src/civiqhub_selene) | The engine, published on PyPI. Deterministic computations on explicit inputs, with no network access and no dependency on CiviqHub services. |
| [`services/api/`](services/api) | The HTTP wrapper of the engine and its container image. It installs the engine; it never reimplements it. |
| [`tests/`](tests) | Unit, property-based and contract tests, and reference vectors. |

## Relationship with `civiqhub`

The [`civiqhub`](https://github.com/civiqhub/civiqhub-py) Python SDK (MPL-2.0) will offer SELENE through an optional dependency on this package. The dependency goes one way only: `civiqhub_selene` never imports the SDK and works without a CiviqHub account.

## Citing

Citation metadata lives in [`CITATION.cff`](CITATION.cff); on GitHub, the "Cite this repository" button exports it as BibTeX or APA.

## Contributing

Contributions are welcome under the [Developer Certificate of Origin](https://developercertificate.org): sign off each commit with `git commit -s`. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full procedure and [`GOVERNANCE.md`](GOVERNANCE.md) for how decisions and releases are made. Report vulnerabilities privately as described in [`SECURITY.md`](SECURITY.md).

## License

[Apache License 2.0](LICENSE). You can use, modify and redistribute `civiqhub-selene`, including commercially, provided you keep the copyright, license and [`NOTICE`](NOTICE) attributions. The license does not grant rights to the CiviqHub name or logos.
