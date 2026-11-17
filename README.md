# civiqhub-selene

[![CI](https://github.com/civiqhub/civiqhub-selene/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/civiqhub/civiqhub-selene/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/civiqhub-selene.svg)](https://pypi.org/project/civiqhub-selene/)

SELENE is the open-source scientific engine of [CiviqHub](https://civiqhub.com) for analysing collective deliberation, usable as a Python library or behind a containerised service.

**Status: foundations and a first method, not released yet.** The published version 0.0.1 contains the repository structure, governance documents and tooling, and exposes nothing but `__version__`. The development branch adds the contract every method shares (errors, results with their provenance, estimators, registry, contract kit), the vote counts of the ordinal family, and the data and the architecture of co-clustering (vote matrix, partitions, response models, base classes of the block models). It also holds a first method, the single-membership block model (`LatentBlockModel`). The dissensus index, the mixed-membership block model and the choice of shape come next: see [`CHANGELOG.md`](CHANGELOG.md) and [`docs/architecture.md`](docs/architecture.md).

## Install

```bash
pip install civiqhub-selene
```

The core and the ordinal family use the standard library only. Co-clustering needs NumPy, behind an optional extra:

```bash
pip install "civiqhub-selene[blocks]"
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
| [`docs/`](docs) | How the engine is built ([`architecture.md`](docs/architecture.md)); the documentation of each method is written with it. |

## Relationship with `civiqhub`

The [`civiqhub`](https://github.com/civiqhub/civiqhub-py) Python SDK (MPL-2.0) will offer SELENE through an optional dependency on this package. The dependency goes one way only: `civiqhub_selene` never imports the SDK and works without a CiviqHub account.

## Citing

Citation metadata lives in [`CITATION.cff`](CITATION.cff); on GitHub, the "Cite this repository" button exports it as BibTeX or APA.

## Contributing

Contributions are welcome under the [Developer Certificate of Origin](https://developercertificate.org): sign off each commit with `git commit -s`. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full procedure and [`GOVERNANCE.md`](GOVERNANCE.md) for how decisions and releases are made. Report vulnerabilities privately as described in [`SECURITY.md`](SECURITY.md).

## License

[Apache License 2.0](LICENSE). You can use, modify and redistribute `civiqhub-selene`, including commercially, provided you keep the copyright, license and [`NOTICE`](NOTICE) attributions. The license does not grant rights to the CiviqHub name or logos.
