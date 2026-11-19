# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), written in [PEP 440](https://peps.python.org/pep-0440/) notation. Before 1.0.0, a minor release may break the public API; such breaks are listed under **Changed** or **Removed**.

Changes to a method, its parameters or its reference vectors are listed explicitly, since they can change published results.

## [Unreleased]

## [0.1.0] - 2026-11-19

First version that computes: the contract every method shares, the vote counts of the ordinal family, and co-clustering with the single-membership block model.

### Changed

- The service scaffold under `services/api` follows the engine: its version is 0.1.0 and it pins `civiqhub-selene==0.1.0`. The version of its HTTP contract, in `openapi.yaml`, is separate and stays at 0.0.1.
- The description of the package no longer says "not functional yet".

### Added

- Core contract shared by every method, on the standard library alone: the `SeleneError` hierarchy (`errors`), the provenance and result types `Method`, `Randomness`, `Provenance`, `Result`, `Unavailable`, `StandardError` and `Interval` (`types`), the `Estimator` base class whose `estimate` is written once (`estimator`), the `Registry` of estimators read from the `civiqhub_selene.estimators` entry points (`registry`), and the contract kit `EstimatorContract` (`testing`).
- `ordinal.VoteCounts`, the five vote counts of an item on the agreement scale, with exact shares.
- Optional extra `blocks` (`pip install "civiqhub-selene[blocks]"`), which depends on NumPy 2.0 or later (BSD-3-Clause) and opens the `civiqhub_selene.blocks` subpackage for co-clustering. Without the extra, importing `civiqhub_selene.blocks` raises `MissingExtraError`; the core and `ordinal` stay on the standard library, which the new `test-core` CI job checks.
- Data of co-clustering in `civiqhub_selene.blocks`: `OrdinalScale`, `VoteMatrix` (a vote matrix with three states per cell: voted, shown without a vote, not shown), `Partition` with `VoterGroup` and `ItemBlock`, `GroupedVotes`, `Cocluster` and `CoclusterTable`, and the synthetic generator `make_votes` with its planted truth.
- Response models of the co-clusters: the `ResponseModel` protocol, `Categorical` (a free distribution over the scores) and `OrdinalProbit` (a latent normal opinion cut by fixed thresholds; Corneli, Bouveyron, Latouche 2020), with reference vectors under `tests/vectors/blocks_numerics.json`.
- Architecture of the block models: `IterativeEstimator` (several starts, one seed per start, convergence, best start), `CoclusteringEstimator`, `BlockModel`, the plain-word lookup `estimator_for`, and the values `Coclustering`, `SingleCoclustering`, `MixedCoclustering` and `Selection`. The mixed-membership model, the choice of shape and the `cocluster` function come next.
- `blocks.LatentBlockModel`, the first block model: co-clustering where each voter belongs to one group and each item to one block, estimated by classification expectation-maximisation (Govaert and Nadif 2003), with the integrated completed likelihood of its value (Biernacki, Celeux and Govaert 2000). It is declared under the entry point `coclustering.single`, and a reference run on stored votes lives in `tests/vectors/latent_block_model.json`.
- Property-based tests with Hypothesis, a test-only dependency.
- `docs/architecture.md`: how the engine is built and how to add a method.
- Sources of a documentation site under `docs/`, built with Sphinx: a getting-started page, a map of the library, and a reference generated from the docstrings, one page per public module. The `documentation` CI job builds it with warnings as errors; the tests check that every public module has its page and that every Python block of the pages runs and prints the output shown.
- Publication of the documentation site on GitHub Pages, at <https://civiqhub.github.io/civiqhub-selene/>, by the `docs` workflow: one folder per version (the release that `main` holds, the earlier release tags, and the development branch), and a version selector on every page.

## [0.0.1] - 2026-10-25

### Added

- Repository scaffold: `civiqhub_selene` package with reserved `ordinal`, `semantic` and `landscape` namespaces and a documented `types` contract, without any computation.
- Service scaffold under `services/api`: documented stubs, OpenAPI shell and commented Dockerfile.
- Test layout: unit, property-based, contract and reference-vector suites.
- Apache-2.0 license, NOTICE, governance, contribution and security policies, citation metadata.
- CI workflow (lint, type checking, tests, build, clean install, DCO) and release workflow publishing to PyPI through Trusted Publishing.

[Unreleased]: https://github.com/civiqhub/civiqhub-selene/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/civiqhub/civiqhub-selene/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/civiqhub/civiqhub-selene/releases/tag/v0.0.1
