# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), written in [PEP 440](https://peps.python.org/pep-0440/) notation. Before 1.0.0, a minor release may break the public API; such breaks are listed under **Changed** or **Removed**.

Changes to a method, its parameters or its reference vectors are listed explicitly, since they can change published results.

## [Unreleased]

### Added

- Core contract shared by every method, on the standard library alone: the `SeleneError` hierarchy (`errors`), the provenance and result types `Method`, `Randomness`, `Provenance`, `Result`, `Unavailable`, `StandardError` and `Interval` (`types`), the `Estimator` base class whose `estimate` is written once (`estimator`), the `Registry` of estimators read from the `civiqhub_selene.estimators` entry points (`registry`), and the contract kit `EstimatorContract` (`testing`).

## [0.0.1] - 2026-10-25

### Added

- Repository scaffold: `civiqhub_selene` package with reserved `ordinal`, `semantic` and `landscape` namespaces and a documented `types` contract, without any computation.
- Service scaffold under `services/api`: documented stubs, OpenAPI shell and commented Dockerfile.
- Test layout: unit, property-based, contract and reference-vector suites.
- Apache-2.0 license, NOTICE, governance, contribution and security policies, citation metadata.
- CI workflow (lint, type checking, tests, build, clean install, DCO) and release workflow publishing to PyPI through Trusted Publishing.

[Unreleased]: https://github.com/civiqhub/civiqhub-selene/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/civiqhub/civiqhub-selene/releases/tag/v0.0.1
