# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), written in [PEP 440](https://peps.python.org/pep-0440/) notation. Before 1.0.0, a minor release may break the public API; such breaks are listed under **Changed** or **Removed**.

Changes to a method, its parameters or its reference vectors are listed explicitly, since they can change published results.

## [Unreleased]

### Changed

- The documentation site publishes only the latest release and the development branch. Earlier releases are not built (Decision of 2026-11-24).
- The development branch carries the version `0.1.0.dev0`, so that the provenance of a result computed from it (`engine_version`) cannot be mistaken for the published 0.0.1. The service scaffold under `services/api` stays at 0.0.1 until the next release.
- Block models take a new setting, `initialization`, the object that gives the arrangement each start begins from. Its default, `RandomStart()`, draws exactly the former random starts: the values of `LatentBlockModel` and its reference vector are unchanged. The setting enters the provenance, so the fingerprint of a result computed before this change differs from the fingerprint of the same computation now.
- The classified log-likelihood of `LatentBlockModel` and the observed log-likelihood of `MixedMembershipBlockModel` take mean-one survey weights as a weighted pseudo-likelihood (Skinner 1989, Pfeffermann 1993). The weighted ICL of the single-membership model is a Hypothesis. Reported co-cluster counts stay the raw number of survey rows.

### Added

- The service under `services/api` (selene-api 0.1.0) serves contract v1: `python -m selene_api` starts one Starlette/Uvicorn process configured by `SELENE_*` variables, which refuses to start without a machine token, or without TLS off the loopback interface. Each job runs `LatentBlockModel` of the engine in a child process with one numerical thread, through the adapter of `coclustering.single` schema 1; the job manager keeps jobs in memory only, admits one computation at a time, replays a key with the same content, refuses another content (409) or a second computation (503), and forgets finished jobs after their time to live. Runtime dependencies are locked with their hashes in `services/api/requirements.lock`. The `service contract` CI job becomes `service` and installs from the lock.
- Contract v1 of the service, in `services/api/openapi.yaml` (contract version 1.0.0): `POST /v1/jobs`, `GET` and `DELETE /v1/jobs/{job_id}`, `GET /v1/service`, `/healthz` and `/readyz`, with the job envelope, the error schema, the limits and the versions of the service, the contract and the engine kept apart. The first kind is `coclustering.single`, payload schema 1. Its compatibility policy, semantics and deployment interfaces are in `services/api/README.md`. TEST ONLY fixtures under `services/api/contract/fixtures/v1`, with a reference result computed locally with the engine, and a contract client on the standard library alone (`services/api/contract/selene_contract.py`). The new `service contract` CI job checks them. Nothing of the engine changes.
- Survey input through `VoteMatrix.from_responses`, `read_survey_csv` and `VoterCovariates`. `from_responses` keeps each item's original ranks and the direction of agreement; it refuses heterogeneous observation scales until SEL-8.2 (ItemScales). Optional mean-one voter weights enter both block models. `VoterCovariates` lives in `blocks.covariates` and is re-exported from `civiqhub_selene.blocks`.

- Core contract shared by every method, on the standard library alone: the `SeleneError` hierarchy (`errors`), the provenance and result types `Method`, `Randomness`, `Provenance`, `Result`, `Unavailable`, `StandardError` and `Interval` (`types`), the `Estimator` base class whose `estimate` is written once (`estimator`), the `Registry` of estimators read from the `civiqhub_selene.estimators` entry points (`registry`), and the contract kit `EstimatorContract` (`testing`).
- `ordinal.VoteCounts`, the five vote counts of an item on the agreement scale, with exact shares.
- Optional extra `blocks` (`pip install "civiqhub-selene[blocks]"`), which depends on NumPy 2.0 or later (BSD-3-Clause) and opens the `civiqhub_selene.blocks` subpackage for co-clustering. Without the extra, importing `civiqhub_selene.blocks` raises `MissingExtraError`; the core and `ordinal` stay on the standard library, which the new `test-core` CI job checks.
- Data of co-clustering in `civiqhub_selene.blocks`: `OrdinalScale`, `VoteMatrix` (a vote matrix with three states per cell: voted, shown without a vote, not shown), `Partition` with `VoterGroup` and `ItemBlock`, `GroupedVotes`, `Cocluster` and `CoclusterTable`, and the synthetic generator `make_votes` with its planted truth.
- Response models of the co-clusters: the `ResponseModel` protocol, `Categorical` (a free distribution over the scores) and `OrdinalProbit` (a latent normal opinion cut by fixed thresholds; Corneli, Bouveyron, Latouche 2020), with reference vectors under `tests/vectors/blocks_numerics.json`.
- Architecture of the block models: `IterativeEstimator` (several starts, one seed per start, convergence, best start), `CoclusteringEstimator`, `BlockModel`, the plain-word lookup `estimator_for`, and the values `Coclustering`, `SingleCoclustering`, `MixedCoclustering` and `Selection`. The mixed-membership model, the choice of shape and the `cocluster` function come next.
- `blocks.LatentBlockModel`, the first block model: co-clustering where each voter belongs to one group and each item to one block, estimated by classification expectation-maximisation (Govaert and Nadif 2003), with the integrated completed likelihood of its value (Biernacki, Celeux and Govaert 2000). It is declared under the entry point `coclustering.single`, and a reference run on stored votes lives in `tests/vectors/latent_block_model.json`.
- Starts of the block models, in `civiqhub_selene.blocks.starts`: the `Initialization` protocol, `RandomStart`, `SpectralStart` and `FixedStart` (a given arrangement, for example the one a single-membership model found, which the provenance then cites).
- Regularised spectral co-clustering of a vote matrix, in `civiqhub_selene.blocks.spectral` (Qin and Rohe 2013; Rohe, Qin and Yu 2016, DI-SIM): `spectral_embedding` divides the encoded votes by the square roots of their regularised degrees and keeps the leading singular vectors; `SpectralEmbedding.labels` projects the rows onto the sphere and runs k-means, with the thresholding of t-RSC as an option. The votes are encoded by default as one-hot states (shown without a vote, or a vote at one level), so that abstentions and the spread of the votes count; the encoding `"scores"` decomposes centred scores. `SpectralStart` hands this arrangement to a block model.
- `blocks.MixedMembershipBlockModel`, the mixed-membership ordinal latent block model (MM-OLBM): each voter has a degree of membership to each group and follows one group per block, each item has a degree of membership to each block. It is estimated by an exact expectation-maximisation over the `n_groups ** n_blocks` profiles, regularised by Dirichlet priors on the memberships (`voter_concentration`, `item_concentration`), and returns a `MixedCoclustering`. It is declared under the entry point `coclustering.mixed`. Its expectation step agrees with the reference engine of the explainer "LBM and MM-OLBM step by step" to 1e-4, and both reach the same optimum; the reference values live in `tests/vectors/mixed_membership_block_model.json`.
- Optional extra `viz` (`pip install "civiqhub-selene[viz]"`), which depends on NetworkX 3.4 or later (BSD-3-Clause) and Matplotlib 3.9 or later (Matplotlib license), and opens the `civiqhub_selene.viz` subpackage: `viz.graphs` turns a co-clustering into NetworkX graphs (co-clusters, memberships, roles from block to block, agreement between groups, votes), `viz.draw` draws them with the sorted vote matrix, the spectrum and the embedding of the spectral start, the convergence and the mixedness, and `viz.style` holds the colours. No other part of the engine imports it.
- Notebooks in `examples/`, on synthetic data: loading votes from CSV and the spectral start (`01_spectral_start`), the single-membership model (`02_olbm`) and the mixed-membership model (`03_mm_olbm`) from end to end, with their figures. `tests/unit/test_examples.py` runs every code cell. The documentation site gains a page that presents them, and *Getting started* shows the spectral start, the mixed memberships and the graphs.
- Dependency group `examples` (nbclient, nbformat, ipykernel) to run the notebooks from the command line, and the stubs of NetworkX in the group `lint`.
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

[Unreleased]: https://github.com/civiqhub/civiqhub-selene/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/civiqhub/civiqhub-selene/releases/tag/v0.0.1
