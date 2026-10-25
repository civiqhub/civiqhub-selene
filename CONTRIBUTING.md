# Contributing to civiqhub-selene

`civiqhub-selene` is an early scaffold and not functional yet. This guide explains how to contribute. Decisions, reviews and releases are described in [`GOVERNANCE.md`](GOVERNANCE.md).

## Developer Certificate of Origin

Every commit carries a sign-off certifying that you have the right to submit it under the project's license, as defined by the [Developer Certificate of Origin 1.1](https://developercertificate.org).

Sign off with:

```bash
git commit -s -m "Fix typo in README"
```

This appends a line with your real name and email:

```
Signed-off-by: Jane Doe <jane.doe@example.org>
```

The sign-off email must match the commit author email. To sign off commits you already made on a branch: `git rebase --signoff develop`. The `dco` job of the CI runs on every pull request and lists any commit without a matching sign-off.

The DCO certifies that you may contribute; it does not transfer your copyright to CiviqHub SAS. If you contribute on behalf of an employer or a client, make sure they allow it before signing off.

## License of contributions

Contributions are accepted under the [Apache License 2.0](LICENSE), the project's license (inbound equals outbound). Start each new source file with:

```python
# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0
```

You may add your own copyright line above the SPDX identifier for a substantial new file. Do not copy code from another project, including the MPL-2.0 `civiqhub` SDK, without checking that its license allows it and recording its attribution in [`NOTICE`](NOTICE).

## Dependency policy

- **Core:** the engine relies on the standard library (dataclasses, enums, protocols). NumPy and SciPy are added when a method actually needs them, not in advance.
- **Optional:** heavy dependencies (embedding models, graph libraries, data adapters) live in the extras declared in `pyproject.toml` and are never imported by the core.
- **Boundaries:** the engine depends on neither the `civiqhub` SDK, nor the CiviqHub backend, nor any network service. The service wrapper under `services/api` keeps its own dependencies; they never enter the scientific package.
- **Licenses:** every dependency, optional ones included, uses a permissive license compatible with Apache-2.0 (BSD, MIT, Apache-2.0, PSF). GPL and AGPL dependencies are excluded by project policy. Model weights and datasets keep their own licenses and are checked separately. Discuss any new dependency in an issue before adding it.

## Scientific contributions

Each method ships with:

1. its mathematical definition in the docstring, with the reference it comes from;
2. its uncertainty (variance, interval or posterior) whenever one can be computed;
3. explicit, typed and versioned parameters with documented defaults, so that a published result can be reproduced with the same settings; an example value is not a validated calibration;
4. unit tests on known cases: empty input, unanimity, extreme splits, small samples and exact threshold values;
5. property-based tests for its invariants, and reference vectors under `tests/vectors` with their numerical tolerance.

Methods are deterministic: any randomness takes an explicit seed, and any model is selected explicitly by the caller and recorded in the result. Results describe a group and decide nothing.

Changing the output of an existing method, its defaults or a reference vector is a scientific change: say so in the pull request, justify it, and list it in [`CHANGELOG.md`](CHANGELOG.md).

## Data

Tests, vectors, examples and documentation use synthetic data only. Never commit data from real participants, even anonymised, and never include such data in an issue or a pull request.

## Service contributions

The service under `services/api` adapts HTTP requests to the engine; it does not compute anything itself. A change to its schemas updates [`services/api/openapi.yaml`](services/api/openapi.yaml) and the contract tests in the same pull request.

## Development workflow

```bash
git clone https://github.com/civiqhub/civiqhub-selene
cd civiqhub-selene
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e . --group dev
ruff check && ruff format --check
mypy
pytest
```

1. Open an issue describing the change, unless it is a small fix.
2. Branch from `develop`, with signed-off commits.
3. Open a pull request to `develop` linked to the issue; all CI checks must pass.
4. A maintainer reviews and merges; scientific changes also get a scientific review.

`develop` is the integration branch. Releases go through a `release/X.Y.Z` branch merged into `main`, tagged `vX.Y.Z`, then merged back into `develop`. Urgent fixes branch from `main`.

## Citation

If your contribution implements a published method, add its reference to the docstring. If you publish research using `civiqhub-selene`, cite it with [`CITATION.cff`](CITATION.cff).
