# Governance

This document describes how `civiqhub-selene` is run: who decides, how changes are reviewed and how official releases are made.

## Steward

**CiviqHub SAS** is the steward of the official project. It controls the GitHub organisation and the PyPI project, appoints maintainers, approves official releases and operates its own deployment of the service.

Stewardship is not ownership of every contribution: contributors keep the copyright of their work, which they license under Apache-2.0 (see [`CONTRIBUTING.md`](CONTRIBUTING.md)). The rights granted by the license to everyone cannot be withdrawn.

## Roles

- **Contributors** propose changes through issues and pull requests. Anyone can be a contributor.
- **Maintainers** review and merge pull requests, triage issues and keep the CI healthy. The current maintainers are listed in [`.github/CODEOWNERS`](.github/CODEOWNERS).
- **Release managers** are maintainers designated by CiviqHub SAS to approve releases and the `pypi` deployment environment.

Being a contributor or a maintainer grants no access to CiviqHub's production systems or data.

### Becoming a maintainer

A maintainer nominates a contributor with a sustained record of quality contributions and reviews. CiviqHub SAS approves or declines the nomination after hearing the other maintainers. Maintainers who are inactive for twelve months, or who no longer follow this document, can be moved to emeritus status by CiviqHub SAS.

## Reviews

Every change goes through a pull request; nobody merges their own change without another maintainer's approval, and all CI checks must pass.

- **Technical review** covers correctness, tests, typing, packaging, the dependency policy and the contract between the engine and the service.
- **Scientific review** is required for any new method and for any change to the output of an existing method, its parameters or its reference vectors. It is performed by a maintainer with relevant expertise and checks the method and its results, not only the code: definition and references, assumptions and limits, uncertainty, test cases and vectors.

## Decisions

Decisions are made by consensus among maintainers in the relevant issue or pull request. When consensus cannot be reached, CiviqHub SAS decides and records the reasoning in the discussion.

## Releases

Official releases are made only from this repository, by release managers:

1. a `release/X.Y.Z` branch is opened from `develop`, with the version and [`CHANGELOG.md`](CHANGELOG.md) updated;
2. it is merged into `main` through a reviewed pull request, then merged back into `develop`;
3. the protected tag `vX.Y.Z` is pushed on `main`;
4. the release workflow builds the distributions, checks that the tag matches the version, and publishes them to PyPI through Trusted Publishing after approval of the `pypi` environment.

A published version number is never reused. A release that fails partway is reported as such and fixed with a new version. Publishing a release does not mean that CiviqHub deploys it: the deployed version of the CiviqHub service is decided separately.

## Security

Vulnerabilities are handled privately by the maintainers according to [`SECURITY.md`](SECURITY.md).

## Trademarks

The Apache License 2.0 does not grant permission to use the CiviqHub name or logos, except to describe the origin of the software. Forks must not present themselves as the official project.

## Changes to this document

Changes to this document are proposed by pull request and require the approval of CiviqHub SAS.
