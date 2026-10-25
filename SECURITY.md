# Security policy

## Supported versions

`civiqhub-selene` is in its 0.x series. Security fixes are made on the latest release only.

## Reporting a vulnerability

Do not open a public issue, pull request or discussion for a vulnerability.

Report it privately through GitHub: on the repository page, open the **Security** tab and choose **Report a vulnerability**. If you cannot use GitHub, write to contact@civiqhub.com with "SECURITY civiqhub-selene" in the subject.

Include the affected version or commit, a description of the issue and its impact, and the steps to reproduce it. Do not include real personal data in your report; use synthetic inputs.

## What to expect

- an acknowledgement within 3 business days;
- a first assessment within 10 business days;
- a fix, or a mitigation plan, coordinated with you before any disclosure;
- public disclosure through a GitHub security advisory once a fixed release is available, at the latest 90 days after the report unless we agree otherwise.

We credit reporters in the advisory unless they prefer to remain anonymous.

## Scope

In scope: the engine (`src/civiqhub_selene`), the service wrapper and its image definition (`services/api`), and the CI and release workflows of this repository.

Out of scope: the CiviqHub platform and its deployments, and vulnerabilities in third-party dependencies that are not specific to how this project uses them; report those to their maintainers. For the CiviqHub platform, write to contact@civiqhub.com.

## Good faith

We will not pursue legal action against research conducted in good faith that respects this policy, avoids privacy violations and service disruption, and does not access data that does not belong to the researcher.
