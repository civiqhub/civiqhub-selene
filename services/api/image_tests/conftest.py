# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Tests of the final image of the service. TEST ONLY data.

They run real containers of the image named by ``SELENE_IMAGE``, built by
``services/api/image/build.sh``, with the restrictions of the deployments: non-root,
read-only root file system, every capability dropped, no privilege escalation, bounded
memory, processes and temporary space, and synthetic certificates and tokens. Run them
from the root of the repository::

    services/api/image/build.sh
    SELENE_IMAGE=selene-api:candidate pytest services/api/image_tests

They fail, rather than skip, when no image is named: a CI job that runs them must test
an image.
"""

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
for _directory in (_HERE.parent / "contract", _HERE.parent / "tests", _HERE):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

pytest_plugins = ["imagefixtures"]
