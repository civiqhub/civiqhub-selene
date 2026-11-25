# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""``python -m selene_api``: check the configuration, then run the service."""

import os
import sys

from selene_api.config import ConfigError, Settings
from selene_api.security import TokenFileError

EX_CONFIG = 78
"""Exit status of a configuration error (sysexits.h)."""


def main() -> int:
    """Start the service, or exit with ``EX_CONFIG`` and the reason on the error stream."""
    try:
        settings = Settings.from_environ(os.environ)
        from selene_api.server import serve

        serve(settings)
    except (ConfigError, TokenFileError) as error:
        print(f"selene-api: configuration error: {error}", file=sys.stderr)
        return EX_CONFIG
    return 0


if __name__ == "__main__":
    sys.exit(main())
