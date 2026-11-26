# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Machine tokens: read from a mounted file, compared in constant time, reloaded on change.

The token file holds one token per line, so that a rotation can list the old and the
new token for a while; empty lines and lines that start with ``#`` are ignored. A token
has at least 32 printable characters without spaces. The file is read again when it
changes, at most once per second. If it becomes unreadable or holds no valid token, every
call is refused until it is fixed: the service fails closed. A token is never logged.
"""

import hmac
import pathlib
import threading
import time
from collections.abc import Callable

__all__ = ["MIN_TOKEN_LENGTH", "TokenFileError", "TokenStore", "bearer_token"]

MIN_TOKEN_LENGTH = 32


class TokenFileError(ValueError):
    """The token file holds no valid token."""


def _read_tokens(path: pathlib.Path) -> tuple[bytes, ...]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError):
        raise TokenFileError("the token file cannot be read as ASCII text") from None
    tokens = []
    for line in lines:
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        if len(token) < MIN_TOKEN_LENGTH or not token.isprintable() or " " in token:
            raise TokenFileError(
                f"a token of the token file is shorter than {MIN_TOKEN_LENGTH} characters "
                "or holds a space"
            )
        tokens.append(token.encode("ascii"))
    if not tokens:
        raise TokenFileError("the token file holds no token")
    return tuple(tokens)


def bearer_token(header: str | None) -> str | None:
    """Return the token of an ``Authorization: Bearer`` header, or ``None``."""
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


class TokenStore:
    """The tokens this instance accepts.

    Args:
        path: The token file.
        min_interval: Seconds between two checks of the file for a change.
        clock: Monotonic clock.

    Raises:
        TokenFileError: If the file holds no valid token at start-up.
    """

    def __init__(
        self,
        path: pathlib.Path,
        *,
        min_interval: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.path = path
        self.min_interval = min_interval
        self._clock = clock
        self._lock = threading.Lock()
        self._tokens: tuple[bytes, ...] = _read_tokens(path)
        self._signature = self._stat()
        self._checked = clock()
        self.reloads = 0

    def _stat(self) -> tuple[int, int, int] | None:
        try:
            status = self.path.stat()
        except OSError:
            return None
        return (status.st_ino, status.st_size, status.st_mtime_ns)

    def _refresh(self) -> None:
        moment = self._clock()
        if moment - self._checked < self.min_interval:
            return
        with self._lock:
            self._checked = moment
            signature = self._stat()
            if signature == self._signature:
                return
            self._signature = signature
            try:
                self._tokens = _read_tokens(self.path)
            except TokenFileError:
                self._tokens = ()
            self.reloads += 1

    @property
    def usable(self) -> bool:
        """Whether at least one token is loaded."""
        self._refresh()
        return bool(self._tokens)

    def verify(self, presented: str | None) -> bool:
        """Whether ``presented`` is one of the tokens, compared in constant time."""
        self._refresh()
        if presented is None:
            return False
        try:
            candidate = presented.encode("ascii")
        except UnicodeEncodeError:
            return False
        matched = False
        for token in self._tokens:
            matched |= hmac.compare_digest(candidate, token)
        return matched
