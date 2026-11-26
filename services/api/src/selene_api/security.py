# Copyright 2026 CiviqHub SAS
# SPDX-License-Identifier: Apache-2.0

"""Machine tokens and the server certificate, read from mounted files and reloaded.

Machine tokens: read from a mounted file or from a single in-memory value, compared in
constant time. A file is reloaded on change; an in-memory token is not.

The token file holds one token per line, so that a rotation can list the old and the
new token for a while; empty lines and lines that start with ``#`` are ignored. A token
has at least 32 printable characters without spaces. The file is read again when it
changes, at most once per second. If it becomes unreadable or holds no valid token, every
call is refused until it is fixed: the service fails closed. A token is never logged,
written back to disk, or stored in a form that would print it.
"""

import hmac
import pathlib
import ssl
import threading
import time
from collections.abc import Callable

__all__ = [
    "MIN_TOKEN_LENGTH",
    "CertificateReloader",
    "TokenFileError",
    "TokenStore",
    "bearer_token",
    "token_form_is_valid",
    "token_listed",
]

MIN_TOKEN_LENGTH = 32


class TokenFileError(ValueError):
    """The token file holds no valid token."""


def token_form_is_valid(text: str) -> bool:
    """Whether ``text`` is a machine token: 32 or more printable ASCII characters, no space."""
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return False
    return len(text) >= MIN_TOKEN_LENGTH and text.isprintable() and " " not in text


def token_listed(tokens: tuple[bytes, ...], presented: str) -> bool:
    """Whether ``presented`` is one of ``tokens``, compared in constant time."""
    try:
        candidate = presented.encode("ascii")
    except UnicodeEncodeError:
        return False
    matched = False
    for token in tokens:
        matched |= hmac.compare_digest(candidate, token)
    return matched


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
        if not token_form_is_valid(token):
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
        path: The token file, or ``None`` when the tokens come from memory.
        token: A single in-memory token, used when ``path`` is ``None``. When both
            ``path`` and ``token`` are set, the file is the live source; ``token`` is
            ignored after start-up.
        min_interval: Seconds between two checks of the file for a change.
        clock: Monotonic clock.

    Raises:
        TokenFileError: If the file holds no valid token at start-up, or if there is
            no in-memory token when ``path`` is ``None``.
    """

    def __init__(
        self,
        path: pathlib.Path | None = None,
        *,
        token: str | None = None,
        min_interval: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.path = path
        self.min_interval = min_interval
        self._clock = clock
        self._lock = threading.Lock()
        self.reloads = 0
        self._tokens: tuple[bytes, ...]
        if path is None:
            if token is None or not token_form_is_valid(token):
                raise TokenFileError(
                    f"a token is shorter than {MIN_TOKEN_LENGTH} characters or holds a space"
                )
            self._tokens = (token.encode("ascii"),)
            self._signature = None
            self._checked = clock()
            return
        self._tokens = _read_tokens(path)
        self._signature = self._stat()
        self._checked = clock()

    def _stat(self) -> tuple[int, int, int] | None:
        path = self.path
        if path is None:
            return None
        try:
            status = path.stat()
        except OSError:
            return None
        return (status.st_ino, status.st_size, status.st_mtime_ns)

    def _refresh(self) -> None:
        path = self.path
        if path is None:
            return
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
                self._tokens = _read_tokens(path)
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
        return token_listed(self._tokens, presented)


class CertificateReloader:
    """Reload the server certificate into the live TLS context when its files change.

    ``ssl.SSLContext.load_cert_chain`` on the context Uvicorn serves with applies to
    the next handshakes: a renewed certificate is served without a restart, and open
    connections keep theirs. If the new files cannot be loaded (half written, key and
    certificate that do not match), the previous certificate stays and the failure is
    counted; TLS is never turned off.

    Args:
        certificate: The PEM certificate chain file.
        key: The PEM private key file.
        context: The server TLS context; attached later with :meth:`attach`.
    """

    def __init__(
        self,
        certificate: pathlib.Path,
        key: pathlib.Path,
        context: "ssl.SSLContext | None" = None,
    ) -> None:
        self.certificate = certificate
        self.key = key
        self.context = context
        self.reloads = 0
        self.failures = 0
        self._signature = self._stat()

    def attach(self, context: "ssl.SSLContext") -> None:
        """Use ``context``, already loaded with the current files."""
        self.context = context
        self._signature = self._stat()

    def _stat(self) -> tuple[tuple[int, int, int] | None, ...]:
        signatures: list[tuple[int, int, int] | None] = []
        for path in (self.certificate, self.key):
            try:
                status = path.stat()
            except OSError:
                signatures.append(None)
            else:
                signatures.append((status.st_ino, status.st_size, status.st_mtime_ns))
        return tuple(signatures)

    def refresh(self) -> bool:
        """Reload the files if they changed; return whether a new certificate is served."""
        if self.context is None:
            return False
        signature = self._stat()
        if signature == self._signature or None in signature:
            return False
        self._signature = signature
        try:
            self.context.load_cert_chain(str(self.certificate), str(self.key))
        except (OSError, ssl.SSLError):
            self.failures += 1
            return False
        self.reloads += 1
        return True
