"""Streamed, checksum-verified downloads for LocusBlend reference files.

Standard library only (``urllib``): no third-party network dependency.
Downloads are always explicit - nothing in this module runs during ``plot()``.

Guarantees:

* ``https://`` (and ``file://`` for local/testing use) only; other schemes are
  refused, including plain ``http://``
* streamed chunked writes into ``<name>.part``
* SHA256 verification *before* the file is moved into place
* atomic ``os.replace`` so an interrupted download never leaves a corrupt file
  at the final path
* ``.part`` files are deleted when a download or checksum fails
* an existing file with the expected checksum is skipped (unless ``force=True``)
* checksum problems always raise (:class:`ChecksumError`), never ignored
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DEFAULT_CHUNK_SIZE = 1 << 20  # 1 MiB
DEFAULT_TIMEOUT = 30.0
PART_SUFFIX = ".part"
USER_AGENT = "locusblend-reference-installer/0.1.0.dev0"

#: Schemes accepted for reference downloads.
SUPPORTED_URL_SCHEMES = ("https", "file")

STATUS_DOWNLOADED = "downloaded"
STATUS_SKIPPED = "skipped"
STATUS_REPLACED = "replaced"


class DownloadError(RuntimeError):
    """Raised when a reference file cannot be downloaded."""


class ChecksumError(DownloadError):
    """Raised when a downloaded or existing file fails SHA256 verification."""


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of one :func:`download_file` call."""

    path: str
    status: str
    bytes_written: int = 0
    sha256: Optional[str] = None

    @property
    def skipped(self) -> bool:
        return self.status == STATUS_SKIPPED

    def describe(self) -> str:
        return f"{self.status}: {self.path}"


def normalize_checksum(value) -> str:
    """Validate and normalize an expected SHA256 hex digest."""
    if not isinstance(value, str) or not value.strip():
        raise ChecksumError("A SHA256 checksum is required to verify downloads.")
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest.split(":", 1)[1]
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ChecksumError(
            f"Invalid SHA256 checksum {value!r}: expected 64 hexadecimal characters."
        )
    return digest


def validate_url(url) -> str:
    """Return *url* when its scheme is accepted, else raise :class:`DownloadError`."""
    parsed = urlparse(str(url))
    if parsed.scheme not in SUPPORTED_URL_SCHEMES:
        raise DownloadError(
            f"Unsupported URL scheme {parsed.scheme or '<none>'!r} for {url!r}. "
            "Reference downloads support https:// (and file:// for local/testing "
            "use) only."
        )
    return str(url)


def sha256_file(path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Return the SHA256 hex digest of *path* (streamed; no full read into memory)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_matches_checksum(path, expected_sha256, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> bool:
    """True when *path* exists and its SHA256 digest matches *expected_sha256*."""
    expected = normalize_checksum(expected_sha256)
    path = Path(path)
    if not path.is_file():
        return False
    return sha256_file(path, chunk_size=chunk_size) == expected


def open_url(url, *, timeout: float = DEFAULT_TIMEOUT):
    """Open *url* for binary reading using the standard library."""
    url = validate_url(url)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        return urlopen(request, timeout=timeout)
    except HTTPError as exc:
        raise DownloadError(f"HTTP error {exc.code} while fetching {url}: {exc.reason}") from exc
    except URLError as exc:
        raise DownloadError(f"Could not fetch {url}: {exc.reason}") from exc
    except OSError as exc:
        raise DownloadError(f"Could not fetch {url}: {exc}") from exc


def fetch_bytes(url, *, timeout: float = DEFAULT_TIMEOUT, limit: Optional[int] = None) -> bytes:
    """Fetch a small resource (for example a manifest) into memory."""
    chunk_size = 64 * 1024
    chunks = []
    total = 0
    with open_url(url, timeout=timeout) as response:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if limit is not None and total > limit:
                raise DownloadError(f"Resource at {url} exceeds the {limit} byte limit.")
            chunks.append(chunk)
    return b"".join(chunks)


def _remove_part_file(part_path: Path) -> None:
    try:
        part_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def download_file(
    url,
    destination,
    *,
    sha256,
    size=None,
    force: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    timeout: float = DEFAULT_TIMEOUT,
) -> DownloadResult:
    """Download *url* to *destination*, verifying SHA256 (and size when known).

    * existing file with the expected checksum -> skipped unless ``force=True``
    * existing file with a different checksum -> downloaded again and replaced
    * failures delete the ``.part`` file; the final path is only ever created by
      an atomic rename of a verified download
    """
    expected = normalize_checksum(sha256)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(destination.name + PART_SUFFIX)
    previous_exists = destination.is_file()

    if previous_exists and not force and file_matches_checksum(
        destination, expected, chunk_size=chunk_size
    ):
        return DownloadResult(str(destination), STATUS_SKIPPED, 0, expected)

    _remove_part_file(part_path)
    digest = hashlib.sha256()
    written = 0
    try:
        with open_url(url, timeout=timeout) as response, open(part_path, "wb") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                written += len(chunk)
                if size is not None and written > int(size):
                    raise ChecksumError(
                        f"Downloaded data from {url} exceeds the manifest size "
                        f"({size} bytes); aborting."
                    )
                digest.update(chunk)
                handle.write(chunk)
    except BaseException:
        _remove_part_file(part_path)
        raise

    actual = digest.hexdigest()
    if size is not None and written != int(size):
        _remove_part_file(part_path)
        raise ChecksumError(
            f"Size mismatch for {url}: manifest declares {size} bytes, downloaded "
            f"{written} bytes. The file was not installed."
        )
    if actual != expected:
        _remove_part_file(part_path)
        raise ChecksumError(
            f"Checksum mismatch for {url}: expected sha256 {expected}, got {actual}. "
            "The downloaded data was discarded and no file was installed."
        )

    os.replace(part_path, destination)
    status = STATUS_REPLACED if previous_exists else STATUS_DOWNLOADED
    return DownloadResult(str(destination), status, written, actual)