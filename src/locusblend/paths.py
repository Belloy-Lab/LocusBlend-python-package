"""Reference-data location resolution for LocusBlend.

LocusBlend reference data live outside the package. Resolution priority is:

1. an explicit ``reference_dir=...`` argument
2. the ``LOCUSBLEND_REFERENCE_DIR`` environment variable
3. the managed default reference directory, when it already exists

The managed directory is a per-user cache location derived from
``platformdirs`` (never hard-coded per OS), for example:

* Windows:  ``%LOCALAPPDATA%\\locusblend\\Cache\\references``
* macOS:    ``~/Library/Caches/locusblend/references``
* Linux:    ``~/.cache/locusblend/references``

Nothing here downloads anything: :func:`resolve_reference_dir` only inspects
paths and reports which source was used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import platformdirs

#: Application name used for the managed cache location.
APP_NAME = "locusblend"

#: Sub-directory of the managed cache that holds reference data.
MANAGED_REFERENCE_SUBDIR = "references"

#: Environment variable that overrides the managed default location.
REFERENCE_DIR_ENV_VAR = "LOCUSBLEND_REFERENCE_DIR"

#: Possible outcomes of reference-dir resolution.
SOURCE_EXPLICIT = "explicit"
SOURCE_ENV = "env"
SOURCE_MANAGED = "managed"


def get_default_reference_dir() -> Path:
    """Return the managed default reference directory (not created here).

    The location comes from :func:`platformdirs.user_cache_dir` with
    ``appauthor=False`` so it is identical on every platform apart from the
    platform-specific cache root. The directory is *not* created and nothing is
    downloaded; use :func:`locusblend.install_reference` to populate it.
    """
    cache_root = platformdirs.user_cache_dir(APP_NAME, appauthor=False)
    return Path(cache_root) / MANAGED_REFERENCE_SUBDIR


def reference_dir_from_env() -> Optional[Path]:
    """Return ``LOCUSBLEND_REFERENCE_DIR`` as a path, or None when unset."""
    value = os.environ.get(REFERENCE_DIR_ENV_VAR, "").strip()
    return Path(value).expanduser() if value else None


@dataclass(frozen=True)
class ReferenceLocation:
    """Result of :func:`resolve_reference_dir`.

    ``source`` is one of ``"explicit"``, ``"env"``, ``"managed"`` or ``None``
    (nothing usable configured). ``path`` is None when no source applied.
    """

    path: Optional[Path] = None
    source: Optional[str] = None

    @property
    def is_resolved(self) -> bool:
        return self.path is not None


def resolve_reference_dir(
    reference_dir=None,
    *,
    use_managed_default: bool = True,
    require_existing_managed: bool = True,
) -> ReferenceLocation:
    """Resolve a reference directory using the documented priority.

    Parameters
    ----------
    reference_dir:
        Explicit directory (highest priority).
    use_managed_default:
        When False, only the explicit path and the environment variable are
        considered (used by callers that must not touch the managed cache).
    require_existing_managed:
        When True (default), the managed default is only used if it already
        exists as a directory, so a missing managed install keeps failing
        loudly instead of masking missing reference data.
    """
    if reference_dir is not None and str(reference_dir).strip() != "":
        return ReferenceLocation(Path(reference_dir).expanduser(), SOURCE_EXPLICIT)

    env_dir = reference_dir_from_env()
    if env_dir is not None:
        return ReferenceLocation(env_dir, SOURCE_ENV)

    if use_managed_default:
        managed = get_default_reference_dir()
        if not require_existing_managed or managed.is_dir():
            return ReferenceLocation(managed, SOURCE_MANAGED)

    return ReferenceLocation(None, None)


def reference_source_hint() -> str:
    """Return the standard actionable hint for a missing reference directory."""
    return (
        "Provide reference_dir=..., set the "
        f"{REFERENCE_DIR_ENV_VAR} environment variable, or install managed "
        "reference data explicitly with locusblend.install_reference(ancestry=...)."
    )

