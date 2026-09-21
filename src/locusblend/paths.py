"""Reference-data location resolution for LocusBlend.

LocusBlend does not bundle or download reference data: you prepare a reference
directory yourself and point the package at it. Resolution order:

1. an explicit ``reference_dir=...`` argument
2. the ``LOCUSBLEND_REFERENCE_DIR`` environment variable

If neither is set, nothing is resolved and callers report the missing
configuration (``ReferenceManager`` validation and ``reference_status()`` do
this) rather than guessing a location. Nothing in this module touches the
network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: Environment variable pointing at a prepared reference directory.
REFERENCE_DIR_ENV_VAR = "LOCUSBLEND_REFERENCE_DIR"

#: Possible outcomes of reference-dir resolution.
SOURCE_EXPLICIT = "explicit"
SOURCE_ENV = "env"


def reference_dir_from_env() -> Optional[Path]:
    """Return ``LOCUSBLEND_REFERENCE_DIR`` as a path, or None when unset."""
    value = os.environ.get(REFERENCE_DIR_ENV_VAR, "").strip()
    return Path(value).expanduser() if value else None


@dataclass(frozen=True)
class ReferenceLocation:
    """Result of :func:`resolve_reference_dir`.

    ``source`` is ``"explicit"``, ``"env"`` or ``None`` (nothing configured);
    ``path`` is None when no source applied.
    """

    path: Optional[Path] = None
    source: Optional[str] = None

    @property
    def is_resolved(self) -> bool:
        return self.path is not None


def resolve_reference_dir(reference_dir=None) -> ReferenceLocation:
    """Resolve a reference directory: explicit argument, then environment."""
    if reference_dir is not None and str(reference_dir).strip() != "":
        return ReferenceLocation(Path(reference_dir).expanduser(), SOURCE_EXPLICIT)

    env_dir = reference_dir_from_env()
    if env_dir is not None:
        return ReferenceLocation(env_dir, SOURCE_ENV)

    return ReferenceLocation(None, None)


def reference_source_hint() -> str:
    """Return the standard actionable hint for a missing reference directory."""
    return (
        "Provide reference_dir=... or set the "
        f"{REFERENCE_DIR_ENV_VAR} environment variable."
    )
