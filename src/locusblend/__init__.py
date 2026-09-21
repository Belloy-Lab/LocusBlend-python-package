"""LocusBlend: reusable, Streamlit-independent core for regional GWAS visualization.

This package holds the reusable logic extracted from the single-file baseline
``app.2.8.12.py``. Importing :mod:`locusblend` must never import Streamlit: the
Streamlit application will consume these modules in a later refactoring pass.

Public entry points: :func:`locusblend.plot` (internal local 1000G reference
workflow), :func:`locusblend.install_reference` (explicit reference-data
installation - never triggered by ``plot()``) and
:func:`locusblend.reference_status`. Everything here is a work in progress and
the public API is not stable yet.
"""

from . import (
    api,
    colors,
    compare,
    config,
    export,
    fetch,
    genes,
    install,
    io,
    ld,
    manifest,
    models,
    paths,
    plotting,
    reference,
    variants,
)
from .api import plot
from .config import LocusBlendConfig
from .fetch import ChecksumError, DownloadError
from .install import (
    InstallResult,
    ReferenceInstallError,
    ReferenceStatus,
    install_reference,
    reference_status,
)
from .manifest import ManifestError, ReferenceManifest
from .models import IndexVariant, LocusBlendResult
from .paths import get_default_reference_dir
from .reference import ReferenceManager

__version__ = "0.1.0.dev0"

__all__ = [
    "ChecksumError",
    "DownloadError",
    "IndexVariant",
    "InstallResult",
    "LocusBlendConfig",
    "LocusBlendResult",
    "ManifestError",
    "ReferenceInstallError",
    "ReferenceManager",
    "ReferenceManifest",
    "ReferenceStatus",
    "__version__",
    "api",
    "colors",
    "compare",
    "config",
    "export",
    "fetch",
    "genes",
    "get_default_reference_dir",
    "install",
    "install_reference",
    "io",
    "ld",
    "manifest",
    "models",
    "paths",
    "plot",
    "plotting",
    "reference",
    "reference_status",
    "variants",
]
