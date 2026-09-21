"""LocusBlend: reusable, Streamlit-independent core for regional GWAS visualization.

This package holds the reusable logic extracted from the single-file baseline
``app.2.8.12.py``. Importing :mod:`locusblend` must never import Streamlit: the
Streamlit application will consume these modules in a later refactoring pass.

Public entry points: :func:`locusblend.plot` (internal local 1000G reference
workflow) and :func:`locusblend.reference_status` (inspect a reference directory
you prepared yourself). Reference data are never bundled or downloaded: point
the package at them with ``reference_dir=...`` or ``LOCUSBLEND_REFERENCE_DIR``.
Everything here is a work in progress and the public API is not stable yet.
"""

from . import (
    api,
    colors,
    compare,
    config,
    export,
    genes,
    io,
    ld,
    models,
    paths,
    plotting,
    reference,
    variants,
)
from .api import plot
from .config import LocusBlendConfig
from .models import IndexVariant, LocusBlendResult
from .reference import (
    ReferenceManager,
    ReferenceStatus,
    ReferenceValidation,
    reference_status,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "IndexVariant",
    "LocusBlendConfig",
    "LocusBlendResult",
    "ReferenceManager",
    "ReferenceStatus",
    "ReferenceValidation",
    "__version__",
    "api",
    "colors",
    "compare",
    "config",
    "export",
    "genes",
    "io",
    "ld",
    "models",
    "paths",
    "plot",
    "plotting",
    "reference",
    "reference_status",
    "variants",
]
