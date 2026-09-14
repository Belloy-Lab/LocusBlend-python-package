"""LocusBlend: reusable, Streamlit-independent core for regional GWAS visualization.

This package holds the reusable logic extracted from the single-file baseline
``app.2.8.12.py``. Importing :mod:`locusblend` must never import Streamlit: the
Streamlit application will consume these modules in a later refactoring pass,
and a standalone Python API will be built on top of them.

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
    plotting,
    reference,
    variants,
)
from .config import LocusBlendConfig
from .models import IndexVariant, LocusBlendResult
from .reference import ReferenceManager

__version__ = "0.1.0.dev0"

__all__ = [
    "IndexVariant",
    "LocusBlendConfig",
    "LocusBlendResult",
    "ReferenceManager",
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
    "plotting",
    "reference",
    "variants",
]

