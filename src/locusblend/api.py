"""Future public API surface for LocusBlend.

This module is a placeholder for the orchestration that will eventually drive
the whole pipeline (data loading -> reference matching -> LD -> figures).
Implementing that pipeline by copying the baseline app's ``main`` block is
explicitly out of scope for this first pass, so the entry points below raise
``NotImplementedError`` until the extracted components are verified.

What already works today, without Streamlit, is the component level:

* ``locusblend.io`` - summary-statistic loading and normalization
* ``locusblend.reference`` - external reference-data path resolution
* ``locusblend.variants`` / ``locusblend.ld`` - matching, LD and clumping
* ``locusblend.genes`` / ``locusblend.plotting`` / ``locusblend.compare``
* ``locusblend.export`` - PNG/PDF rendering helpers
"""

from __future__ import annotations

from typing import Optional

from .config import LocusBlendConfig
from .models import LocusBlendResult
from .reference import (
    INTERNAL_1000G_ANCESTRY_OPTIONS,
    INTERNAL_1000G_DEFAULT_ANCESTRY,
    ReferenceManager,
)

__all__ = [
    "LocusBlendConfig",
    "LocusBlendResult",
    "available_ancestries",
    "default_reference_manager",
    "plot_compare",
    "plot_locus",
]


def available_ancestries():
    """Return the supported 1000G super-population codes (AFR, AMR, EAS, EUR, SAS)."""
    return tuple(INTERNAL_1000G_ANCESTRY_OPTIONS)


def default_reference_manager(
    reference_dir: Optional[str] = None,
    ancestry: str = INTERNAL_1000G_DEFAULT_ANCESTRY,
) -> ReferenceManager:
    """Build a :class:`~locusblend.reference.ReferenceManager`.

    Thin convenience wrapper: ``reference_dir=None`` falls back to the
    ``LOCUSBLEND_REFERENCE_DIR`` environment variable.
    """
    return ReferenceManager(reference_dir=reference_dir, ancestry=ancestry)


def plot_locus(config: LocusBlendConfig, reference: Optional[ReferenceManager] = None):
    """Render a single-locus LocusBlend figure and return a :class:`LocusBlendResult`.

    Not implemented in this refactoring pass. Until the orchestration pass
    lands, use the component modules directly (see the module docstring).
    """
    raise NotImplementedError(
        "locusblend.api.plot_locus is a placeholder. The pipeline orchestration "
        "is implemented in a later refactoring pass; this pass extracts and "
        "verifies the reusable components only."
    )


def plot_compare(config: LocusBlendConfig, reference: Optional[ReferenceManager] = None):
    """Render a Locus Compare figure and return a :class:`LocusBlendResult`.

    Not implemented in this refactoring pass; see :func:`plot_locus`.
    """
    raise NotImplementedError(
        "locusblend.api.plot_compare is a placeholder. The pipeline orchestration "
        "is implemented in a later refactoring pass; this pass extracts and "
        "verifies the reusable components only."
    )

