"""Result models for the public LocusBlend API.

``locusblend.plot`` returns a populated :class:`LocusBlendResult`; the
container types are deliberately plain dataclasses (no serialization layer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class IndexVariant:
    """A selected index variant (manual input or auto-selected by clumping).

    Field names follow the columns produced by
    ``variants.auto_select_index_variants_by_clumping`` /
    ``variants.resolve_index_variant_from_input``:
    ``rank``, ``DISPLAY_ID``, ``REF_SNP``, ``CHR``, ``BP``, ``P``.
    """

    rank: int = 1
    display_id: str = ""
    ref_snp: Optional[str] = None
    chrom: Optional[str] = None
    bp: Optional[int] = None
    p: Optional[float] = None
    source: Optional[str] = None  # "top" or "bottom" dataset


@dataclass
class LocusBlendResult:
    """Result of a LocusBlend run.

    Figure fields hold Plotly figures (the interactive objects returned to the
    caller); ``dataset*_processed`` hold the reference-matched, LD-annotated
    per-dataset DataFrames that the figures were built from.
    """

    # figures
    locus_figure: Any = None
    compare_figure: Any = None

    # locus definition
    chromosome: Optional[str] = None
    center_bp: Optional[int] = None
    window_kb: Optional[float] = None
    window_start_bp: Optional[int] = None
    window_end_bp: Optional[int] = None
    mode: Optional[str] = None
    ancestry: Optional[str] = None

    # index variants
    index_variants: Tuple[str, ...] = ()
    index_reference_snps: Tuple[Optional[str], ...] = ()
    selected_index_variants: Tuple[IndexVariant, ...] = ()

    # processed data / annotation
    dataset1_processed: Any = None
    dataset2_processed: Any = None
    genes: Any = None
    ld_status: Dict[str, Any] = field(default_factory=dict)
    n_window_snps: Optional[int] = None

    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()

    @property
    def locus_fig(self):
        """Alias for :attr:`locus_figure` (baseline naming)."""
        return self.locus_figure

    @property
    def compare_fig(self):
        """Alias for :attr:`compare_figure` (baseline naming)."""
        return self.compare_figure

    window_start_bp: Optional[int] = None
    window_end_bp: Optional[int] = None
    n_window_snps: Optional[int] = None
    selected_index_variants: Tuple[IndexVariant, ...] = ()
    ld_metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()
