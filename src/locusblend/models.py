"""Result models for the future public LocusBlend API.

This first pass only sketches the container types: the orchestration that fills
them in is implemented in a later refactoring pass, after the extracted
components have been verified against ``app.2.8.12.py``.
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
    """Initial result container for a LocusBlend run.

    Not produced by any function yet: ``api.plot_locus`` is a placeholder until
    the orchestration pass. The window bounds and SNP count mirror the return
    values of ``plotting.get_plotly_locus_py``.
    """

    locus_fig: Any = None
    compare_fig: Any = None
    chromosome: Optional[str] = None
    center_bp: Optional[int] = None
    window_kb: Optional[float] = None
    window_start_bp: Optional[int] = None
    window_end_bp: Optional[int] = None
    n_window_snps: Optional[int] = None
    selected_index_variants: Tuple[IndexVariant, ...] = ()
    ld_metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()

