"""Public Python API for LocusBlend (internal local 1000G reference workflow).

This module implements the first standalone pipeline::

    import locusblend

    result = locusblend.plot(
        dataset1="trait1.tsv.gz",
        dataset2="trait2.tsv.gz",
        reference_dir=r"D:/locusblend_reference",
        ancestry="EUR",
        mode="three",
        output="locusblend.png",
    )

It follows the baseline ``app.2.8.12.py`` main pipeline for the internal 1000G
reference only:

1. read both summary-statistic files with the app's format detection
2. infer chromosome / center BP when they are not supplied
3. resolve the 1000G PLINK bfile, GENCODE annotation and recombination track
   through :class:`~locusblend.reference.ReferenceManager`
4. load the chromosome BIM and reference-match both datasets (two-pass
   forward / allele-flip matching)
5. resolve index variants: manual input, or automatic LD clumping
6. compute LD with PLINK and merge the LD annotation into both datasets
7. build both locus panels (``plotting.get_plotly_locus_py``)
8. load the GENCODE gene track
9. assemble the three-row combined locus figure
10. build the locus compare figure (baseline default: three separate panels)
11. optionally write the combined locus figure to a PNG file

Uploaded-LD orchestration is intentionally not part of the public API in this
pass; the uploaded-LD helpers stay available in :mod:`locusblend.ld`.

Reference data (1000G PLINK panels, GENCODE GTF, recombination BigWig) and
PLINK itself live outside this repository and are never downloaded by the
package. The pipeline is GRCh38/hg38 only; no liftover is performed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .compare import apply_locus_compare_safe_autoscale, build_compare_figure_triptych
from .config import (
    AUTO_INDEX_SOURCE_DATASET1,
    AUTO_INDEX_SOURCE_DATASET2,
    COMPARE_MODE_SEPARATE,
    DEFAULT_CLUMP_R2,
    DEFAULT_COMBINED_HEIGHT,
    DEFAULT_COMPARE_SIZE,
    DEFAULT_GENE_TRACK_GAP,
    DEFAULT_MIN_YLIM,
    DEFAULT_RECOMB_MAX,
    DEFAULT_VERTICAL_SPACING,
    DEFAULT_WINDOW_KB,
    DEFAULT_YLIM_PAD_FRAC,
    GENE_DISPLAY_MODES,
    GENE_DISPLAY_PROTEIN_CODING,
    INDEX_SELECTION_AUTO_BOTTOM,
    INDEX_SELECTION_AUTO_TOP,
    INDEX_SELECTION_MANUAL,
    LD_SOURCE_INTERNAL_1000G,
    LocusBlendConfig,
    MODE_STANDARD,
    MODE_THREE_INDEX,
    get_required_n_indices,
    normalize_auto_index_source,
    normalize_public_mode,
    parse_highlighted_genes,
)
from .export import render_plotly_figure_to_png_bytes
from .genes import load_genes_from_gtf, load_genes_from_table
from .io import (
    _clean_locus_df,
    choose_sync_center_bp,
    choose_sync_chromosome,
    chrom_mask,
    format_chrom_label,
    is_supported_chrom,
    load_locus_path,
    make_dataset_title_from_filename,
    normalize_chrom,
    recommend_y_axis_max_for_dataset,
    summarize_locus_dataset,
)
from .ld import (
    build_ld_annot_for_window,
    compute_ld_maps_with_plink,
    load_reference_bim,
    merge_ld_annot,
)
from .models import IndexVariant, LocusBlendResult
from .plotting import (
    apply_locusblend_plot_theme,
    build_combined_locus_figure,
    get_ld_reference_labels,
    get_plotly_locus_py,
)
from .reference import (
    INTERNAL_1000G_ANCESTRY_OPTIONS,
    INTERNAL_1000G_DEFAULT_ANCESTRY,
    ReferenceManager,
)
from .variants import (
    attach_reference_snp_two_pass,
    auto_select_index_variants_by_clumping,
    resolve_index_variant_from_input,
)

__all__ = ["available_ancestries", "default_reference_manager", "plot"]

# PNG rendering defaults. The combined locus figure sets its own height
# (baseline: 980) but no width, so a fixed development-stage width is used.
DEFAULT_PNG_WIDTH_PX = 1000
DEFAULT_PNG_HEIGHT_PX = DEFAULT_COMBINED_HEIGHT

# auto_index_source -> (baseline source label, baseline index-selection method)
_AUTO_SOURCE_INFO = {
    AUTO_INDEX_SOURCE_DATASET1: ("dataset 1 (top)", INDEX_SELECTION_AUTO_TOP),
    AUTO_INDEX_SOURCE_DATASET2: ("dataset 2 (bottom)", INDEX_SELECTION_AUTO_BOTTOM),
}


def available_ancestries():
    """Return the supported 1000G super-population codes (AFR, AMR, EAS, EUR, SAS)."""
    return tuple(INTERNAL_1000G_ANCESTRY_OPTIONS)


def default_reference_manager(
    reference_dir: Optional[str] = None,
    ancestry: str = INTERNAL_1000G_DEFAULT_ANCESTRY,
) -> ReferenceManager:
    """Build a :class:`~locusblend.reference.ReferenceManager`.

    ``reference_dir=None`` falls back to the ``LOCUSBLEND_REFERENCE_DIR``
    environment variable.
    """
    return ReferenceManager(reference_dir=reference_dir, ancestry=ancestry)


def _safe_str(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _safe_int(value):
    text = _safe_str(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _safe_float(value):
    text = _safe_str(value)
    if text is None:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _load_dataset(dataset, label):
    """Return ``(cleaned_dataframe, title_hint)`` for a path or DataFrame input.

    Paths go through ``io.load_locus_path`` (the app's uploaded-file parsing
    behavior, which also accepts .tsv/.txt/.gz); DataFrames are normalized with
    the same ``_clean_locus_df`` used for parsed files, so both input kinds
    behave identically downstream.
    """
    if isinstance(dataset, pd.DataFrame):
        return _clean_locus_df(dataset.copy(), source_name=label), None

    if isinstance(dataset, (str, Path)) or hasattr(dataset, "__fspath__"):
        path = Path(dataset)
        return load_locus_path(path), make_dataset_title_from_filename(path.name)

    raise TypeError(
        f"{label} must be a filesystem path or a pandas.DataFrame, "
        f"got {type(dataset).__name__}."
    )


def _normalize_index_variants_input(index_variants):
    """Return the user-supplied index variants as a list of stripped strings."""
    if isinstance(index_variants, str):
        texts = [index_variants]
    else:
        try:
            texts = list(index_variants)
        except TypeError as exc:
            raise TypeError(
                "index_variants must be a variant id string or a sequence of "
                "variant id strings."
            ) from exc
    return [str(text).strip() for text in texts]


def _index_variant_records(table, source):
    """Build :class:`IndexVariant` records from an auto-selection table."""
    records = []
    for position, (_, row) in enumerate(table.iterrows(), start=1):
        rank = _safe_int(row.get("rank", position)) or position
        records.append(
            IndexVariant(
                rank=rank,
                display_id=_safe_str(row.get("DISPLAY_ID")) or "",
                ref_snp=_safe_str(row.get("REF_SNP")),
                chrom=(normalize_chrom(row.get("CHR")) if _safe_str(row.get("CHR")) else None),
                bp=_safe_int(row.get("BP")),
                p=_safe_float(row.get("P")),
                source=source,
            )
        )
    return tuple(records)


def _ld_status_caption(index_status, mode_label, ancestry, chrom):
    """Rebuild the baseline's LD status caption string."""
    source_name = f"1000G {ancestry} LD"
    parts = [
        f"variant 1: {('found in ' + source_name) if index_status['variant 1'] else ('not in ' + source_name + ' -> grey X')}",
    ]
    if mode_label != MODE_STANDARD:
        parts.append(
            f"variant 2: {('found in ' + source_name) if index_status['variant 2'] else ('not in ' + source_name + ' -> grey X')}",
        )
    if mode_label == MODE_THREE_INDEX:
        parts.append(
            f"variant 3: {('found in ' + source_name) if index_status['variant 3'] else ('not in ' + source_name + ' -> grey X')}",
        )
    caption = " | ".join(parts)
    caption += f"  |  Chromosome: {format_chrom_label(chrom)}"
    caption += f"  |  Ref: 1000G {ancestry} {format_chrom_label(chrom)}"
    return caption


def plot(
    dataset1,
    dataset2,
    *,
    reference_dir,
    ancestry=INTERNAL_1000G_DEFAULT_ANCESTRY,
    mode="three",
    chrom=None,
    center_bp=None,
    window_kb=DEFAULT_WINDOW_KB,
    index_variants=None,
    auto_index_source=AUTO_INDEX_SOURCE_DATASET1,
    clump_r2=DEFAULT_CLUMP_R2,
    plink_path=None,
    title1=None,
    title2=None,
    gene_display_mode=GENE_DISPLAY_PROTEIN_CODING,
    highlight_genes=None,
    show_recombination=True,
    output=None,
):
    """Build LocusBlend locus + compare figures from two summary-statistic datasets.

    Parameters
    ----------
    dataset1, dataset2:
        Summary statistics as a filesystem path (.csv, .tsv, .txt and their
        .gz variants) or as an already-loaded ``pandas.DataFrame``. Files use
        the same parsing/aliasing as the web app; GRCh38/hg38 coordinates are
        required (no liftover is performed).
    reference_dir:
        External reference directory holding ``1000g/<ancestry>`` PLINK
        panels, ``gencode`` annotations and the ``recombination`` BigWig. When
        ``None``, the ``LOCUSBLEND_REFERENCE_DIR`` environment variable is used.
    ancestry:
        One of ``AFR``, ``AMR``, ``EAS``, ``EUR`` (default), ``SAS``.
    mode:
        Public mode value: ``"standard"`` (1 index variant), ``"two"`` (2) or
        ``"three"`` (3, default). Mapped internally to the baseline mode names.
    chrom, center_bp:
        Locus definition. Anything omitted is inferred from the datasets with
        the same helpers the app uses for upload-driven sync.
    window_kb:
        Half-window size in kb (baseline default 500).
    index_variants:
        ``None`` (default) selects index variants automatically by PLINK LD
        clumping from ``auto_index_source``. Otherwise pass one variant id or a
        list/tuple whose length matches what ``mode`` requires; ids are
        resolved through the baseline's index-variant matching.
    auto_index_source:
        ``"dataset1"`` (default) or ``"dataset2"`` - which dataset the
        automatic clumping runs on.
    clump_r2:
        Clumping r^2 threshold (baseline default ``0.01``).
    plink_path:
        Optional explicit PLINK executable path. Resolution order is
        ``plink_path``, ``LOCUSBLEND_PLINK``, then the system PATH.
    title1, title2:
        Dataset titles. Defaults: the file name (without extension) for path
        inputs, ``"Dataset 1"`` / ``"Dataset 2"`` for DataFrame inputs.
    gene_display_mode:
        ``"protein_coding"`` (baseline default) or ``"all"``.
    highlight_genes:
        Gene names to highlight in the gene track; a comma/semicolon/whitespace
        separated string or any iterable of names (case-insensitive).
    show_recombination:
        Draw the recombination-rate overlay when the reference BigWig exists.
        A missing BigWig is not an error: the overlay is skipped (baseline
        best-effort behavior).
    output:
        Optional PNG path for the combined locus figure. ``None`` writes
        nothing. Static rendering needs Plotly's kaleido (and, depending on the
        kaleido version, Chrome); a missing dependency raises the existing
        clear RuntimeError.

    Returns
    -------
    LocusBlendResult
        Plotly figures plus the locus definition, index variants, processed
        datasets, genes, LD status and run metadata.
    """
    mode_label = normalize_public_mode(mode)
    auto_source = normalize_auto_index_source(auto_index_source)
    if gene_display_mode not in GENE_DISPLAY_MODES:
        raise ValueError(
            f"Unsupported gene_display_mode {gene_display_mode!r}. Use one of: "
            + ", ".join(repr(v) for v in GENE_DISPLAY_MODES)
            + "."
        )

    required_n = get_required_n_indices(mode_label)
    window_kb = float(window_kb)
    window_bp = int(window_kb * 1000)
    clump_r2 = float(clump_r2)
    warnings = []

    # ------------------------------------------------------------------
    # 1. datasets
    # ------------------------------------------------------------------
    df1, title_hint1 = _load_dataset(dataset1, "Dataset 1")
    df2, title_hint2 = _load_dataset(dataset2, "Dataset 2")

    title_top = str(title1) if title1 is not None else (title_hint1 or "Dataset 1")
    title_bottom = str(title2) if title2 is not None else (title_hint2 or "Dataset 2")

    # ------------------------------------------------------------------
    # 2. locus definition (supplied, or inferred with the baseline helpers)
    # ------------------------------------------------------------------
    if chrom is not None and str(chrom).strip() != "":
        selected_chrom = normalize_chrom(chrom)
    else:
        top_summary = summarize_locus_dataset(df1, "Dataset 1")
        bottom_summary = summarize_locus_dataset(df2, "Dataset 2")
        inferred_chrom, _chrom_source = choose_sync_chromosome(top_summary, bottom_summary)
        if inferred_chrom is None:
            raise ValueError(
                "Could not infer a chromosome from the supplied datasets. "
                "Pass chrom=... (supported chromosomes are 1-22 and X)."
            )
        selected_chrom = normalize_chrom(inferred_chrom)

    if not is_supported_chrom(selected_chrom):
        raise ValueError(
            f"Unsupported chromosome {selected_chrom}. Supported chromosomes are 1-22 and X."
        )

    if center_bp is not None:
        center = int(center_bp)
    else:
        inferred_bp, _bp_source = choose_sync_center_bp(df1, df2, selected_chrom)
        if inferred_bp is None:
            raise ValueError(
                f"Could not infer a center BP on chromosome {selected_chrom} from the "
                "supplied datasets. Pass center_bp=..."
            )
        center = int(inferred_bp)

    top_mask = chrom_mask(df1, selected_chrom)
    bottom_mask = chrom_mask(df2, selected_chrom)
    if not top_mask.any() and not bottom_mask.any():
        raise ValueError(
            f"No rows found for chromosome {selected_chrom} in either dataset. "
            "Check the chromosome selection or the input data."
        )

    # Filter to the selected chromosome for the LD pipeline (baseline step).
    df1_chr = df1.loc[top_mask].copy()
    df2_chr = df2.loc[bottom_mask].copy()

    # ------------------------------------------------------------------
    # 3. reference wiring (all reference paths come from ReferenceManager)
    # ------------------------------------------------------------------
    ref = ReferenceManager(reference_dir=reference_dir, ancestry=ancestry)
    ancestry_code = ref.ancestry

    ld_bfile_prefix = ref.get_bfile_prefix(selected_chrom)
    gtf_path = ref.get_gtf_path(selected_chrom)
    recombination_bw = ref.get_recombination_bw_path()
    # Pass the configured path (not an environment-resolved one) so a missing
    # file simply skips the overlay instead of using a different directory.
    recombination_bw_arg = (
        recombination_bw if recombination_bw is not None else str(ref.recombination_bw_path())
    )
    if show_recombination and recombination_bw is None:
        warnings.append(
            "Recombination BigWig not found at "
            f"{recombination_bw_arg}; the recombination overlay is skipped."
        )

    ld_source = LD_SOURCE_INTERNAL_1000G
    ld_labels = get_ld_reference_labels(ld_source, ancestry_code)

    # ------------------------------------------------------------------
    # 4. reference matching (two-pass forward / allele flip)
    # ------------------------------------------------------------------
    bim_ref = load_reference_bim(ld_bfile_prefix)
    df1_ref = attach_reference_snp_two_pass(df1_chr, bim_ref)
    df2_ref = attach_reference_snp_two_pass(df2_chr, bim_ref)

    # ------------------------------------------------------------------
    # 5. index variants (manual input or automatic LD clumping)
    # ------------------------------------------------------------------
    auto_summary = None
    auto_source_label = None
    if index_variants is None:
        source_ref = df1_ref if auto_source == AUTO_INDEX_SOURCE_DATASET1 else df2_ref
        source_label, index_selection_method = _AUTO_SOURCE_INFO[auto_source]

        auto_selected, auto_summary = auto_select_index_variants_by_clumping(
            df_source_ref=source_ref,
            selected_chrom=selected_chrom,
            center_bp=center,
            window_bp=window_bp,
            required_n=required_n,
            active_ld_source=ld_source,
            bfile_prefix=ld_bfile_prefix,
            plink_path=plink_path,
            clump_r2=clump_r2,
            active_locusblend_mode=mode_label,
        )
        auto_source_label = source_label

        index_labels = []
        index_refs = []
        for position in range(required_n):
            if position < len(auto_selected):
                index_labels.append(str(auto_selected.iloc[position]["DISPLAY_ID"]))
                index_refs.append(auto_selected.iloc[position]["REF_SNP"])
            else:
                index_labels.append("")
                index_refs.append(None)

        index_records = _index_variant_records(auto_selected, source_label)
    else:
        texts = _normalize_index_variants_input(index_variants)
        if len(texts) != required_n:
            raise ValueError(
                f"mode={mode!r} requires exactly {required_n} index variant(s); "
                f"got {len(texts)}."
            )
        index_selection_method = INDEX_SELECTION_MANUAL

        index_labels = list(texts)
        index_refs = []
        records = []
        for rank, text in enumerate(texts, start=1):
            row = resolve_index_variant_from_input(df1_ref, df2_ref, text)
            ref_snp = _safe_str(row.get("REF_SNP"))
            # unresolved reference ids stay None (never pandas.NA) so callers
            # can rely on a plain None check
            index_refs.append(ref_snp)
            records.append(
                IndexVariant(
                    rank=rank,
                    display_id=text,
                    ref_snp=ref_snp,
                    chrom=(normalize_chrom(row.get("CHR")) if _safe_str(row.get("CHR")) else None),
                    bp=_safe_int(row.get("BP")),
                    p=_safe_float(row.get("P")),
                    source=_safe_str(row.get("SOURCE")),
                )
            )
        index_records = tuple(records)

    idx1_ref, idx2_ref, idx3_ref = (list(index_refs) + [None, None, None])[:3]
    idx1_label, idx2_label, idx3_label = (list(index_labels) + ["", "", ""])[:3]

    # ------------------------------------------------------------------
    # 6. LD within the locus window
    # ------------------------------------------------------------------
    ld_bp_start = int(center - window_bp)
    ld_bp_end = int(center + window_bp)

    top_window = df1_ref[
        chrom_mask(df1_ref, selected_chrom)
        & (df1_ref["BP"] >= ld_bp_start)
        & (df1_ref["BP"] <= ld_bp_end)
    ][["REF_SNP", "CHR", "BP"]].copy()

    bottom_window = df2_ref[
        chrom_mask(df2_ref, selected_chrom)
        & (df2_ref["BP"] >= ld_bp_start)
        & (df2_ref["BP"] <= ld_bp_end)
    ][["REF_SNP", "CHR", "BP"]].copy()

    window_union = pd.concat([top_window, bottom_window], ignore_index=True).drop_duplicates("REF_SNP")
    window_snps = tuple(sorted(window_union["REF_SNP"].dropna().astype(str).unique().tolist()))

    ld_maps, index_status, ref_snps = compute_ld_maps_with_plink(
        bfile_prefix=ld_bfile_prefix,
        chrom=selected_chrom,
        start=ld_bp_start,
        end=ld_bp_end,
        window_snps=window_snps,
        idx1_ref=idx1_ref,
        idx2_ref=idx2_ref,
        idx3_ref=idx3_ref,
        plink_path=plink_path,
    )

    ld_annot = build_ld_annot_for_window(
        window_union_df=window_union,
        ld_maps=ld_maps,
        ref_snps=ref_snps,
        idx1_ref=idx1_ref,
        idx2_ref=idx2_ref,
        idx3_ref=idx3_ref,
    )

    df1_plot = merge_ld_annot(df1_ref, ld_annot)
    df2_plot = merge_ld_annot(df2_ref, ld_annot)

    ld_status_caption = _ld_status_caption(index_status, mode_label, ancestry_code, selected_chrom)

    # ------------------------------------------------------------------
    # 7. per-dataset locus panels
    # ------------------------------------------------------------------
    max_ylim_top = recommend_y_axis_max_for_dataset(
        df1_plot,
        chrom=selected_chrom,
        center_bp=center,
        window_kb=window_kb,
        min_default=DEFAULT_MIN_YLIM,
        pad_frac=DEFAULT_YLIM_PAD_FRAC,
    )
    max_ylim_bottom = recommend_y_axis_max_for_dataset(
        df2_plot,
        chrom=selected_chrom,
        center_bp=center,
        window_kb=window_kb,
        min_default=DEFAULT_MIN_YLIM,
        pad_frac=DEFAULT_YLIM_PAD_FRAC,
    )

    fig_top, n_top, bp_start_top, bp_end_top = get_plotly_locus_py(
        max_ylim=max_ylim_top,
        bp=center,
        window_bp=window_bp,
        merged_female_withld=df1_plot,
        merged_df=df1_plot,
        idx1_ref=idx1_ref,
        idx2_ref=idx2_ref,
        idx3_ref=idx3_ref,
        idx1_label=idx1_label,
        idx2_label=idx2_label,
        idx3_label=idx3_label,
        y_label=title_top,
        ld_labels=ld_labels,
        chrom=selected_chrom,
        show_recomb=bool(show_recombination),
        bw_path=recombination_bw_arg,
        locusblend_mode=mode_label,
    )

    fig_bottom, n_bottom, bp_start_bottom, bp_end_bottom = get_plotly_locus_py(
        max_ylim=max_ylim_bottom,
        bp=center,
        window_bp=window_bp,
        merged_female_withld=df2_plot,
        merged_df=df2_plot,
        idx1_ref=idx1_ref,
        idx2_ref=idx2_ref,
        idx3_ref=idx3_ref,
        idx1_label=idx1_label,
        idx2_label=idx2_label,
        idx3_label=idx3_label,
        y_label=title_bottom,
        ld_labels=ld_labels,
        chrom=selected_chrom,
        show_recomb=bool(show_recombination),
        bw_path=recombination_bw_arg,
        locusblend_mode=mode_label,
    )

    # ------------------------------------------------------------------
    # 8. GENCODE gene track
    # ------------------------------------------------------------------
    gene_bp_start = min(bp_start_top, bp_start_bottom)
    gene_bp_end = max(bp_end_top, bp_end_bottom)

    if str(gtf_path).endswith(".parquet"):
        genes_df = load_genes_from_table(
            chrom=selected_chrom,
            start=gene_bp_start,
            end=gene_bp_end,
            parquet_path=gtf_path,
            gene_display_mode=gene_display_mode,
        )
    else:
        genes_df = load_genes_from_gtf(
            gtf_path=gtf_path,
            chrom=selected_chrom,
            start=gene_bp_start,
            end=gene_bp_end,
            gene_display_mode=gene_display_mode,
        )

    # ------------------------------------------------------------------
    # 9. combined three-row locus figure
    # ------------------------------------------------------------------
    locus_figure, gene_track_rows = build_combined_locus_figure(
        fig_top,
        fig_bottom,
        genes_df,
        chrom=selected_chrom,
        gene_bp_start=gene_bp_start,
        gene_bp_end=gene_bp_end,
        title_top=title_top,
        title_bottom=title_bottom,
        gene_display_mode=gene_display_mode,
        highlight_names=parse_highlighted_genes(highlight_genes),
        gene_track_gap=DEFAULT_GENE_TRACK_GAP,
        combined_height=DEFAULT_COMBINED_HEIGHT,
        vertical_spacing=DEFAULT_VERTICAL_SPACING,
        recomb_max=DEFAULT_RECOMB_MAX,
    )

    # ------------------------------------------------------------------
    # 10. locus compare figure (baseline default: separate panels)
    # ------------------------------------------------------------------
    active_signals = ["variant 1"]
    if mode_label != MODE_STANDARD:
        active_signals.append("variant 2")
    if mode_label == MODE_THREE_INDEX:
        active_signals.append("variant 3")

    compare_figure, compare_counts = build_compare_figure_triptych(
        df_top=df1_plot,
        df_bottom=df2_plot,
        idx1_label=idx1_label,
        idx2_label=idx2_label,
        idx3_label=idx3_label,
        title_top=title_top,
        title_bottom=title_bottom,
        compare_size=DEFAULT_COMPARE_SIZE,
        ld_labels=ld_labels,
        active_signals=active_signals,
    )
    compare_figure = apply_locusblend_plot_theme(compare_figure)
    compare_figure = apply_locus_compare_safe_autoscale(compare_figure)

    # ------------------------------------------------------------------
    # summary metrics mirroring the app
    # ------------------------------------------------------------------
    n_flip_top = (
        int(df1_plot["coding_flipped"].fillna(False).sum())
        if "coding_flipped" in df1_plot.columns
        else 0
    )
    n_flip_bottom = (
        int(df2_plot["coding_flipped"].fillna(False).sum())
        if "coding_flipped" in df2_plot.columns
        else 0
    )

    compare_parts = [f"v1: {compare_counts.get('variant 1', 0):,}"]
    if mode_label != MODE_STANDARD:
        compare_parts.append(f"v2: {compare_counts.get('variant 2', 0):,}")
    if mode_label == MODE_THREE_INDEX:
        compare_parts.append(f"v3: {compare_counts.get('variant 3', 0):,}")
    compare_summary = f"Compare points ({'/'.join(compare_parts)})"

    summary_text = (
        f"Dataset 1 SNP count: {n_top:,} | "
        f"Dataset 2 SNP count: {n_bottom:,} | "
        f"Genes shown ({gene_display_mode}): {len(genes_df):,} | "
        f"{compare_summary} | "
        f"{ld_labels['summary_label']}: {len(ref_snps):,} | "
        f"Coding flipped (dataset 1/dataset 2): {n_flip_top:,}/{n_flip_bottom:,}"
    )

    # ------------------------------------------------------------------
    # 11. optional PNG of the combined locus figure
    # ------------------------------------------------------------------
    output_path = None
    if output is not None:
        output_path = Path(output)
        if str(output_path.parent) not in ("", "."):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        width_px = (
            int(locus_figure.layout.width) if locus_figure.layout.width else DEFAULT_PNG_WIDTH_PX
        )
        height_px = (
            int(locus_figure.layout.height) if locus_figure.layout.height else DEFAULT_PNG_HEIGHT_PX
        )
        png_bytes = render_plotly_figure_to_png_bytes(locus_figure, width_px, height_px, scale=1)
        output_path.write_bytes(png_bytes)

    config = LocusBlendConfig(
        mode=mode_label,
        ancestry=ancestry_code,
        chromosome=selected_chrom,
        center_bp=center,
        window_kb=window_kb,
        index_variants=tuple(index_labels),
        index_selection_method=index_selection_method,
        auto_index_source=(auto_source if index_variants is None else None),
        clump_r2=clump_r2,
        ld_source=ld_source,
        gene_display_mode=gene_display_mode,
        highlighted_genes=(highlight_genes if isinstance(highlight_genes, str) else ""),
        show_recombination=bool(show_recombination),
        title_top=title_top,
        title_bottom=title_bottom,
    )

    ld_status = {
        "source": ld_source,
        "ancestry": ancestry_code,
        "in_reference": dict(index_status),
        "caption": ld_status_caption,
        "n_reference_snps": len(ref_snps),
    }

    metadata = {
        "mode": mode,
        "mode_label": mode_label,
        "required_n_indices": required_n,
        "index_selection_method": index_selection_method,
        "auto_index_source": (auto_source if index_variants is None else None),
        "auto_index_source_label": auto_source_label,
        "auto_index_summary": auto_summary,
        "clump_r2": clump_r2,
        "compare_mode": COMPARE_MODE_SEPARATE,
        "reference_dir": str(ref.reference_dir) if ref.reference_dir is not None else None,
        "bfile_prefix": str(ld_bfile_prefix),
        "gtf_path": str(gtf_path),
        "recombination_bw_path": recombination_bw,
        "plink_path": plink_path,
        "ld_window_start_bp": ld_bp_start,
        "ld_window_end_bp": ld_bp_end,
        "gene_window_start_bp": gene_bp_start,
        "gene_window_end_bp": gene_bp_end,
        "window_snps": window_snps,
        "n_dataset1_window_snps": n_top,
        "n_dataset2_window_snps": n_bottom,
        "n_genes": None if genes_df is None else len(genes_df),
        "gene_track_rows": gene_track_rows,
        "compare_counts": compare_counts,
        "compare_summary": compare_summary,
        "summary_text": summary_text,
        "n_coding_flipped_dataset1": n_flip_top,
        "n_coding_flipped_dataset2": n_flip_bottom,
        "config": config,
        "output_path": str(output_path) if output_path is not None else None,
        "output_written": output_path is not None,
        "warnings": tuple(warnings),
    }

    return LocusBlendResult(
        locus_figure=locus_figure,
        compare_figure=compare_figure,
        chromosome=selected_chrom,
        center_bp=center,
        window_kb=window_kb,
        window_start_bp=ld_bp_start,
        window_end_bp=ld_bp_end,
        mode=mode,
        ancestry=ancestry_code,
        index_variants=tuple(index_labels),
        index_reference_snps=tuple(index_refs),
        selected_index_variants=index_records,
        dataset1_processed=df1_plot,
        dataset2_processed=df2_plot,
        genes=genes_df,
        ld_status=ld_status,
        n_window_snps=len(ref_snps),
        metadata=metadata,
        warnings=tuple(warnings),
    )
