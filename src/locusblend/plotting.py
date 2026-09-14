"""Core locus plotting (Streamlit-independent).

Extracted from ``app.2.8.12.py``: ``get_plotly_locus_py``, tooltip and LD
reference label helpers, recombination-track reading, the layout-only plot
theme, and the figure cloning helper.

Plotting behavior is deliberately untouched in this pass: colors, marker size
logic, LD binning, index-marker colors, missing-reference rendering,
recombination overlay and axis semantics all match the baseline.

Changes required to remove app coupling (documented, no behavior redesign):

* ``get_plotly_locus_py(..., bw_path=...)`` used to default to the BigWig file
  sitting next to the app script. The default is now ``None``: the path is
  resolved from :class:`~locusblend.reference.ReferenceManager` (i.e. from
  ``reference_dir/recombination`` via ``LOCUSBLEND_REFERENCE_DIR``), and when no
  recombination file is available the overlay is skipped exactly like the
  baseline's ``os.path.exists`` guard.
* ``pyBigWig`` is imported lazily inside ``_read_recomb_track_bw`` so the core
  plotting import does not require the optional dependency.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .colors import COLOR_MAPPING
from .io import chrom_mask, dedup_columns, format_chrom_axis_title, normalize_chrom
from .reference import (
    INTERNAL_1000G_DEFAULT_ANCESTRY,
    default_recombination_bw_path,
    normalize_internal_1000g_ancestry,
)

# --- extracted from app.2.8.12.py (verbatim) ---


def _safe_row_nanmax(arr):
    out = np.full(arr.shape[0], np.nan, dtype=float)
    ok = ~np.all(np.isnan(arr), axis=1)
    if ok.any():
        out[ok] = np.nanmax(arr[ok], axis=1)
    return out


def _pick_bw_chrom(chrom, chroms_dict):
    chrom = str(chrom)
    base = chrom.replace("chr", "")
    candidates = [chrom, base, f"chr{base}"]
    for c in candidates:
        if c in chroms_dict:
            return c
    raise ValueError(f"Cannot find chromosome {chrom} in bigWig.")


def _read_recomb_track_bw(chrom, start, end, bw_path, n_bins=800):
    # app.2.8.12.py imported pyBigWig at module import time; here it is an
    # optional dependency of the recombination overlay only.
    try:
        import pyBigWig
    except ImportError as e:
        raise RuntimeError(
            "pyBigWig is required to read the recombination-rate BigWig track. "
            "Install it with: pip install pyBigWig"
        ) from e

    bw = pyBigWig.open(str(bw_path))
    try:
        chroms = bw.chroms()
        chrom_name = _pick_bw_chrom(chrom, chroms)
        vals = bw.stats(chrom_name, int(start), int(end), nBins=int(n_bins), type="mean")
    finally:
        bw.close()

    vals = np.array(vals, dtype=float)
    x = np.linspace(start, end, n_bins, endpoint=False) + (end - start) / n_bins / 2

    rec = pd.DataFrame({
        "BP": x,
        "value": vals
    }).dropna()

    return rec


def make_locus_tooltip(row, ref_status):
    label = row["DISPLAY_ID"] if "DISPLAY_ID" in row.index and pd.notna(row["DISPLAY_ID"]) else row.get("REF_SNP", row.get("SNP", "NA"))
    parts = [
        f"SNP: {label}",
        f"BP: {int(row['BP'])}",
        f"-log10(P): {row['logP']:.3f}",
    ]

    if "A1" in row.index and "A2" in row.index:
        parts.append(f"coding: {row['A1']}/{row['A2']}")
    if "beta" in row.index:
        parts.append(f"beta: {row['beta']:.5g}" if pd.notna(row["beta"]) else "beta: NA")
    if "coding_flipped" in row.index:
        parts.append(f"coding_flipped: {bool(row['coding_flipped'])}")
    if "r2_1" in row.index:
        parts.append(f"r2_1: {row['r2_1']:.3f}" if pd.notna(row["r2_1"]) else "r2_1: NA")
    if "r2_2" in row.index:
        parts.append(f"r2_2: {row['r2_2']:.3f}" if pd.notna(row["r2_2"]) else "r2_2: NA")
    if "r2_3" in row.index:
        parts.append(f"r2_3: {row['r2_3']:.3f}" if pd.notna(row["r2_3"]) else "r2_3: NA")
    if "group_code" in row.index:
        parts.append(f"code: {row['group_code']}")

    parts.append(ref_status)
    return "<br>".join(parts)


def get_plotly_locus_py(
    max_ylim,
    bp,
    window_bp,
    merged_female_withld,
    merged_df,
    idx1_ref,
    idx2_ref,
    idx3_ref,
    idx1_label,
    idx2_label,
    idx3_label,
    y_label,
    ld_labels,
    chrom,
    show_recomb=True,
    bw_path=None,
    n_recomb_bins=800,
    locusblend_mode="Three-index LocusBlend",
):
    # app.2.8.12.py defaulted to the BigWig sitting next to the app script.
    # Reference data now live outside the package, so the default is resolved
    # through the ReferenceManager (LOCUSBLEND_REFERENCE_DIR) and the overlay is
    # skipped when no file is available, exactly like the baseline's
    # os.path.exists guard.
    if bw_path is None:
        bw_path = default_recombination_bw_path()

    chrom = normalize_chrom(chrom)

    merged_female_withld = dedup_columns(merged_female_withld)
    merged_df = dedup_columns(merged_df)

    # Subset to selected chromosome first
    mf = merged_female_withld.loc[chrom_mask(merged_female_withld, chrom)].copy()
    md = merged_df.loc[chrom_mask(merged_df, chrom)].copy()

    if mf.empty or md.empty:
        raise ValueError(f"No SNPs found on chromosome {chrom} in the selected dataset.")

    data_bp_min = int(np.nanmin(mf["BP"]))
    data_bp_max = int(np.nanmax(mf["BP"]))

    bp_start = max(data_bp_min, int(bp - window_bp))
    bp_end = min(data_bp_max, int(bp + window_bp))

    d = md.loc[
        (md["BP"] >= bp_start) & (md["BP"] <= bp_end)
    ].copy()
    d = dedup_columns(d)

    if d.empty:
        raise ValueError(f"No SNPs found in chr{chrom}:{bp_start}-{bp_end}.")

    for col in ["BP", "P", "r2_1", "r2_2", "r2_3"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d["CHR"] = d["CHR"].astype(str)
    d["logP"] = -np.log10(d["P"].clip(lower=np.finfo(float).tiny))
    d["in_ref"] = d["in_ref"].fillna(False).astype(bool)

    bins = [-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf]
    labels = ["0", "2", "4", "6", "8"]

    d["ld_bin"] = pd.cut(d["r2_1"], bins=bins, labels=labels, include_lowest=True).astype("string").fillna("0")

    if locusblend_mode == "Standard locus zoom":
        d["r2_2_bin"] = "0"
        d["r2_3_bin"] = "0"
        d["group_code"] = d["ld_bin"] + "0" + "0"
    elif locusblend_mode == "Two-index LocusBlend":
        d["r2_2_bin"] = pd.cut(d["r2_2"], bins=bins, labels=labels, include_lowest=True).astype("string").fillna("0")
        d["r2_3_bin"] = "0"
        d["group_code"] = d["ld_bin"] + d["r2_2_bin"] + "0"
    else:
        d["r2_2_bin"] = pd.cut(d["r2_2"], bins=bins, labels=labels, include_lowest=True).astype("string").fillna("0")
        d["r2_3_bin"] = pd.cut(d["r2_3"], bins=bins, labels=labels, include_lowest=True).astype("string").fillna("0")
        d["group_code"] = d["ld_bin"] + d["r2_2_bin"] + d["r2_3_bin"]

    # ---------- size logic ----------
    # max_r_any: still used for color / grey split
    # max_r_size: used only for marker size
    # only remove self-LD=1 for true index SNP rows
    if locusblend_mode == "Standard locus zoom":
        arr_any = d[["r2_1"]].to_numpy(dtype=float)
    elif locusblend_mode == "Two-index LocusBlend":
        arr_any = d[["r2_1", "r2_2"]].to_numpy(dtype=float)
    else:
        arr_any = d[["r2_1", "r2_2", "r2_3"]].to_numpy(dtype=float)
    d["max_r_any"] = _safe_row_nanmax(arr_any)

    index_keys = {
        str(x).strip()
        for x in [idx1_ref, idx2_ref, idx3_ref, idx1_label, idx2_label, idx3_label]
        if x is not None and str(x).strip() != ""
    }

    d["is_index"] = False
    for col in ["REF_SNP", "DISPLAY_ID", "rsid", "SNP", "REF_MATCH", "uniqueid", "QUERY_UID"]:
        if col in d.columns:
            d["is_index"] = d["is_index"] | d[col].astype(str).str.strip().isin(index_keys)

    arr_size = arr_any.copy()
    idx_mask = d["is_index"].to_numpy()

    if idx_mask.any():
        idx_vals = arr_size[idx_mask]
        idx_vals[np.isclose(idx_vals, 1.0, equal_nan=False)] = np.nan
        arr_size[idx_mask] = idx_vals

    d["max_r_size"] = _safe_row_nanmax(arr_size)
    d["max_r_size"] = d["max_r_size"].fillna(0)

    colored = d.loc[d["in_ref"] & (d["max_r_any"] >= 0.2)].copy()
    grey = d.loc[d["in_ref"] & ((d["max_r_any"] < 0.2) | (d["max_r_any"].isna()))].copy()
    missing_ref = d.loc[~d["in_ref"]].copy()

    for x in [colored, grey, missing_ref]:
        x["x_mb"] = x["BP"] / 1e6

    colored["fill_hex"] = colored["group_code"].map(COLOR_MAPPING).fillna("#bfbfbf")

    # 这里控制普通 colored 点大小
    # 非 index SNP 若与某个 index 完全 LD (r2=1)，会正常变大
    # index SNP 自己不会因为 self-LD=1 变得过大
    colored["size"] = np.clip(2 + 10 * colored["max_r_size"].fillna(0), 4, 12)

    grey["size"] = 4
    missing_ref["size"] = 6

    if not colored.empty:
        colored["tooltip"] = [make_locus_tooltip(row, ld_labels["in_ref"]) for _, row in colored.iterrows()]
    if not grey.empty:
        grey["tooltip"] = [make_locus_tooltip(row, ld_labels["in_ref"]) for _, row in grey.iterrows()]
    if not missing_ref.empty:
        missing_ref["tooltip"] = [make_locus_tooltip(row, ld_labels["not_in_ref"]) for _, row in missing_ref.iterrows()]

    min_ylim = int(np.floor(np.nanmin(d["logP"])))
    if max_ylim is None:
        max_ylim = int(np.ceil(np.nanmax(d["logP"])))

    if show_recomb and bw_path is not None and os.path.exists(str(bw_path)):
        try:
            rec = _read_recomb_track_bw(chrom, bp_start, bp_end, bw_path=str(bw_path), n_bins=n_recomb_bins)
            rec["x_mb"] = rec["BP"] / 1e6
        except Exception:
            rec = pd.DataFrame(columns=["BP", "value", "x_mb"])
    else:
        rec = pd.DataFrame(columns=["BP", "value", "x_mb"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1) missing_ref (x) first (bottom layer)
    if not missing_ref.empty:
        fig.add_trace(
            go.Scattergl(
                x=missing_ref["x_mb"],
                y=missing_ref["logP"],
                mode="markers",
                marker=dict(
                    symbol="x",
                    color="#d9d9d9",
                    size=4,
                    line=dict(width=0),
                    opacity=0.6,
                ),
                text=missing_ref["tooltip"],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ),
            secondary_y=False,
        )

    # 2) grey points
    if not grey.empty:
        fig.add_trace(
            go.Scattergl(
                x=grey["x_mb"],
                y=grey["logP"],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    color="#bfbfbf",
                    size=grey["size"],
                    line=dict(width=0),
                    opacity=0.45,
                ),
                text=grey["tooltip"],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ),
            secondary_y=False,
        )

    # 3) colored points
    if not colored.empty:
        fig.add_trace(
            go.Scattergl(
                x=colored["x_mb"],
                y=colored["logP"],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    color=colored["fill_hex"],
                    line=dict(color="white", width=1),
                    opacity=1.0,
                    size=colored["size"],
                ),
                text=colored["tooltip"],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ),
            secondary_y=False,
        )

    def add_index_marker(fig, idx_ref, idx_label, fill_color):
        idx = pd.DataFrame()
        if idx_ref is not None:
            idx = d.loc[d["REF_SNP"].astype(str) == str(idx_ref)].copy()

        if idx.empty:
            idx = d.loc[d["DISPLAY_ID"].astype(str) == str(idx_label)].copy()

        if idx.empty:
            return fig

        idx_in_ref = bool(idx["in_ref"].fillna(False).iloc[0])

        if idx_in_ref:
            fig.add_trace(
                go.Scattergl(
                    x=idx["BP"] / 1e6,
                    y=idx["logP"],
                    mode="markers",
                    marker=dict(
                        symbol="diamond",
                        color=fill_color,
                        line=dict(color="black", width=1),
                        size=14,
                        opacity=0.9,
                    ),
                    text=[f"Index SNP: {idx_label}"] * len(idx),
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                ),
                secondary_y=False,
            )
        else:
            fig.add_trace(
                go.Scattergl(
                    x=idx["BP"] / 1e6,
                    y=idx["logP"],
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        color="#d9d9d9",
                        size=4,
                        line=dict(width=0),
                        opacity=0.6,
                    ),
                    text=[f"{ld_labels['index_not_found']}: {idx_label}"] * len(idx),
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                ),
                secondary_y=False,
            )
        return fig

    fig = add_index_marker(fig, idx1_ref, idx1_label, "#00ffdb")
    if locusblend_mode != "Standard locus zoom":
        fig = add_index_marker(fig, idx2_ref, idx2_label, "#ff00fa")
    if locusblend_mode == "Three-index LocusBlend":
        fig = add_index_marker(fig, idx3_ref, idx3_label, "#ffc900")

    if not rec.empty:
        fig.add_trace(
            go.Scattergl(
                x=rec["x_mb"],
                y=rec["value"],
                mode="lines",
                line=dict(color="#2d2d2d", width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title=dict(text=y_label, x=0.5, xanchor="center"),
        dragmode="zoom",
        margin=dict(l=60, r=60, b=50, t=40),
    )

    fig.update_xaxes(title_text=format_chrom_axis_title(chrom), zeroline=False)
    fig.update_yaxes(
        title_text="-log<sub>10</sub>(P)",
        range=[min_ylim, max_ylim],
        zeroline=False,
        secondary_y=False
    )
    fig.update_yaxes(
        title_text="Recombination rate",
        range=[0, 100],
        autorange=False,
        zeroline=False,
        showgrid=False,
        secondary_y=True
    )

    return fig, len(d), bp_start, bp_end


def get_ld_reference_labels(active_ld_source, ancestry=INTERNAL_1000G_DEFAULT_ANCESTRY):
    """Return tooltip/caption labels keyed to the currently active LD source.

    The downstream builders never hard-code the reference name; they read
    from this dict so the same plot machinery serves the 1000G mode and the
    two uploaded-LD modes.
    """
    if active_ld_source == "Use internal 1000G reference":
        ancestry = normalize_internal_1000g_ancestry(ancestry)
        source_name = f"1000G {ancestry} LD"
        return {
            "source_name": source_name,
            "in_ref": f"reference: in {source_name}",
            "not_in_ref": f"reference: not found in {source_name}",
            "index_not_found": f"Index SNP not found in {source_name}",
            "summary_label": f"1000G {ancestry} window SNPs",
        }
    return {
        "source_name": "user uploaded LD",
        "in_ref": "reference: in user uploaded LD",
        "not_in_ref": "reference: not found in user uploaded LD",
        "index_not_found": "Index SNP not found in user uploaded LD",
        "summary_label": "Uploaded LD SNPs",
    }


def apply_locusblend_plot_theme(fig):
    """Force Plotly figures to remain readable in Streamlit/browser dark mode.

    Layout-only. Must not change trace data, marker colors, LD color mapping,
    marker sizes, group_code, or index-marker colors.
    """
    if fig is None:
        return fig

    text = "#111827"
    muted = "#374151"
    grid = "#e5e7eb"
    axis = "#9ca3af"
    bg = "#ffffff"

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(color=text),
        hoverlabel=dict(
            bgcolor=bg,
            bordercolor=axis,
            font=dict(color=text),
        ),
    )

    # Do not overwrite fig.layout.title or title.text.
    # Only set title font color if a title already exists.
    try:
        if getattr(fig.layout, "title", None) is not None:
            existing_title = getattr(fig.layout.title, "text", None)
            if existing_title not in (None, "", "undefined"):
                fig.update_layout(title_font_color=text)
            elif existing_title == "undefined":
                fig.update_layout(title_text=None)
    except Exception:
        pass

    fig.update_xaxes(
        title_font=dict(color=text),
        tickfont=dict(color=muted),
        gridcolor=grid,
        zerolinecolor=grid,
        linecolor=axis,
    )

    fig.update_yaxes(
        title_font=dict(color=text),
        tickfont=dict(color=muted),
        gridcolor=grid,
        zerolinecolor=grid,
        linecolor=axis,
    )

    # Preserve explicit annotation colors, especially highlighted gene labels.
    # Only fill in missing annotation font color.
    for ann in fig.layout.annotations or []:
        try:
            if ann.font is None:
                ann.font = dict(color=text)
            elif not getattr(ann.font, "color", None):
                ann.font.color = text
        except Exception:
            pass

    return fig


def clone_plotly_figure(fig):
    """Return a detached copy of a Plotly figure so export layout changes do not mutate cached figures."""
    if fig is None:
        return None
    try:
        return go.Figure(fig.to_dict())
    except Exception:
        return copy.deepcopy(fig)
