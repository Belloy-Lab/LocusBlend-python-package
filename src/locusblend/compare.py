"""Locus compare plots: separate panels, triptych, and single blended compare.

Extracted from ``app.2.8.12.py``: compare-data construction, individual compare
plots, the triptych builder, the blended single-plot builder, and the
compare autoscale utilities. Standard / Two-index / Three-index behavior is
preserved, including LD binning and index-marker rendering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .colors import COLOR_MAPPING
from .io import dedup_columns
from .plotting import _safe_row_nanmax

# --- extracted from app.2.8.12.py (verbatim) ---


def get_compare_signal_info(signal, idx1_label, idx2_label, idx3_label):
    mapping = {
        "variant 1": {"r2_col": "r2_1", "color": "#00ffdb", "index_label": idx1_label},
        "variant 2": {"r2_col": "r2_2", "color": "#ff00fa", "index_label": idx2_label},
        "variant 3": {"r2_col": "r2_3", "color": "#ffc900", "index_label": idx3_label},
    }
    return mapping[signal]


def build_compare_data(df_top, df_bottom, signal, idx1_label, idx2_label, idx3_label):
    df_top = dedup_columns(df_top)
    df_bottom = dedup_columns(df_bottom)

    info = get_compare_signal_info(signal, idx1_label, idx2_label, idx3_label)
    r2_col = info["r2_col"]

    top = df_top[["CHR", "BP", "DISPLAY_ID", "P", r2_col, "in_ref", "REF_SNP"]].copy().rename(
        columns={"DISPLAY_ID": "DISPLAY_ID_top", "P": "P_top", r2_col: "r2_use", "in_ref": "in_ref_top", "REF_SNP": "REF_SNP_top"}
    )
    bottom = df_bottom[["CHR", "BP", "DISPLAY_ID", "P", "in_ref", "REF_SNP"]].copy().rename(
        columns={"DISPLAY_ID": "DISPLAY_ID_bottom", "P": "P_bottom", "in_ref": "in_ref_bottom", "REF_SNP": "REF_SNP_bottom"}
    )

    # 优先用 REF_SNP 做内部对齐；没有时退回 position key
    top["MERGE_KEY"] = np.where(
        top["REF_SNP_top"].notna(),
        top["REF_SNP_top"].astype(str),
        "POS:" + top["CHR"].astype(str) + ":" + top["BP"].astype("Int64").astype(str)
    )
    bottom["MERGE_KEY"] = np.where(
        bottom["REF_SNP_bottom"].notna(),
        bottom["REF_SNP_bottom"].astype(str),
        "POS:" + bottom["CHR"].astype(str) + ":" + bottom["BP"].astype("Int64").astype(str)
    )

    merged = pd.merge(top, bottom, on="MERGE_KEY", how="inner")
    merged["P_top"] = pd.to_numeric(merged["P_top"], errors="coerce")
    merged["P_bottom"] = pd.to_numeric(merged["P_bottom"], errors="coerce")
    merged["r2_use"] = pd.to_numeric(merged["r2_use"], errors="coerce")
    merged["in_ref"] = merged["in_ref_top"].fillna(False).astype(bool)

    merged = merged[
        merged["P_top"].notna() &
        merged["P_bottom"].notna() &
        (merged["P_top"] > 0) &
        (merged["P_bottom"] > 0)
    ].copy()

    merged["logP_top"] = -np.log10(merged["P_top"])
    merged["logP_bottom"] = -np.log10(merged["P_bottom"])
    merged["size"] = np.maximum(8, 4 + 12 * merged["r2_use"].fillna(0))
    return merged, info


def build_compare_figure(df_top, df_bottom, signal, idx1_label, idx2_label, idx3_label, title_top, title_bottom, compare_size, ld_labels):
    merged, info = build_compare_data(df_top, df_bottom, signal, idx1_label, idx2_label, idx3_label)

    if merged.empty:
        fig = go.Figure()
        fig.update_layout(width=compare_size, height=compare_size, title=f"Locus compare ({signal})")
        return fig, 0

    missing_ref = merged[~merged["in_ref"]].copy()
    grey = merged[merged["in_ref"] & ((merged["r2_use"] < 0.2) | (merged["r2_use"].isna()))].copy()
    colored = merged[merged["in_ref"] & (merged["r2_use"] >= 0.2)].copy()

    def tooltip_text(r):
        r2_text = "NA" if pd.isna(r["r2_use"]) else f"{r['r2_use']:.3f}"
        ref_text = ld_labels["in_ref"] if bool(r["in_ref"]) else ld_labels["not_in_ref"]
        label = r["DISPLAY_ID_top"] if pd.notna(r["DISPLAY_ID_top"]) else r["MERGE_KEY"]
        return (
            f"SNP: {label}"
            f"<br>-log10(P top): {r['logP_top']:.3f}"
            f"<br>-log10(P bottom): {r['logP_bottom']:.3f}"
            f"<br>r2: {r2_text}"
            f"<br>{ref_text}"
        )

    if not missing_ref.empty:
        missing_ref["tooltip"] = [tooltip_text(r) for _, r in missing_ref.iterrows()]
    if not grey.empty:
        grey["tooltip"] = [tooltip_text(r) for _, r in grey.iterrows()]
    if not colored.empty:
        colored["tooltip"] = [tooltip_text(r) for _, r in colored.iterrows()]

    fig = go.Figure()

    # 1) missing_ref (x) first
    if not missing_ref.empty:
        fig.add_trace(
            go.Scattergl(
                x=missing_ref["logP_top"],
                y=missing_ref["logP_bottom"],
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
            )
        )

    # 2) grey points
    if not grey.empty:
        fig.add_trace(
            go.Scattergl(
                x=grey["logP_top"],
                y=grey["logP_bottom"],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    color="#FFFFFF",
                    line=dict(color="#3c3c3c", width=1),
                    opacity=0.25,
                    size=8,
                ),
                text=grey["tooltip"],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )

    # 3) colored on top
    if not colored.empty:
        fig.add_trace(
            go.Scattergl(
                x=colored["logP_top"],
                y=colored["logP_bottom"],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    color=info["color"],
                    line=dict(color="white", width=1),
                    opacity=1.0,
                    size=colored["size"],
                ),
                text=colored["tooltip"],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )

    idx_label = str(info["index_label"]).strip()
    idx_df = merged[merged["DISPLAY_ID_top"].astype(str) == idx_label].copy()
    if idx_df.empty:
        idx_df = merged[merged["DISPLAY_ID_bottom"].astype(str) == idx_label].copy()

    if not idx_df.empty:
        idx_in_ref = bool(idx_df["in_ref"].fillna(False).iloc[0])
        if idx_in_ref:
            fig.add_trace(
                go.Scattergl(
                    x=idx_df["logP_top"],
                    y=idx_df["logP_bottom"],
                    mode="markers",
                    marker=dict(
                        symbol="diamond",
                        color=info["color"],
                        line=dict(color="black", width=2),
                        size=22,
                        opacity=1.0
                    ),
                    text=[f"Index SNP: {idx_label}"] * len(idx_df),
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False
                )
            )
        else:
            fig.add_trace(
                go.Scattergl(
                    x=idx_df["logP_top"],
                    y=idx_df["logP_bottom"],
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        color="#d9d9d9",
                        size=4,
                        line=dict(width=0),
                        opacity=0.6,
                    ),
                    text=[f"{ld_labels['index_not_found']}: {idx_label}"] * len(idx_df),
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False
                )
            )

    x_min = float(np.floor(np.nanmin(merged["logP_top"])))
    x_max = float(np.ceil(np.nanmax(merged["logP_top"])))
    y_min = float(np.floor(np.nanmin(merged["logP_bottom"])))
    y_max = float(np.ceil(np.nanmax(merged["logP_bottom"])))

    if x_max <= x_min:
        x_max = x_min + 1
    if y_max <= y_min:
        y_max = y_min + 1

    fig.update_xaxes(
        title_text=f"-log10(P): {title_top}",
        range=[x_min, x_max],
        zeroline=False,
        showgrid=False
    )
    fig.update_yaxes(
        title_text=f"-log10(P): {title_bottom}",
        range=[y_min, y_max],
        zeroline=False,
        showgrid=False
    )

    fig.update_layout(
        title=dict(text=f"Locus compare ({signal})", x=0.5, xanchor="center"),
        width=compare_size,
        height=compare_size,
        dragmode="zoom",
        margin=dict(l=60, r=30, b=60, t=50),
        showlegend=False
    )

    return fig, len(merged)


def _add_compare_panel(
    fig,
    merged,
    info,
    title_top,
    title_bottom,
    row,
    col,
    x_range,
    y_range,
    ld_labels,
    show_y_title=False
):
    if merged.empty:
        fig.add_annotation(
            x=(x_range[0] + x_range[1]) / 2,
            y=(y_range[0] + y_range[1]) / 2,
            text="No overlapping SNPs",
            showarrow=False,
            font=dict(size=12, color="#666666"),
            row=row,
            col=col
        )
        fig.update_xaxes(
            title_text=f"-log10(P): {title_top}",
            range=x_range,
            zeroline=False,
            showgrid=False,
            row=row,
            col=col
        )
        fig.update_yaxes(
            title_text=f"-log10(P): {title_bottom}" if show_y_title else None,
            range=y_range,
            zeroline=False,
            showgrid=False,
            row=row,
            col=col
        )
        return 0

    missing_ref = merged[~merged["in_ref"]].copy()
    grey = merged[merged["in_ref"] & ((merged["r2_use"] < 0.2) | (merged["r2_use"].isna()))].copy()
    colored = merged[merged["in_ref"] & (merged["r2_use"] >= 0.2)].copy()

    def tooltip_text(r):
        r2_text = "NA" if pd.isna(r["r2_use"]) else f"{r['r2_use']:.3f}"
        ref_text = ld_labels["in_ref"] if bool(r["in_ref"]) else ld_labels["not_in_ref"]
        label = r["DISPLAY_ID_top"] if pd.notna(r["DISPLAY_ID_top"]) else r["MERGE_KEY"]
        return (
            f"SNP: {label}"
            f"<br>-log10(P top): {r['logP_top']:.3f}"
            f"<br>-log10(P bottom): {r['logP_bottom']:.3f}"
            f"<br>r2: {r2_text}"
            f"<br>{ref_text}"
        )

    if not missing_ref.empty:
        missing_ref["tooltip"] = [tooltip_text(r) for _, r in missing_ref.iterrows()]
        fig.add_trace(
            go.Scattergl(
                x=missing_ref["logP_top"],
                y=missing_ref["logP_bottom"],
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
            row=row,
            col=col
        )

    if not grey.empty:
        grey["tooltip"] = [tooltip_text(r) for _, r in grey.iterrows()]
        fig.add_trace(
            go.Scattergl(
                x=grey["logP_top"],
                y=grey["logP_bottom"],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    color="#FFFFFF",
                    line=dict(color="#3c3c3c", width=1),
                    opacity=0.25,
                    size=8,
                ),
                text=grey["tooltip"],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col
        )

    if not colored.empty:
        colored["tooltip"] = [tooltip_text(r) for _, r in colored.iterrows()]
        fig.add_trace(
            go.Scattergl(
                x=colored["logP_top"],
                y=colored["logP_bottom"],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    color=info["color"],
                    line=dict(color="white", width=1),
                    opacity=1.0,
                    size=colored["size"],
                ),
                text=colored["tooltip"],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col
        )

    idx_label = str(info["index_label"]).strip()
    idx_df = merged[merged["DISPLAY_ID_top"].astype(str) == idx_label].copy()
    if idx_df.empty:
        idx_df = merged[merged["DISPLAY_ID_bottom"].astype(str) == idx_label].copy()

    if not idx_df.empty:
        idx_in_ref = bool(idx_df["in_ref"].fillna(False).iloc[0])
        if idx_in_ref:
            fig.add_trace(
                go.Scattergl(
                    x=idx_df["logP_top"],
                    y=idx_df["logP_bottom"],
                    mode="markers",
                    marker=dict(
                        symbol="diamond",
                        color=info["color"],
                        line=dict(color="black", width=2),
                        size=22,
                        opacity=1.0
                    ),
                    text=[f"Index SNP: {idx_label}"] * len(idx_df),
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False
                ),
                row=row,
                col=col
            )
        else:
            fig.add_trace(
                go.Scattergl(
                    x=idx_df["logP_top"],
                    y=idx_df["logP_bottom"],
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        color="#d9d9d9",
                        size=4,
                        line=dict(width=0),
                        opacity=0.6,
                    ),
                    text=[f"{ld_labels['index_not_found']}: {idx_label}"] * len(idx_df),
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False
                ),
                row=row,
                col=col
            )

    fig.update_xaxes(
        title_text=f"-log10(P): {title_top}",
        range=x_range,
        zeroline=False,
        showgrid=False,
        row=row,
        col=col
    )
    fig.update_yaxes(
        title_text=f"-log10(P): {title_bottom}" if show_y_title else None,
        range=y_range,
        zeroline=False,
        showgrid=False,
        row=row,
        col=col
    )

    return len(merged)


def build_compare_figure_triptych(
    df_top,
    df_bottom,
    idx1_label,
    idx2_label,
    idx3_label,
    title_top,
    title_bottom,
    compare_size,
    ld_labels,
    active_signals=None,
):
    if active_signals is None:
        active_signals = ["variant 1", "variant 2", "variant 3"]
    signals = list(active_signals)
    n_cols = len(signals)
    panel_data = []

    global_xmin = np.inf
    global_xmax = -np.inf
    global_ymin = np.inf
    global_ymax = -np.inf

    for signal in signals:
        merged, info = build_compare_data(
            df_top=df_top,
            df_bottom=df_bottom,
            signal=signal,
            idx1_label=idx1_label,
            idx2_label=idx2_label,
            idx3_label=idx3_label
        )
        panel_data.append((signal, merged, info))

        if not merged.empty:
            global_xmin = min(global_xmin, float(np.floor(np.nanmin(merged["logP_top"]))))
            global_xmax = max(global_xmax, float(np.ceil(np.nanmax(merged["logP_top"]))))
            global_ymin = min(global_ymin, float(np.floor(np.nanmin(merged["logP_bottom"]))))
            global_ymax = max(global_ymax, float(np.ceil(np.nanmax(merged["logP_bottom"]))))

    if not np.isfinite(global_xmin):
        global_xmin, global_xmax = 0.0, 1.0
    if not np.isfinite(global_ymin):
        global_ymin, global_ymax = 0.0, 1.0

    if global_xmax <= global_xmin:
        global_xmax = global_xmin + 1
    if global_ymax <= global_ymin:
        global_ymax = global_ymin + 1

    x_range = [global_xmin, global_xmax]
    y_range = [global_ymin, global_ymax]

    fig = make_subplots(
        rows=1,
        cols=n_cols,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        subplot_titles=[f"Locus compare ({s})" for s in signals]
    )

    counts = {}

    for i, (signal, merged, info) in enumerate(panel_data, start=1):
        counts[signal] = _add_compare_panel(
            fig=fig,
            merged=merged,
            info=info,
            title_top=title_top,
            title_bottom=title_bottom,
            row=1,
            col=i,
            x_range=x_range,
            y_range=y_range,
            ld_labels=ld_labels,
            show_y_title=(i == 1)
        )

    fig.update_layout(
        width=int(compare_size * (1.1 * n_cols)),
        height=int(compare_size),
        dragmode="zoom",
        margin=dict(l=60, r=30, b=60, t=60),
        showlegend=False
    )

    return fig, counts


def build_single_blended_locuscompare(
    df_top,
    df_bottom,
    idx1_ref,
    idx2_ref,
    idx3_ref,
    idx1_label,
    idx2_label,
    idx3_label,
    title_top,
    title_bottom,
    compare_size,
    ld_labels,
    locusblend_mode="Three-index LocusBlend",
):
    df_top = dedup_columns(df_top)
    df_bottom = dedup_columns(df_bottom)

    top_cols = ["CHR", "BP", "DISPLAY_ID", "P", "r2_1", "r2_2", "r2_3", "in_ref", "REF_SNP"]
    top = df_top[[c for c in top_cols if c in df_top.columns]].copy().rename(
        columns={"DISPLAY_ID": "DISPLAY_ID_top", "P": "P_top",
                 "in_ref": "in_ref_top", "REF_SNP": "REF_SNP_top"}
    )
    bottom = df_bottom[["CHR", "BP", "DISPLAY_ID", "P", "in_ref", "REF_SNP"]].copy().rename(
        columns={"DISPLAY_ID": "DISPLAY_ID_bottom", "P": "P_bottom",
                 "in_ref": "in_ref_bottom", "REF_SNP": "REF_SNP_bottom"}
    )

    top["MERGE_KEY"] = np.where(
        top["REF_SNP_top"].notna(),
        top["REF_SNP_top"].astype(str),
        "POS:" + top["CHR"].astype(str) + ":" + top["BP"].astype("Int64").astype(str)
    )
    bottom["MERGE_KEY"] = np.where(
        bottom["REF_SNP_bottom"].notna(),
        bottom["REF_SNP_bottom"].astype(str),
        "POS:" + bottom["CHR"].astype(str) + ":" + bottom["BP"].astype("Int64").astype(str)
    )

    merged = pd.merge(top, bottom, on="MERGE_KEY", how="inner", suffixes=("", "_b"))
    merged = merged.drop_duplicates("MERGE_KEY").copy()

    merged["P_top"] = pd.to_numeric(merged["P_top"], errors="coerce")
    merged["P_bottom"] = pd.to_numeric(merged["P_bottom"], errors="coerce")
    for c in ["r2_1", "r2_2", "r2_3"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
        else:
            merged[c] = np.nan
    merged["in_ref"] = merged["in_ref_top"].fillna(False).astype(bool)

    merged = merged[
        merged["P_top"].notna() &
        merged["P_bottom"].notna() &
        (merged["P_top"] > 0) &
        (merged["P_bottom"] > 0)
    ].copy()

    if merged.empty:
        fig = go.Figure()
        fig.update_layout(
            width=compare_size,
            height=compare_size,
            title=dict(text="Locus compare (blended)", x=0.5, xanchor="center")
        )
        return fig, 0

    merged["logP_top"] = -np.log10(merged["P_top"])
    merged["logP_bottom"] = -np.log10(merged["P_bottom"])

    bins = [-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf]
    labels = ["0", "2", "4", "6", "8"]
    merged["ld_bin"] = pd.cut(merged["r2_1"], bins=bins, labels=labels, include_lowest=True).astype("string").fillna("0")

    if locusblend_mode == "Standard locus zoom":
        merged["r2_2_bin"] = "0"
        merged["r2_3_bin"] = "0"
        merged["group_code"] = merged["ld_bin"] + "0" + "0"
    elif locusblend_mode == "Two-index LocusBlend":
        merged["r2_2_bin"] = pd.cut(merged["r2_2"], bins=bins, labels=labels, include_lowest=True).astype("string").fillna("0")
        merged["r2_3_bin"] = "0"
        merged["group_code"] = merged["ld_bin"] + merged["r2_2_bin"] + "0"
    else:
        merged["r2_2_bin"] = pd.cut(merged["r2_2"], bins=bins, labels=labels, include_lowest=True).astype("string").fillna("0")
        merged["r2_3_bin"] = pd.cut(merged["r2_3"], bins=bins, labels=labels, include_lowest=True).astype("string").fillna("0")
        merged["group_code"] = merged["ld_bin"] + merged["r2_2_bin"] + merged["r2_3_bin"]

    if locusblend_mode == "Standard locus zoom":
        arr_any = merged[["r2_1"]].to_numpy(dtype=float)
    elif locusblend_mode == "Two-index LocusBlend":
        arr_any = merged[["r2_1", "r2_2"]].to_numpy(dtype=float)
    else:
        arr_any = merged[["r2_1", "r2_2", "r2_3"]].to_numpy(dtype=float)
    merged["max_r_any"] = _safe_row_nanmax(arr_any)

    index_keys = {
        str(x).strip()
        for x in [idx1_ref, idx2_ref, idx3_ref, idx1_label, idx2_label, idx3_label]
        if x is not None and str(x).strip() != ""
    }

    is_index = pd.Series(False, index=merged.index)
    for col in ["REF_SNP_top", "REF_SNP_bottom", "DISPLAY_ID_top", "DISPLAY_ID_bottom"]:
        if col in merged.columns:
            is_index = is_index | merged[col].astype(str).str.strip().isin(index_keys)
    merged["is_index"] = is_index

    arr_size = arr_any.copy()
    idx_mask = merged["is_index"].to_numpy()
    if idx_mask.any():
        idx_vals = arr_size[idx_mask]
        idx_vals[np.isclose(idx_vals, 1.0, equal_nan=False)] = np.nan
        arr_size[idx_mask] = idx_vals
    merged["max_r_size"] = _safe_row_nanmax(arr_size)
    merged["max_r_size"] = merged["max_r_size"].fillna(0)

    colored = merged[merged["in_ref"] & (merged["max_r_any"] >= 0.2)].copy()
    grey = merged[merged["in_ref"] & ((merged["max_r_any"] < 0.2) | (merged["max_r_any"].isna()))].copy()
    missing_ref = merged[~merged["in_ref"]].copy()

    if not colored.empty:
        colored["fill_hex"] = colored["group_code"].map(COLOR_MAPPING).fillna("#bfbfbf")
        colored["size"] = np.clip(2 + 10 * colored["max_r_size"].fillna(0), 4, 12)
    if not grey.empty:
        grey["size"] = 4
    if not missing_ref.empty:
        missing_ref["size"] = 6

    def tooltip_text(r):
        label = r["DISPLAY_ID_top"] if pd.notna(r.get("DISPLAY_ID_top")) else r["MERGE_KEY"]
        ref_text = ld_labels["in_ref"] if bool(r["in_ref"]) else ld_labels["not_in_ref"]
        parts = [
            f"SNP: {label}",
            f"-log10(P top): {r['logP_top']:.3f}",
            f"-log10(P bottom): {r['logP_bottom']:.3f}",
        ]
        for k in ["r2_1", "r2_2", "r2_3"]:
            v = r.get(k)
            parts.append(f"{k}: {'NA' if pd.isna(v) else f'{v:.3f}'}")
        parts.append(ref_text)
        return "<br>".join(parts)

    if not missing_ref.empty:
        missing_ref["tooltip"] = [tooltip_text(r) for _, r in missing_ref.iterrows()]
    if not grey.empty:
        grey["tooltip"] = [tooltip_text(r) for _, r in grey.iterrows()]
    if not colored.empty:
        colored["tooltip"] = [tooltip_text(r) for _, r in colored.iterrows()]

    fig = go.Figure()

    if not missing_ref.empty:
        fig.add_trace(
            go.Scattergl(
                x=missing_ref["logP_top"],
                y=missing_ref["logP_bottom"],
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
            )
        )

    if not grey.empty:
        fig.add_trace(
            go.Scattergl(
                x=grey["logP_top"],
                y=grey["logP_bottom"],
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
            )
        )

    if not colored.empty:
        fig.add_trace(
            go.Scattergl(
                x=colored["logP_top"],
                y=colored["logP_bottom"],
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
            )
        )

    def add_blended_index_marker(fig, idx_ref, idx_label, fill_color):
        if idx_ref is None and (idx_label is None or str(idx_label).strip() == ""):
            return fig

        idx_df = pd.DataFrame()
        if idx_ref is not None:
            for col in ["REF_SNP_top", "REF_SNP_bottom"]:
                if col in merged.columns:
                    cand = merged[merged[col].astype(str) == str(idx_ref)]
                    if not cand.empty:
                        idx_df = cand.copy()
                        break

        if idx_df.empty and idx_label is not None and str(idx_label).strip() != "":
            for col in ["DISPLAY_ID_top", "DISPLAY_ID_bottom"]:
                if col in merged.columns:
                    cand = merged[merged[col].astype(str) == str(idx_label).strip()]
                    if not cand.empty:
                        idx_df = cand.copy()
                        break

        if idx_df.empty:
            return fig

        idx_in_ref = bool(idx_df["in_ref"].fillna(False).iloc[0])
        if idx_in_ref:
            fig.add_trace(
                go.Scattergl(
                    x=idx_df["logP_top"],
                    y=idx_df["logP_bottom"],
                    mode="markers",
                    marker=dict(
                        symbol="diamond",
                        color=fill_color,
                        line=dict(color="black", width=2),
                        size=18,
                        opacity=1.0,
                    ),
                    text=[f"Index SNP: {idx_label}"] * len(idx_df),
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                )
            )
        else:
            fig.add_trace(
                go.Scattergl(
                    x=idx_df["logP_top"],
                    y=idx_df["logP_bottom"],
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        color="#d9d9d9",
                        size=4,
                        line=dict(width=0),
                        opacity=0.6,
                    ),
                    text=[f"{ld_labels['index_not_found']}: {idx_label}"] * len(idx_df),
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                )
            )
        return fig

    fig = add_blended_index_marker(fig, idx1_ref, idx1_label, "#00ffdb")
    if locusblend_mode != "Standard locus zoom":
        fig = add_blended_index_marker(fig, idx2_ref, idx2_label, "#ff00fa")
    if locusblend_mode == "Three-index LocusBlend":
        fig = add_blended_index_marker(fig, idx3_ref, idx3_label, "#ffc900")

    x_min = float(np.floor(np.nanmin(merged["logP_top"])))
    x_max = float(np.ceil(np.nanmax(merged["logP_top"])))
    y_min = float(np.floor(np.nanmin(merged["logP_bottom"])))
    y_max = float(np.ceil(np.nanmax(merged["logP_bottom"])))
    if x_max <= x_min:
        x_max = x_min + 1
    if y_max <= y_min:
        y_max = y_min + 1

    fig.update_xaxes(
        title_text=f"-log10(P): {title_top}",
        range=[x_min, x_max],
        zeroline=False,
        showgrid=False
    )
    fig.update_yaxes(
        title_text=f"-log10(P): {title_bottom}",
        range=[y_min, y_max],
        zeroline=False,
        showgrid=False
    )
    fig.update_layout(
        title=dict(text="Locus compare (blended)", x=0.5, xanchor="center"),
        width=int(compare_size),
        height=int(compare_size),
        autosize=False,
        dragmode="zoom",
        margin=dict(l=70, r=40, b=70, t=50),
        showlegend=False
    )

    return fig, len(merged)


def _numeric_values(values):
    """Return finite numeric values from a Plotly trace coordinate array."""
    if values is None:
        return []
    try:
        arr = pd.to_numeric(pd.Series(list(values)), errors="coerce")
        arr = arr[np.isfinite(arr)]
        return arr.astype(float).tolist()
    except Exception:
        return []


def _padded_range(values, include_zero=True, pad_frac=0.12):
    """Compute a safe padded numeric range for Plotly axes.

    Similar to ggplot2 expand: add visual breathing room around the data so
    large diamond/index markers are not clipped by axis limits.
    """
    vals = [float(v) for v in values if np.isfinite(v)]
    if not vals:
        return None

    vmin = min(vals)
    vmax = max(vals)

    if include_zero:
        vmin = min(0.0, vmin)

    span = max(vmax - vmin, 1.0)
    pad = max(span * float(pad_frac), 0.35)

    lower = vmin - pad
    upper = vmax + pad

    if include_zero:
        lower = min(0.0, lower)

    return [lower, upper]


def apply_locus_compare_safe_autoscale(fig, pad_frac=0.12):
    """Set a safe initial axis range for locus compare figures.

    This prevents Plotly 'Reset axes' from returning to a too-tight or
    stale range that clips points or diamond markers.

    Layout-only. Must not change trace data, marker colors, LD colors,
    marker sizes, group_code, or index-marker colors.
    """
    if fig is None:
        return fig

    all_x = []
    all_y = []

    for trace in fig.data:
        all_x.extend(_numeric_values(getattr(trace, "x", None)))
        all_y.extend(_numeric_values(getattr(trace, "y", None)))

    x_range = _padded_range(all_x, include_zero=True, pad_frac=pad_frac)
    y_range = _padded_range(all_y, include_zero=True, pad_frac=pad_frac)

    if x_range is not None:
        fig.update_xaxes(range=x_range, autorange=False)

    if y_range is not None:
        fig.update_yaxes(range=y_range, autorange=False)

    return fig


def get_locus_compare_plotly_config(base_config=None):
    """Return Plotly config for locus compare charts.

    Removes Reset axes because safe autoscale is the intended recovery
    behavior for compare plots. Autoscale should remain available.
    """
    cfg = dict(base_config or {})
    remove = list(cfg.get("modeBarButtonsToRemove", []))

    for button_name in ["resetScale2d"]:
        if button_name not in remove:
            remove.append(button_name)

    cfg["modeBarButtonsToRemove"] = remove
    return cfg
