"""Gene annotation parsing and the Plotly gene track.

Extracted from ``app.2.8.12.py``: GTF attribute parsing, parquet gene-table
loading, gene filtering (protein-coding by default), row assignment for the
gene track, and construction of the gene track subplot. The highlighted-gene
behavior is preserved.

``load_gene_table`` / ``load_genes_from_gtf`` no longer carry Streamlit cache
decorators; nothing else changed.
"""

from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go

from .io import log, normalize_chrom

# --- extracted from app.2.8.12.py (verbatim) ---


GTF_COLS = [
    "seqname", "source", "feature", "start", "end",
    "score", "strand", "frame", "attribute"
]


def get_attr(attr, key):
    m = re.search(fr'{key} "([^"]+)"', str(attr))
    return m.group(1) if m else None


def load_gene_table(parquet_path):
    df = pd.read_parquet(parquet_path)

    if "chrom" in df.columns:
        df["chrom"] = df["chrom"].map(normalize_chrom)
    elif "seqname" in df.columns:
        df["chrom"] = df["seqname"].map(normalize_chrom)

    return df


def load_genes_from_table(chrom, start, end, parquet_path, gene_display_mode="protein_coding"):
    genes = load_gene_table(parquet_path)
    chrom = normalize_chrom(chrom)

    sub = genes[
        (genes["chrom"] == chrom) &
        (genes["end"] >= start) &
        (genes["start"] <= end)
    ].copy()

    if gene_display_mode == "protein_coding" and "gene_type" in sub.columns:
        sub = sub[sub["gene_type"] == "protein_coding"].copy()

    if "seqname" not in sub.columns:
        sub["seqname"] = "chr" + sub["chrom"].astype(str)

    if "gene_name" not in sub.columns:
        sub["gene_name"] = None
    if "gene_id" not in sub.columns:
        sub["gene_id"] = None
    if "gene_type" not in sub.columns:
        sub["gene_type"] = None
    if "strand" not in sub.columns:
        sub["strand"] = None

    return sub


def load_genes_from_gtf(gtf_path, chrom, start, end, gene_display_mode="protein_coding", chunksize=200000):
    log(f"load_genes_from_gtf start: {gtf_path}, chrom={chrom}, start={start}, end={end}, mode={gene_display_mode}")
    keep = []

    chrom = normalize_chrom(chrom)
    chrom_candidates = [chrom, f"chr{chrom}"]

    for chunk in pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        header=None,
        names=GTF_COLS,
        compression="infer",
        chunksize=chunksize
    ):
        chunk["seqname"] = chunk["seqname"].astype(str)

        sub = chunk[
            (chunk["feature"] == "gene") &
            (chunk["seqname"].isin(chrom_candidates)) &
            (chunk["end"] >= start) &
            (chunk["start"] <= end)
        ].copy()

        if not sub.empty:
            keep.append(sub)

    if not keep:
        return pd.DataFrame(columns=["seqname", "start", "end", "strand", "gene_id", "gene_name", "gene_type"])

    g = pd.concat(keep, ignore_index=True)
    g["gene_id"] = g["attribute"].apply(lambda x: get_attr(x, "gene_id"))
    g["gene_name"] = g["attribute"].apply(lambda x: get_attr(x, "gene_name"))
    g["gene_type"] = g["attribute"].apply(lambda x: get_attr(x, "gene_type"))

    g = g[["seqname", "start", "end", "strand", "gene_id", "gene_name", "gene_type"]].sort_values("start").reset_index(drop=True)

    if gene_display_mode == "protein_coding":
        g = g[g["gene_type"] == "protein_coding"].copy()

    log(f"load_genes_from_gtf done: n_genes={len(g)}")
    return g


def assign_gene_rows(df, min_gap=30000):
    if df.empty:
        out = df.copy()
        out["track_row"] = pd.Series(dtype=int)
        return out

    df = df.sort_values("start").copy()
    row_ends = []
    rows = []

    for _, r in df.iterrows():
        placed = False
        for i in range(len(row_ends)):
            if r["start"] > row_ends[i] + min_gap:
                rows.append(i)
                row_ends[i] = r["end"]
                placed = True
                break
        if not placed:
            rows.append(len(row_ends))
            row_ends.append(r["end"])

    df["track_row"] = rows
    return df


def add_gene_track_to_subplot(fig, genes_df, row, col=1, min_gap=30000, highlight_names=None):
    if genes_df.empty:
        return fig, 1

    if highlight_names is None:
        highlight_names = set()

    genes_plot = assign_gene_rows(genes_df, min_gap=min_gap)

    for _, r in genes_plot.iterrows():
        y = -int(r["track_row"])
        label = r["gene_name"] if pd.notna(r["gene_name"]) else r["gene_id"]
        is_highlighted = label.lower() in highlight_names

        line_color = "#c45c00" if is_highlighted else "#2f6f7e"
        line_width = 9 if is_highlighted else 6
        font_color = "#c45c00" if is_highlighted else "#2f2f2f"
        font_family = "Arial Black" if is_highlighted else None
        font_size = 11 if is_highlighted else 10

        fig.add_trace(
            go.Scatter(
                x=[r["start"] / 1e6, r["end"] / 1e6],
                y=[y, y],
                mode="lines",
                line=dict(width=line_width, color=line_color),
                hovertemplate=(
                    f"gene: {label}<br>"
                    f"start: {r['start']}<br>"
                    f"end: {r['end']}<br>"
                    f"strand: {r['strand']}<br>"
                    f"type: {r['gene_type']}<extra></extra>"
                ),
                showlegend=False
            ),
            row=row,
            col=col
        )

        _font = dict(size=font_size, color=font_color)
        if font_family:
            _font["family"] = font_family
        fig.add_annotation(
            x=((r["start"] + r["end"]) / 2) / 1e6,
            y=y + 0.18,
            text=label,
            showarrow=False,
            font=_font,
            xanchor="center",
            yanchor="bottom",
            row=row,
            col=col
        )

    n_rows = int(genes_plot["track_row"].max()) + 1
    return fig, n_rows
