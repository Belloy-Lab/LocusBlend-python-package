import base64
from io import BytesIO
import copy
import hashlib
import os
import re
import time
import traceback
import shutil
import tempfile
import subprocess
import glob
import html as html_lib
from mimetypes import guess_type
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pyBigWig
import streamlit as st
import streamlit.components.v1 as components


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
BIN_DIR = APP_DIR / "bin"
ASSET_DIR = APP_DIR / "assets"
DISABLE_WELCOME_DIALOG = True
# app.2.7.py: chromosome X / 23 support.

INTERNAL_1000G_ANCESTRIES = {
    "AFR": "African / African ancestry",
    "AMR": "Admixed American ancestry",
    "EAS": "East Asian ancestry",
    "EUR": "European ancestry",
    "SAS": "South Asian ancestry",
}
INTERNAL_1000G_DEFAULT_ANCESTRY = "EUR"
INTERNAL_1000G_ANCESTRY_OPTIONS = list(INTERNAL_1000G_ANCESTRIES.keys())


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dedup_columns(df):
    return df.loc[:, ~pd.Index(df.columns).duplicated()].copy()


COLOR_MAPPING = {
    "002":"#ffc900","004":"#ffc900","006":"#ffc900","008":"#ffc900",
    "020":"#ff00fa","022":"#ff657d","024":"#ff8653","026":"#ff973f","028":"#ffa132",
    "040":"#ff00fa","042":"#ff43a7","044":"#ff657d","046":"#ff7964","048":"#ff8653",
    "060":"#ff00fa","062":"#ff32bc","064":"#ff5096","066":"#ff657d","068":"#ff736b",
    "080":"#ff00fa","082":"#ff28c8","084":"#ff43a7","086":"#ff568f","088":"#ff657d",
    "200":"#00ffdb","202":"#80e46e","204":"#aadb49","206":"#bfd737","208":"#ccd42c",
    "220":"#8080eb","222":"#aa989c","224":"#bfa475","226":"#ccac5e","228":"#d5b14e",
    "240":"#aa55f0","242":"#bf72b4","244":"#cc8390","246":"#d58f78","248":"#db9767",
    "260":"#bf40f2","262":"#cc5bc2","264":"#d56ea2","266":"#db7b8a","268":"#df8479",
    "280":"#cc33f4","282":"#d54ccb","284":"#db5eae","286":"#df6b98","288":"#e37687",
    "400":"#00ffdb","402":"#55ed92","404":"#80e46e","406":"#99df58","408":"#aadb49",
    "420":"#55aae5","422":"#80b2ac","424":"#99b68a","426":"#aaba73","428":"#b6bc62",
    "440":"#8080eb","442":"#998ebc","444":"#aa989c","446":"#b69f86","448":"#bfa475",
    "460":"#9966ee","462":"#aa77c6","464":"#b682aa","466":"#bf8b95","468":"#c69284",
    "480":"#aa55f0","482":"#b666cd","484":"#bf72b4","486":"#c67ca0","488":"#cc8390",
    "600":"#00ffdb","602":"#40f2a4","604":"#66e983","606":"#80e46e","608":"#92e05e",
    "620":"#40bfe3","622":"#66c1b5","624":"#80c397","626":"#92c382","628":"#9fc471",
    "640":"#6699e7","642":"#80a1c1","644":"#92a7a5","646":"#9fab91","648":"#aaae81",
    "660":"#8080eb","662":"#928ac9","664":"#9f92b0","666":"#aa989c","668":"#b39d8d",
    "680":"#926ded","682":"#9f79cf","684":"#aa82b8","686":"#b389a6","688":"#b98f97",
    "800":"#00ffdb","802":"#33f4af","804":"#55ed92","806":"#6de87d","808":"#80e46e",
    "820":"#33cce1","822":"#55ccbc","824":"#6dcba1","826":"#80cb8d","828":"#8ecb7d",
    "840":"#55aae5","842":"#6daec5","844":"#80b2ac","846":"#8eb499","848":"#99b68a",
    "860":"#6d92e8","862":"#8099cb","864":"#8e9eb5","866":"#99a2a3","868":"#a2a694",
    "880":"#8080eb","882":"#8e88d0","884":"#998ebc","886":"#a294ab","888":"#aa989c"
}

GTF_COLS = [
    "seqname", "source", "feature", "start", "end",
    "score", "strand", "frame", "attribute"
]


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
    bw = pyBigWig.open(bw_path)
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


def get_attr(attr, key):
    m = re.search(fr'{key} "([^"]+)"', str(attr))
    return m.group(1) if m else None


def _clean_locus_df(df, source_name="uploaded data"):
    df = dedup_columns(df).copy()

    # ---------- normalize raw column names ----------
    df.columns = pd.Index(df.columns).map(
        lambda x: str(x).replace("\ufeff", "").strip()
    )

    # 保存原始列名信息，方便调试
    original_cols = list(df.columns)

    # 建一个不区分大小写的查找表
    lower_to_actual = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl not in lower_to_actual:
            lower_to_actual[cl] = c

    def pick(*aliases):
        """Return the first existing actual column name from aliases (case-insensitive)."""
        for a in aliases:
            key = str(a).strip().lower()
            if key in lower_to_actual:
                return lower_to_actual[key]
        return None

    rename_map = {}

    # ---------- standard required fields ----------
    c = pick("CHR", "chr", "#chr", "chrom", "chromosome")
    if c and c != "CHR":
        rename_map[c] = "CHR"

    c = pick("BP", "bp", "pos", "position", "base_pair_location")
    if c and c != "BP":
        rename_map[c] = "BP"

    c = pick("P", "p", "pval", "pvalue", "p_value", "P-value", "P_VALUE")
    if c and c != "P":
        rename_map[c] = "P"

    # ---------- rsid / snp ----------
    c = pick("rsid", "RSID", "SNP", "snp", "MarkerName", "markername", "ID", "id")
    if c and c != "rsid":
        rename_map[c] = "rsid"

    # ---------- alleles ----------
    # 你的这份数据里就是 ALLELE1 / ALLELE0
    c = pick("A1", "a1", "EA", "ea", "effect_allele", "effect allele", "ALLELE1", "allele1")
    if c and c != "A1":
        rename_map[c] = "A1"

    c = pick(
        "A2",
        "a2",
        "NEA",
        "nea",
        "other_allele",
        "non_effect_allele",
        "non effect allele",
        "ALLELE0",
        "allele0",
    )
    if c and c != "A2":
        rename_map[c] = "A2"

    # ---------- optional/common fields ----------
    c = pick("BETA", "beta", "Effect", "effect", "estimate")
    if c and c != "BETA":
        rename_map[c] = "BETA"

    c = pick("SE", "se", "StdErr", "stderr", "standard_error")
    if c and c != "SE":
        rename_map[c] = "SE"

    c = pick("A1FREQ", "a1freq", "EAF", "eaf", "MAF", "maf", "freq")
    if c and c != "A1FREQ":
        rename_map[c] = "A1FREQ"

    c = pick("N", "n", "N_incl", "n_incl", "samplesize", "sample_size")
    if c and c != "N":
        rename_map[c] = "N"

    df = df.rename(columns=rename_map)

    # ---------- clean values ----------
    if "CHR" in df.columns:
        df["CHR"] = (
            df["CHR"]
            .astype(str)
            .str.replace("^chr", "", regex=True)
            .str.strip()
        )

    for col in ["BP", "P", "BETA", "SE", "A1FREQ", "N"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["A1", "A2"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.upper()
            )

    if "rsid" in df.columns:
        df["rsid"] = df["rsid"].astype(str).str.strip()

    # ---------- required columns ----------
    required = ["CHR", "BP", "P", "A1", "A2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[DEBUG] {source_name} original columns: {original_cols}", flush=True)
        print(f"[DEBUG] {source_name} normalized columns: {list(df.columns)}", flush=True)
        raise ValueError(f"{source_name} missing columns: {missing}")

    # ---------- optional helper column ----------
    # 如果没有 posID，就自动补一个 chr:bp
    if "posID" not in df.columns:
        df["posID"] = df["CHR"].astype(str) + ":" + df["BP"].astype("Int64").astype(str)

    # 你后面做 uniqueid 匹配时会用到
    if "uniqueid" not in df.columns:
        df["uniqueid"] = (
            df["CHR"].astype(str)
            + ":"
            + df["BP"].astype("Int64").astype(str)
            + ":"
            + df["A1"].astype(str)
            + ":"
            + df["A2"].astype(str)
        )

    # ---------- DISPLAY_ID (for UI / hover) ----------
    if "DISPLAY_ID" not in df.columns:
        if "rsid" in df.columns:
            df["DISPLAY_ID"] = df["rsid"].astype(str).str.strip()
        elif "SNP" in df.columns:
            df["DISPLAY_ID"] = df["SNP"].astype(str).str.strip()
        elif "uniqueid" in df.columns:
            df["DISPLAY_ID"] = df["uniqueid"].astype(str).str.strip()
        elif "posID" in df.columns:
            df["DISPLAY_ID"] = df["posID"].astype(str).str.strip()
        else:
            df["DISPLAY_ID"] = (
                df["CHR"].astype(str)
                + ":"
                + pd.to_numeric(df["BP"], errors="coerce").astype("Int64").astype(str)
            )

    return df


@st.cache_data(show_spinner=False)
def load_locus_csv(path):
    log(f"loading csv from path: {path}")
    df = pd.read_csv(path)
    print("RAW COLUMNS:", [repr(c) for c in df.columns], flush=True)
    df = _clean_locus_df(df, source_name=path)
    log(f"{path} loaded, shape={df.shape}")
    return df


@st.cache_data(show_spinner=False)
def load_locus_csv_uploaded(file_bytes, file_name):
    log(f"loading uploaded summary-statistic file: {file_name}")
    raw = BytesIO(file_bytes)
    raw.name = file_name
    df = read_summary_stats_file(raw)
    print("RAW COLUMNS:", [repr(c) for c in df.columns], flush=True)
    df = _clean_locus_df(df, source_name=file_name)
    log(f"{file_name} loaded, shape={df.shape}")
    return df


def get_uploaded_summary_file_kind(file_name):
    """Return the supported uploaded summary-statistic file kind."""
    if not file_name:
        return None
    name = os.path.basename(str(file_name)).lower()
    for suffix in [".csv.gz", ".tsv.gz", ".txt.gz", ".csv", ".tsv", ".txt"]:
        if name.endswith(suffix):
            return suffix.lstrip(".")
    return None


def _read_uploaded_table_once(file_obj, sep, compression=None, engine=None):
    file_obj.seek(0)
    kwargs = {"sep": sep}
    if compression:
        kwargs["compression"] = compression
    if engine:
        kwargs["engine"] = engine
    return pd.read_csv(file_obj, **kwargs)


def read_summary_stats_file(file_obj):
    """Read an uploaded summary-statistic file in CSV, TSV, TXT, or gzip form."""
    kind = get_uploaded_summary_file_kind(getattr(file_obj, "name", ""))
    unsupported_message = (
        "Unsupported summary-statistic file format. Please upload .csv, .tsv, "
        ".txt, .csv.gz, .tsv.gz, or .txt.gz."
    )
    if kind is None:
        raise ValueError(unsupported_message)

    compression = "gzip" if kind.endswith(".gz") else None

    if kind in {"csv", "csv.gz"}:
        return _read_uploaded_table_once(file_obj, sep=",", compression=compression)

    if kind in {"tsv", "tsv.gz"}:
        return _read_uploaded_table_once(file_obj, sep="\t", compression=compression)

    if kind in {"txt", "txt.gz"}:
        attempts = [
            {"sep": "\t", "engine": None},
            {"sep": r"\s+", "engine": "python"},
            {"sep": ",", "engine": None},
        ]
        errors = []
        for attempt in attempts:
            try:
                df = _read_uploaded_table_once(
                    file_obj,
                    sep=attempt["sep"],
                    compression=compression,
                    engine=attempt["engine"],
                )
                if df.shape[1] >= 2:
                    return df
            except Exception as e:
                errors.append(str(e))
        detail = f" Parser errors: {'; '.join(errors)}" if errors else ""
        raise ValueError(
            "Could not parse TXT summary-statistic file. Please use tab-delimited, "
            f"whitespace-delimited, or comma-delimited text with a header row.{detail}"
        )

    raise ValueError(unsupported_message)


def make_uploaded_dataset_signature(uploaded_top, uploaded_bottom):
    """Return a stable upload signature, or None when no summary file is uploaded."""
    if uploaded_top is None and uploaded_bottom is None:
        return None

    def file_sig(uploaded):
        if uploaded is None:
            return None
        data = uploaded.getvalue()
        return {
            "name": getattr(uploaded, "name", ""),
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    return {
        "top": file_sig(uploaded_top),
        "bottom": file_sig(uploaded_bottom),
    }


def make_dataset_title_from_filename(file_name):
    """Create a plot title from an uploaded summary-stat filename."""
    if not file_name:
        return ""
    title = os.path.basename(str(file_name)).strip()
    if not title:
        return ""
    lower_title = title.lower()
    for suffix in [".csv.gz", ".tsv.gz", ".txt.gz", ".csv", ".tsv", ".txt", ".gz"]:
        if lower_title.endswith(suffix):
            title = title[: -len(suffix)]
            break
    return title.strip()


def summarize_locus_dataset(df, label):
    """Summarize usable locus rows for upload-driven chromosome/BP inference."""
    summary = {
        "label": label,
        "usable": False,
        "chrom_counts": {},
        "best_rows": {},
        "median_bp": {},
        "warnings": [],
    }
    if df is None or df.empty:
        summary["warnings"].append(f"{label} dataset is empty.")
        return summary
    missing = [c for c in ["CHR", "BP"] if c not in df.columns]
    if missing:
        summary["warnings"].append(f"{label} dataset missing columns: {missing}")
        return summary

    work = df.copy()
    work["CHR_NORM"] = work["CHR"].map(normalize_chrom)
    work["BP_NUM"] = pd.to_numeric(work["BP"], errors="coerce")
    if "P" in work.columns:
        work["P_NUM"] = pd.to_numeric(work["P"], errors="coerce")
    else:
        work["P_NUM"] = np.nan

    work = work[
        work["CHR_NORM"].isin(get_supported_chromosomes())
        & work["BP_NUM"].notna()
    ].copy()
    if work.empty:
        summary["warnings"].append(f"{label} dataset has no valid chr1-22/X BP rows.")
        return summary

    summary["usable"] = True
    summary["chrom_counts"] = {
        str(chrom): int(count)
        for chrom, count in work.groupby("CHR_NORM").size().to_dict().items()
    }

    for chrom, chrom_df in work.groupby("CHR_NORM"):
        valid_p = chrom_df[
            chrom_df["P_NUM"].notna()
            & (chrom_df["P_NUM"] > 0)
            & (chrom_df["P_NUM"] <= 1)
        ].copy()
        if not valid_p.empty:
            best = valid_p.sort_values(["P_NUM", "BP_NUM"], ascending=[True, True]).iloc[0]
            best_p = float(best["P_NUM"])
        else:
            best = chrom_df.sort_values("BP_NUM").iloc[len(chrom_df) // 2]
            best_p = None
        summary["best_rows"][str(chrom)] = {
            "bp": int(best["BP_NUM"]),
            "p": best_p,
        }
        summary["median_bp"][str(chrom)] = int(round(float(chrom_df["BP_NUM"].median())))

    return summary


def choose_sync_chromosome(top_summary, bottom_summary):
    """Choose the chromosome most compatible with uploaded/active datasets."""
    usable_summaries = [s for s in [top_summary, bottom_summary] if s and s.get("usable")]
    if not usable_summaries:
        return None, "none"

    top_chroms = set((top_summary or {}).get("chrom_counts", {}).keys())
    bottom_chroms = set((bottom_summary or {}).get("chrom_counts", {}).keys())
    shared = top_chroms & bottom_chroms
    candidate_chroms = shared if shared else set().union(*(set(s["chrom_counts"].keys()) for s in usable_summaries))

    def score(chrom):
        count = 0
        best_p_values = []
        for summary in usable_summaries:
            count += int(summary["chrom_counts"].get(chrom, 0))
            p = summary.get("best_rows", {}).get(chrom, {}).get("p")
            if p is not None:
                best_p_values.append(float(p))
        best_p = min(best_p_values) if best_p_values else 1.0
        shared_bonus = 1 if chrom in shared else 0
        return (shared_bonus, count, -best_p, -chrom_sort_key(chrom))

    chrom = max(candidate_chroms, key=score)
    if chrom in top_chroms and chrom in bottom_chroms:
        source = "shared"
    elif chrom in top_chroms:
        source = "top"
    else:
        source = "bottom"
    return chrom, source


def choose_sync_center_bp(df_top, df_bottom, chrom):
    """Choose center BP from the best valid P row on chrom, falling back to median BP."""
    candidates = []
    medians = []
    for label, df in [("top", df_top), ("bottom", df_bottom)]:
        if df is None or df.empty or "CHR" not in df.columns or "BP" not in df.columns:
            continue
        work = df.loc[chrom_mask(df, chrom)].copy()
        if work.empty:
            continue
        work["BP_NUM"] = pd.to_numeric(work["BP"], errors="coerce")
        work = work[work["BP_NUM"].notna()].copy()
        if work.empty:
            continue
        medians.append((label, int(round(float(work["BP_NUM"].median())))))
        if "P" in work.columns:
            work["P_NUM"] = pd.to_numeric(work["P"], errors="coerce")
            valid_p = work[
                work["P_NUM"].notna()
                & (work["P_NUM"] > 0)
                & (work["P_NUM"] <= 1)
            ].copy()
            if not valid_p.empty:
                best = valid_p.sort_values(["P_NUM", "BP_NUM"], ascending=[True, True]).iloc[0]
                source_priority = 0 if label == "top" else 1
                candidates.append((float(best["P_NUM"]), source_priority, label, int(best["BP_NUM"])))

    if candidates:
        _, _, source_label, bp = min(candidates)
        return bp, source_label
    if medians:
        source_label, bp = medians[0]
        return bp, source_label
    return None, None


def recommend_y_axis_max_for_dataset(
    df,
    chrom=None,
    center_bp=None,
    window_kb=None,
    min_default=7.0,
    pad_frac=0.12,
):
    """Recommend a y-axis max from -log10(P) for the active uploaded locus.

    Lightweight only. Does not run LD, PLINK, clumping, or figure building.
    """
    try:
        if df is None or df.empty or "P" not in df.columns:
            return float(min_default)

        x = df.copy()

        if chrom is not None and "CHR" in x.columns:
            chrom_str = normalize_chrom(chrom)
            x["_CHR_NORM"] = x["CHR"].map(normalize_chrom)
            x = x[x["_CHR_NORM"] == chrom_str]

        if center_bp is not None and window_kb is not None and "BP" in x.columns:
            bp = pd.to_numeric(x["BP"], errors="coerce")
            center = int(center_bp)
            half_window = int(float(window_kb) * 1000)
            x = x[(bp >= center - half_window) & (bp <= center + half_window)]

        p = pd.to_numeric(x["P"], errors="coerce")
        p = p[np.isfinite(p) & (p > 0)]
        if p.empty:
            return float(min_default)

        y = -np.log10(p)
        y = y[np.isfinite(y)]
        if y.empty:
            return float(min_default)

        ymax = float(np.nanmax(y))
        recommended = max(float(min_default), ymax * (1.0 + float(pad_frac)) + 0.5)

        # Keep a clean display value.
        if recommended <= 20:
            return float(np.ceil(recommended))
        return float(np.ceil(recommended / 5.0) * 5.0)
    except Exception:
        return float(min_default)


def infer_uploaded_locus_sync(df_top, df_bottom, uploaded_top, uploaded_bottom):
    """Infer upload-compatible locus controls without touching LD or plot state."""
    top_uploaded = uploaded_top is not None
    bottom_uploaded = uploaded_bottom is not None
    top_sync_df = df_top if top_uploaded else None
    bottom_sync_df = df_bottom if bottom_uploaded else None
    top_summary = summarize_locus_dataset(top_sync_df, "top") if top_uploaded else None
    bottom_summary = summarize_locus_dataset(bottom_sync_df, "bottom") if bottom_uploaded else None

    chrom, chrom_source = choose_sync_chromosome(top_summary, bottom_summary)
    if chrom is None:
        return {
            "status": "warning",
            "applied": False,
            "message": "Uploaded data were detected, but no valid CHR/BP/P rows could be used for automatic locus sync. Existing locus controls were left unchanged.",
        }

    bp, bp_source = choose_sync_center_bp(top_sync_df, bottom_sync_df, chrom)
    if bp is None:
        return {
            "status": "warning",
            "applied": False,
            "message": "Uploaded data were detected, but no valid CHR/BP/P rows could be used for automatic locus sync. Existing locus controls were left unchanged.",
        }

    preferred_source = "top" if top_uploaded and chrom in top_summary.get("chrom_counts", {}) else "bottom"
    if preferred_source == "bottom" and not bottom_uploaded:
        preferred_source = bp_source or "top"
    index_method = (
        "Auto-select by LD clumping from dataset 1 (top)"
        if preferred_source == "top"
        else "Auto-select by LD clumping from dataset 2 (bottom)"
    )

    no_shared = (
        top_uploaded
        and bottom_uploaded
        and chrom_source != "shared"
    )
    if no_shared:
        message = (
            f"Uploaded datasets do not share a chromosome; synced to {format_chrom_label(chrom)} "
            f"from the {preferred_source} dataset. Check settings before updating."
        )
        status = "warning"
    else:
        message = (
            f"New uploaded data detected. Locus controls synced to {format_chrom_label(chrom)}:{int(bp):,}; "
            f"index selection switched to auto-select from {preferred_source} dataset. "
            "Select Update plot to apply."
        )
        status = "info"

    return {
        "status": status,
        "applied": True,
        "chrom": normalize_chrom(chrom),
        "bp": int(bp),
        "index_selection_method": index_method,
        "preferred_source": preferred_source,
        "message": message,
    }


def apply_uploaded_dataset_sync_if_new(
    signature,
    sync_result,
    df_top=None,
    df_bottom=None,
    uploaded_top=None,
    uploaded_bottom=None,
):
    """Apply upload-inferred state once per new uploaded dataset signature."""
    if signature is None:
        return False
    if st.session_state.get("last_uploaded_dataset_signature") == signature:
        return False

    if sync_result.get("applied"):
        st.session_state["chrom"] = sync_result["chrom"]
        st.session_state["bp"] = int(sync_result["bp"])
        st.session_state["index_selection_method"] = sync_result["index_selection_method"]
        if uploaded_top is not None:
            st.session_state["title_top"] = make_dataset_title_from_filename(uploaded_top.name)
        if uploaded_bottom is not None:
            st.session_state["title_bottom"] = make_dataset_title_from_filename(uploaded_bottom.name)

        synced_chrom = sync_result["chrom"]
        synced_bp = int(sync_result["bp"])
        synced_window_kb = st.session_state.get("window_kb", 500)

        st.session_state["max_ylim_top"] = recommend_y_axis_max_for_dataset(
            df_top,
            chrom=synced_chrom,
            center_bp=synced_bp,
            window_kb=synced_window_kb,
            min_default=7.0,
            pad_frac=0.12,
        )
        st.session_state["max_ylim_bottom"] = recommend_y_axis_max_for_dataset(
            df_bottom,
            chrom=synced_chrom,
            center_bp=synced_bp,
            window_kb=synced_window_kb,
            min_default=7.0,
            pad_frac=0.12,
        )

        st.session_state["rsid"] = ""
        st.session_state["rsid_2"] = ""
        st.session_state["rsid_3"] = ""

    st.session_state["last_uploaded_dataset_signature"] = signature
    st.session_state["last_upload_sync_result"] = sync_result
    return True


def format_upload_sync_message(sync_result):
    if not sync_result:
        return ""
    return str(sync_result.get("message", ""))


@st.cache_data(show_spinner=False)
def read_uploaded_ld_long(file_bytes, file_name):
    """
    Parse an uploaded LD long-format table. Returns:
      ld_long:     DataFrame with canonical columns SNP_A, SNP_B, R2 (R2 in [0,1])
      ld_universe: tuple of SNP IDs known to the LD reference, taken from
                   self-pair rows where SNP_A == SNP_B and R2 == 1.
    Raises ValueError if required columns or self-pair rows are missing.
    """
    from io import BytesIO
    raw = BytesIO(file_bytes)
    name = (file_name or "").lower()
    if name.endswith((".tsv", ".tab", ".txt", ".ld")):
        df = pd.read_csv(raw, sep=r"\s+", engine="python")
    else:
        try:
            df = pd.read_csv(raw, sep=None, engine="python")
        except Exception:
            raw.seek(0)
            df = pd.read_csv(raw)
    df = dedup_columns(df)

    alias = {
        "snp_a": "SNP_A", "snp1": "SNP_A", "id1": "SNP_A", "snpa": "SNP_A",
        "snp_b": "SNP_B", "snp2": "SNP_B", "id2": "SNP_B", "snpb": "SNP_B",
        "r2": "R2", "rsq": "R2", "r^2": "R2",
    }
    rename = {c: alias[c.lower()] for c in df.columns if c.lower() in alias}
    df = df.rename(columns=rename)

    missing = [c for c in ["SNP_A", "SNP_B", "R2"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Uploaded LD long table missing required columns: {missing}. "
            "Expected SNP_A, SNP_B, R2 (or aliases SNP1/ID1/SNP2/ID2/r2)."
        )

    df = df[["SNP_A", "SNP_B", "R2"]].copy()
    df["SNP_A"] = df["SNP_A"].astype(str).str.strip()
    df["SNP_B"] = df["SNP_B"].astype(str).str.strip()
    df["R2"] = pd.to_numeric(df["R2"], errors="coerce")
    df = df.dropna(subset=["SNP_A", "SNP_B", "R2"])
    df = df[(df["SNP_A"] != "") & (df["SNP_B"] != "")]
    df["R2"] = df["R2"].clip(lower=0.0, upper=1.0)

    self_pairs = df[(df["SNP_A"] == df["SNP_B"]) & np.isclose(df["R2"], 1.0)]
    if self_pairs.empty:
        raise ValueError(
            "Uploaded LD long table must include self-pair rows "
            "SNP_A=SNP_B, R2=1 to distinguish low LD from missing LD."
        )

    ld_universe = tuple(sorted(set(self_pairs["SNP_A"].astype(str))))
    return df, ld_universe


@st.cache_data(show_spinner=False)
def read_uploaded_ld_matrix(file_bytes, file_name):
    """
    Parse an uploaded LD matrix. First column = row SNP IDs, first row =
    column SNP IDs, cells = R^2. Returns the matrix as a DataFrame with
    SNP IDs on both axes, plus a tuple ld_universe = sorted(rows | cols).
    """
    from io import BytesIO
    raw = BytesIO(file_bytes)
    name = (file_name or "").lower()
    if name.endswith((".tsv", ".tab", ".txt", ".ld")):
        df = pd.read_csv(raw, sep=r"\s+", engine="python", index_col=0)
    else:
        try:
            df = pd.read_csv(raw, sep=None, engine="python", index_col=0)
        except Exception:
            raw.seek(0)
            df = pd.read_csv(raw, index_col=0)
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    df = df.apply(pd.to_numeric, errors="coerce")

    row_ids = [s for s in df.index.tolist() if s and s.lower() != "nan"]
    col_ids = [s for s in df.columns.tolist() if s and s.lower() != "nan"]
    if not row_ids and not col_ids:
        raise ValueError("Uploaded LD matrix has no SNP IDs in row or column names.")

    ld_universe = tuple(sorted(set(row_ids) | set(col_ids)))
    return df, ld_universe


def _norm_chr(chrom):
    return normalize_chrom(chrom)


def _normalize_chrom_legacy_autosome_only(chrom):
    """Return chromosome as a clean string without 'chr' prefix.

    Examples: 'chr14' → '14', '14' → '14', 14 → '14'
    """
    return str(chrom).replace("chr", "").strip()


def normalize_chrom(chrom):
    """Return a canonical chromosome string for autosomes and chromosome X."""
    if chrom is None:
        return ""
    try:
        if pd.isna(chrom):
            return ""
    except (TypeError, ValueError):
        pass

    s = str(chrom).strip()
    if not s:
        return ""
    if s.lower().startswith("chr"):
        s = s[3:].strip()
    if s.endswith(".0"):
        s = s[:-2]

    if s.upper() == "X" or s == "23":
        return "X"
    if s.isdigit():
        n = int(s)
        if 1 <= n <= 22:
            return str(n)
    return s


def is_supported_chrom(chrom):
    """Return True for chromosomes supported by the visualizer."""
    return normalize_chrom(chrom) in get_supported_chromosomes()


def get_supported_chromosomes():
    """Return supported autosomes plus chromosome X."""
    return [str(i) for i in range(1, 23)] + ["X"]


def chrom_sort_key(chrom):
    """Sort chromosomes 1-22 numerically, then X."""
    c = normalize_chrom(chrom)
    if c == "X":
        return 23
    try:
        return int(c)
    except (TypeError, ValueError):
        return 10_000


def format_chrom_label(chrom):
    """Return display label such as chr14 or chrX."""
    c = normalize_chrom(chrom)
    return f"chr{c}" if c else "chr"


def format_chrom_axis_title(chrom):
    """Return a locus x-axis title with the selected chromosome."""
    chrom = normalize_chrom(chrom)
    return f"Chromosome {chrom} (Mb)"


def chrom_mask(df, chrom):
    """Return a boolean mask matching df['CHR'] to the selected chromosome.

    Both sides are normalised (chr prefix stripped) before comparison.
    """
    target = normalize_chrom(chrom)
    if "CHR" not in df.columns:
        raise KeyError(f"DataFrame has no 'CHR' column; columns: {list(df.columns)}")
    return df["CHR"].map(normalize_chrom) == target


def normalize_internal_1000g_ancestry(value):
    """Return a supported 1000G super-population code, defaulting to EUR."""
    ancestry = str(value or INTERNAL_1000G_DEFAULT_ANCESTRY).strip().upper()
    return ancestry if ancestry in INTERNAL_1000G_ANCESTRIES else INTERNAL_1000G_DEFAULT_ANCESTRY


def format_internal_1000g_ancestry_option(ancestry):
    ancestry = normalize_internal_1000g_ancestry(ancestry)
    label = INTERNAL_1000G_ANCESTRIES[ancestry].split(" / ", 1)[0]
    return f"{ancestry} - {label}"


def get_internal_1000g_prefix(chrom, ancestry=INTERNAL_1000G_DEFAULT_ANCESTRY):
    ancestry = normalize_internal_1000g_ancestry(ancestry)
    chrom = normalize_chrom(chrom)
    return DATA_DIR / f"1000g_{ancestry}_hg38_high_coverage_Illumina.filtered.SNV_INDEL_SV_phased_panel_include_MHC_ch{chrom}"


def get_internal_bfile_prefix_for_chrom(chrom, ancestry=INTERNAL_1000G_DEFAULT_ANCESTRY):
    """Build the chromosome-specific PLINK bfile prefix and verify all three
    files (.bed, .bim, .fam) exist."""
    chrom = normalize_chrom(chrom)
    ancestry = normalize_internal_1000g_ancestry(ancestry)
    prefix = str(get_internal_1000g_prefix(chrom, ancestry))
    missing = []
    for suf in [".bed", ".bim", ".fam"]:
        if not os.path.exists(prefix + suf):
            missing.append(prefix + suf)
    if missing:
        if chrom == "X":
            message = (
                f"Internal 1000G {ancestry} reference files for chromosome X were not found. "
                "Select an ancestry with chrX support or upload your own LD reference."
            )
        else:
            message = f"Internal 1000G {ancestry} reference files for chromosome {chrom} were not found."
        raise FileNotFoundError(
            message + "\n" +
            "\n".join(f"  {m}" for m in missing)
        )
    return prefix


def get_gtf_path_for_chrom(chrom):
    """Build the chromosome-specific GENCODE GTF path and verify it exists."""
    chrom = normalize_chrom(chrom)
    path = DATA_DIR / f"gencode.v49.annotation.chr{chrom}.gtf.gz"
    if chrom == "X" and not path.is_file():
        full_path = DATA_DIR / "gencode.v49.annotation.gtf.gz"
        if full_path.is_file():
            return str(full_path)
        raise FileNotFoundError(
            f"Missing GENCODE annotation for chromosome X. Add "
            f"{DATA_DIR / 'gencode.v49.annotation.chrX.gtf.gz'} or "
            f"{DATA_DIR / 'gencode.v49.annotation.gtf.gz'}."
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing GENCODE annotation for chromosome {chrom}: {path}"
        )
    return str(path)


def get_required_n_indices(active_mode):
    """Return the number of index variants required by the active mode."""
    if active_mode == "Standard locus zoom":
        return 1
    if active_mode == "Two-index LocusBlend":
        return 2
    return 3


def prepare_clump_candidates(df_ref, selected_chrom, center_bp, window_bp):
    """Filter to selected chromosome and BP window, keep rows with valid P
    and REF_SNP, sort by P ascending.  Returns columns: REF_SNP, DISPLAY_ID,
    CHR, BP, P."""
    df = dedup_columns(df_ref).copy()
    mask = (
        chrom_mask(df, selected_chrom)
        & (df["BP"] >= center_bp - window_bp)
        & (df["BP"] <= center_bp + window_bp)
    )
    cand = df.loc[mask].copy()
    cand["P"] = pd.to_numeric(cand["P"], errors="coerce")
    cand = cand[
        cand["P"].notna() & (cand["P"] > 0) & (cand["P"] <= 1)
    ].copy()
    if "REF_SNP" not in cand.columns:
        raise KeyError("Candidate table missing REF_SNP column — run reference matching first.")
    cand = cand[cand["REF_SNP"].notna()].copy()
    cand = cand.drop_duplicates("REF_SNP").copy()
    cand = cand.sort_values("P").reset_index(drop=True)
    if "DISPLAY_ID" not in cand.columns:
        cand["DISPLAY_ID"] = cand["REF_SNP"]
    return cand[["REF_SNP", "DISPLAY_ID", "CHR", "BP", "P"]]


def run_plink_clump_for_auto_indices(
    candidates,
    bfile_prefix,
    selected_chrom,
    start_bp,
    end_bp,
    max_indices,
    clump_r2=0.01,
    plink_path=None,
):
    """Run PLINK --clump on the candidate list, returning up to max_indices
    independent lead SNPs (ranked by PLINK clump order)."""
    if plink_path is None:
        plink_path = _find_plink_exec()
    bfile_prefix = _resolve_bfile_prefix(bfile_prefix)
    selected_chrom = normalize_chrom(selected_chrom)

    if candidates.empty:
        raise ValueError("No valid candidate SNPs for clumping.")

    window_kb = max(1, int((int(end_bp) - int(start_bp)) / 1000) + 1)

    with tempfile.TemporaryDirectory(prefix="plink_clump_") as tmpdir:
        clump_in = os.path.join(tmpdir, "clump_input.txt")
        extract_in = os.path.join(tmpdir, "extract.snplist")
        out_prefix = os.path.join(tmpdir, "clump_out")

        # Write clump input (SNP P)
        with open(clump_in, "w") as f:
            f.write("SNP P\n")
            for _, row in candidates.iterrows():
                f.write(f"{row['REF_SNP']} {row['P']:.6e}\n")

        # Write extract list (candidate REF_SNPs only)
        with open(extract_in, "w") as f:
            for snp in candidates["REF_SNP"]:
                f.write(f"{snp}\n")

        cmd = [
            plink_path,
            "--threads", "1",
            "--bfile", bfile_prefix,
            "--chr", selected_chrom,
            "--from-bp", str(int(start_bp)),
            "--to-bp", str(int(end_bp)),
            "--extract", extract_in,
            "--clump", clump_in,
            "--clump-snp-field", "SNP",
            "--clump-field", "P",
            "--clump-p1", "1",
            "--clump-p2", "1",
            "--clump-r2", str(clump_r2),
            "--clump-kb", str(window_kb),
            "--out", out_prefix,
        ]
        if selected_chrom == "X":
            cmd.extend(["--allow-extra-chr"])

        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        clumped_path = out_prefix + ".clumped"
        if res.returncode != 0:
            log(f"PLINK clump failed:\n{res.stdout}")
            if not os.path.exists(clumped_path):
                raise ValueError(f"PLINK clumping failed. Output:\n{res.stdout}")

        if not os.path.exists(clumped_path):
            raise ValueError("PLINK clumping produced no .clumped output.")

        clumped = pd.read_csv(clumped_path, sep=r"\s+")
        if clumped.empty or "SNP" not in clumped.columns:
            raise ValueError("PLINK clumping produced an empty result — no independent lead SNPs found.")

        lead_snps = clumped["SNP"].astype(str).str.strip().head(max_indices).tolist()

    # Map back to candidate metadata
    selected_rows = []
    for snp in lead_snps:
        match = candidates[candidates["REF_SNP"].astype(str) == snp]
        if not match.empty:
            selected_rows.append(match.iloc[0].to_dict())

    if not selected_rows:
        raise ValueError("None of the PLINK lead SNPs matched the candidate table.")

    out = pd.DataFrame(selected_rows)
    out["rank"] = range(1, len(out) + 1)
    return out[["rank", "DISPLAY_ID", "REF_SNP", "CHR", "BP", "P"]]


def greedy_clump_uploaded_ld_for_auto_indices(
    candidates,
    max_indices=3,
    clump_r2=0.01,
    ld_long_table=None,
    ld_matrix_table=None,
):
    """Greedy independent-variant selection using uploaded LD.

    Sorts candidates by P ascending, iteratively picks the top-significant
    SNP not yet blocked, then blocks all SNPs with r² > clump_r2 to the
    lead SNP.
    """
    ld_long = ld_long_table
    ld_matrix = ld_matrix_table

    if ld_matrix is not None:
        rows_set = {str(x) for x in ld_matrix.index}
        cols_set = {str(x) for x in ld_matrix.columns}
        ld_universe = rows_set | cols_set

        def get_partners(ref_snp):
            partners = {}
            s = str(ref_snp)
            if s in rows_set:
                vec = ld_matrix.loc[s]
                if isinstance(vec, pd.DataFrame):
                    vec = vec.iloc[0]
                vec = pd.to_numeric(vec, errors="coerce").dropna()
                partners.update({str(k): float(v) for k, v in vec.items()})
            if s in cols_set:
                # column lookup
                vec = ld_matrix[s]
                if isinstance(vec, pd.DataFrame):
                    vec = vec.iloc[:, 0]
                vec = pd.to_numeric(vec, errors="coerce").dropna()
                vec.index = ld_matrix.index.astype(str)
                for k, v in vec.items():
                    if str(k) not in partners:
                        partners[str(k)] = float(v)
            return partners
    elif ld_long is not None:
        ld_universe_set = set()
        ld_map = {}
        for _, row in ld_long.iterrows():
            a, b, r = str(row["SNP_A"]), str(row["SNP_B"]), float(row["R2"])
            ld_universe_set.update([a, b])
            ld_map.setdefault(a, {})[b] = r
            ld_map.setdefault(b, {})[a] = r
        ld_universe = ld_universe_set

        def get_partners(ref_snp):
            return ld_map.get(str(ref_snp), {})
    else:
        raise ValueError("No uploaded LD table provided for greedy clumping.")

    # Filter to candidates present in LD universe
    cand = candidates[candidates["REF_SNP"].astype(str).isin(ld_universe)].copy()
    if cand.empty:
        raise ValueError("No candidate SNPs found in the uploaded LD reference.")

    blocked = set()
    selected = []

    for _, row in cand.iterrows():
        s = str(row["REF_SNP"])
        if s in blocked:
            continue
        selected.append(row.to_dict())
        if len(selected) >= max_indices:
            break
        partners = get_partners(s)
        for partner, r in partners.items():
            if r > clump_r2:
                blocked.add(partner)

    if not selected:
        raise ValueError(
            f"No independent lead SNPs found at r²={clump_r2} "
            f"within the selected locus window."
        )

    out = pd.DataFrame(selected)
    out["rank"] = range(1, len(out) + 1)
    needed_cols = ["rank", "DISPLAY_ID", "REF_SNP", "CHR", "BP", "P"]
    for c in needed_cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[needed_cols]


def auto_select_index_variants_by_clumping(
    df_source_ref,
    selected_chrom,
    center_bp,
    window_bp,
    required_n,
    active_ld_source,
    bfile_prefix=None,
    plink_path=None,
    ld_long_table=None,
    ld_matrix_table=None,
    clump_r2=0.01,
):
    """Dispatch to PLINK or greedy clumping, returning (selected_rows, summary_dict)."""
    candidates = prepare_clump_candidates(
        df_source_ref, selected_chrom, center_bp, window_bp,
    )
    if candidates.empty:
        raise ValueError(
            f"No valid candidate SNPs found in chr{selected_chrom}:"
            f"{center_bp - window_bp}-{center_bp + window_bp}. "
            "Check the chromosome selection, window size, and that the "
            "dataset has been matched to an LD reference."
        )

    start_bp = int(center_bp - window_bp)
    end_bp = int(center_bp + window_bp)

    if active_ld_source == "Use internal 1000G reference":
        selected = run_plink_clump_for_auto_indices(
            candidates=candidates,
            bfile_prefix=bfile_prefix,
            selected_chrom=selected_chrom,
            start_bp=start_bp,
            end_bp=end_bp,
            max_indices=required_n,
            clump_r2=clump_r2,
            plink_path=plink_path,
        )
    else:
        selected = greedy_clump_uploaded_ld_for_auto_indices(
            candidates=candidates,
            max_indices=required_n,
            clump_r2=clump_r2,
            ld_long_table=ld_long_table,
            ld_matrix_table=ld_matrix_table,
        )

    if len(selected) < required_n:
        raise ValueError(
            f"Only {len(selected)} independent index variant(s) were found "
            f"at r² = {clump_r2} within chr{selected_chrom}:"
            f"{start_bp}-{end_bp}. "
            f"The '{st.session_state.get('active_locusblend_mode', 'selected')}' mode "
            f"requires {required_n}. Please enlarge the window, switch to "
            f"a mode requiring fewer index variants, or use manual input."
        )

    n_candidates = len(candidates)
    summary = {
        "n_candidates": n_candidates,
        "n_selected": len(selected),
        "clump_r2": clump_r2,
        "selected_chrom": selected_chrom,
        "start_bp": start_bp,
        "end_bp": end_bp,
    }
    return selected, summary


def _normalize_bfile_prefix(p):
    p = str(p).strip()
    for suf in [".bed", ".bim", ".fam"]:
        if p.endswith(suf):
            p = p[:-len(suf)]
    return p


def _resolve_bfile_prefix(bfile_prefix):
    raw = str(bfile_prefix).strip()
    prefix = _normalize_bfile_prefix(raw)

    search_dirs = [DATA_DIR, APP_DIR]

    candidates = []
    if prefix:
        candidates.append(prefix)
        for d in search_dirs:
            candidates.append(str(d / os.path.basename(prefix)))

    for d in search_dirs:
        for bim in sorted(glob.glob(str(d / "*.bim"))):
            candidates.append(bim[:-4])

    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)

    for c in uniq:
        bed = c + ".bed"
        bim = c + ".bim"
        fam = c + ".fam"
        if os.path.exists(bed) and os.path.exists(bim) and os.path.exists(fam):
            log(f"Resolved PLINK prefix: {c}")
            return c

    debug_files = []
    for d in search_dirs:
        debug_files.extend(sorted(glob.glob(str(d / "*"))))
    raise FileNotFoundError(
        "Could not resolve a valid PLINK bfile prefix. "
        f"Input was: {raw!r}. "
        f"Searched directories: {[str(d) for d in search_dirs]}. "
        f"Available files include: {[os.path.basename(x) for x in debug_files[:50]]}"
    )


def _find_plink_exec(plink_path=str(BIN_DIR / "plink")):
    candidates = [str(plink_path).strip(), str(BIN_DIR / "plink"), "./plink", "plink"]
    for c in candidates:
        if c and (os.path.exists(c) or shutil.which(c)):
            return c
    raise FileNotFoundError("PLINK executable not found. Please install plink first.")


@st.cache_data(show_spinner=False)
def load_reference_bim(bfile_prefix):
    bfile_prefix = _resolve_bfile_prefix(bfile_prefix)
    bim_path = f"{bfile_prefix}.bim"

    log(f"Using BIM path: {bim_path}")
    log(f"cwd: {os.getcwd()}")

    bim = pd.read_csv(
        bim_path,
        sep=r"\s+",
        header=None,
        names=["CHR", "SNP", "CM", "BP", "A1", "A2"]
    )
    bim["CHR"] = bim["CHR"].map(normalize_chrom)
    bim["BP"] = pd.to_numeric(bim["BP"], errors="coerce")
    bim["A1"] = bim["A1"].astype(str).str.upper()
    bim["A2"] = bim["A2"].astype(str).str.upper()
    bim["SNP"] = bim["SNP"].astype(str)
    bim = bim.dropna(subset=["BP"]).copy()
    return bim


def build_bim_uid_table(bim):
    bim = dedup_columns(bim).copy()
    bim["CHR"] = bim["CHR"].map(normalize_chrom)
    bim["BP"] = pd.to_numeric(bim["BP"], errors="coerce")
    bim["A1"] = bim["A1"].astype(str).str.upper()
    bim["A2"] = bim["A2"].astype(str).str.upper()
    bim["SNP"] = bim["SNP"].astype(str)

    bim["UID"] = (
        bim["CHR"].astype(str) + ":" +
        bim["BP"].astype("Int64").astype(str) + ":" +
        bim["A1"].astype(str) + ":" +
        bim["A2"].astype(str)
    )
    return bim


def attach_reference_snp_two_pass(df, bim_ref):
    """
    Match input summary stats to BIM reference in two passes:
    1) CHR:BP:A1:A2
    2) CHR:BP:A2:A1

    Returns original df plus:
      - QUERY_UID
      - REF_UID
      - REF_UID_FLIP
      - REF_MATCH
      - REF_MATCH_TYPE   ("forward", "flip", or NA)
      - NEED_FLIP        (True if matched on swapped alleles)
    """

    out = df.copy()
    ref = bim_ref.copy()

    # ---------- helper ----------
    def norm_chr(s):
        return s.map(normalize_chrom)

    def norm_allele(s):
        return (
            s.astype(str)
             .str.strip()
             .str.upper()
        )

    # ---------- make sure df has required cols ----------
    required_df = ["CHR", "BP", "A1", "A2"]
    miss_df = [c for c in required_df if c not in out.columns]
    if miss_df:
        raise ValueError(f"input df missing columns for reference matching: {miss_df}")

    out["CHR"] = norm_chr(out["CHR"])
    out["BP"] = pd.to_numeric(out["BP"], errors="coerce")
    out["A1"] = norm_allele(out["A1"])
    out["A2"] = norm_allele(out["A2"])

    # 用 Int64 防止 NA 时直接崩
    out["BP"] = out["BP"].astype("Int64")

    # ---------- build QUERY_UID on df ----------
    out["QUERY_UID"] = (
        out["CHR"].astype(str) + ":"
        + out["BP"].astype(str) + ":"
        + out["A1"].astype(str) + ":"
        + out["A2"].astype(str)
    )

    out["QUERY_UID_FLIP"] = (
        out["CHR"].astype(str) + ":"
        + out["BP"].astype(str) + ":"
        + out["A2"].astype(str) + ":"
        + out["A1"].astype(str)
    )

    # ---------- normalize bim_ref ----------
    # 兼容常见 BIM 列名
    rename_map = {}
    if "#CHROM" in ref.columns and "CHR" not in ref.columns:
        rename_map["#CHROM"] = "CHR"
    if "chr" in ref.columns and "CHR" not in ref.columns:
        rename_map["chr"] = "CHR"
    if "pos" in ref.columns and "BP" not in ref.columns:
        rename_map["pos"] = "BP"
    if "bp" in ref.columns and "BP" not in ref.columns:
        rename_map["bp"] = "BP"
    if "a1" in ref.columns and "A1" not in ref.columns:
        rename_map["a1"] = "A1"
    if "a2" in ref.columns and "A2" not in ref.columns:
        rename_map["a2"] = "A2"
    if "snp" in ref.columns and "SNP" not in ref.columns:
        rename_map["snp"] = "SNP"
    if "id" in ref.columns and "SNP" not in ref.columns:
        rename_map["id"] = "SNP"

    ref = ref.rename(columns=rename_map)

    required_ref = ["CHR", "BP", "A1", "A2"]
    miss_ref = [c for c in required_ref if c not in ref.columns]
    if miss_ref:
        raise ValueError(f"bim_ref missing columns for reference matching: {miss_ref}")

    ref["CHR"] = norm_chr(ref["CHR"])
    ref["BP"] = pd.to_numeric(ref["BP"], errors="coerce").astype("Int64")
    ref["A1"] = norm_allele(ref["A1"])
    ref["A2"] = norm_allele(ref["A2"])

    # BIM 里的 SNP 名
    if "SNP" not in ref.columns:
        ref["SNP"] = (
            ref["CHR"].astype(str) + ":"
            + ref["BP"].astype(str) + ":"
            + ref["A1"].astype(str) + ":"
            + ref["A2"].astype(str)
        )

    # 正向 UID
    ref["REF_UID"] = (
        ref["CHR"].astype(str) + ":"
        + ref["BP"].astype(str) + ":"
        + ref["A1"].astype(str) + ":"
        + ref["A2"].astype(str)
    )

    # 反向 UID
    ref["REF_UID_FLIP"] = (
        ref["CHR"].astype(str) + ":"
        + ref["BP"].astype(str) + ":"
        + ref["A2"].astype(str) + ":"
        + ref["A1"].astype(str)
    )

    # ---------- first pass: forward ----------
    m1 = (
        ref[["REF_UID", "SNP"]]
        .drop_duplicates("REF_UID")
        .rename(columns={
            "REF_UID": "QUERY_UID",
            "SNP": "REF_MATCH_FWD",
        })
    )

    out = out.merge(m1, on="QUERY_UID", how="left")

    # ---------- second pass: allele flipped ----------
    m2 = (
        ref[["REF_UID_FLIP", "SNP"]]
        .drop_duplicates("REF_UID_FLIP")
        .rename(columns={
            "REF_UID_FLIP": "QUERY_UID",
            "SNP": "REF_MATCH_REV",
        })
    )

    out = out.merge(
        m2,
        left_on="QUERY_UID",
        right_on="QUERY_UID",
        how="left",
    )

    # ---------- combine ----------
    out["REF_MATCH"] = out["REF_MATCH_FWD"]
    out.loc[out["REF_MATCH"].isna(), "REF_MATCH"] = out.loc[
        out["REF_MATCH"].isna(), "REF_MATCH_REV"
    ]

    out["REF_MATCH_TYPE"] = pd.NA
    out.loc[out["REF_MATCH_FWD"].notna(), "REF_MATCH_TYPE"] = "forward"
    out.loc[
        out["REF_MATCH_FWD"].isna() & out["REF_MATCH_REV"].notna(),
        "REF_MATCH_TYPE",
    ] = "flip"

    out["NEED_FLIP"] = out["REF_MATCH_TYPE"].eq("flip")

    # backward compatibility
    out["REF_SNP"] = out["REF_MATCH"]

    # keep REF_UID columns for downstream use/debugging
    out["REF_UID"] = out["QUERY_UID"]
    out["REF_UID_FLIP"] = out["QUERY_UID_FLIP"]

    return out


def resolve_index_variant_from_input(df_top_ref, df_bottom_ref, user_text):
    """
    Resolve user-entered SNP text against dataset 1 (top) and dataset 2 (bottom).

    Matching priority:
      1) DISPLAY_ID
      2) rsid
      3) SNP
      4) REF_MATCH
      5) uniqueid
      6) QUERY_UID

    Returns one matched row (as Series).
    """

    user_text = str(user_text).strip()
    if user_text == "":
        raise ValueError("Empty index SNP input.")

    def prep(df, source_label):
        x = df.copy()

        # 保证一些常见列存在时先转成字符串
        for col in ["DISPLAY_ID", "rsid", "SNP", "REF_MATCH", "uniqueid", "QUERY_UID"]:
            if col in x.columns:
                x[col] = x[col].astype(str).str.strip()

        # 如果没有 uniqueid，就现建一个
        if "uniqueid" not in x.columns and all(c in x.columns for c in ["CHR", "BP", "A1", "A2"]):
            x["uniqueid"] = (
                x["CHR"].map(normalize_chrom)
                + ":"
                + pd.to_numeric(x["BP"], errors="coerce").astype("Int64").astype(str)
                + ":"
                + x["A1"].astype(str).str.strip().str.upper()
                + ":"
                + x["A2"].astype(str).str.strip().str.upper()
            )

        # 自动生成 DISPLAY_ID
        if "DISPLAY_ID" not in x.columns:
            if "rsid" in x.columns:
                x["DISPLAY_ID"] = x["rsid"]
            elif "SNP" in x.columns:
                x["DISPLAY_ID"] = x["SNP"]
            elif "REF_MATCH" in x.columns:
                x["DISPLAY_ID"] = x["REF_MATCH"]
            elif "uniqueid" in x.columns:
                x["DISPLAY_ID"] = x["uniqueid"]
            elif "QUERY_UID" in x.columns:
                x["DISPLAY_ID"] = x["QUERY_UID"]
            else:
                x["DISPLAY_ID"] = pd.NA

        x["SOURCE"] = source_label
        return x

    top = prep(df_top_ref, "top")
    bottom = prep(df_bottom_ref, "bottom")
    merged = pd.concat([top, bottom], axis=0, ignore_index=True, sort=False)

    # 统一去空格
    for col in ["DISPLAY_ID", "rsid", "SNP", "REF_MATCH", "uniqueid", "QUERY_UID"]:
        if col in merged.columns:
            merged[col] = merged[col].astype(str).str.strip()

    # 依次尝试匹配
    search_cols = ["DISPLAY_ID", "rsid", "SNP", "REF_MATCH", "uniqueid", "QUERY_UID"]

    for col in search_cols:
        if col in merged.columns:
            hit = merged[merged[col].astype(str) == user_text].copy()
            if len(hit) > 0:
                # 优先 top，再 bottom；也可以改成别的规则
                hit["_priority"] = hit["SOURCE"].map({"top": 0, "bottom": 1}).fillna(9)
                hit = hit.sort_values(["_priority"]).drop(columns=["_priority"])
                return hit.iloc[0]

    # 再做一次不区分大小写匹配（主要给 rsid / SNP 用）
    user_upper = user_text.upper()
    for col in search_cols:
        if col in merged.columns:
            hit = merged[merged[col].astype(str).str.upper() == user_upper].copy()
            if len(hit) > 0:
                hit["_priority"] = hit["SOURCE"].map({"top": 0, "bottom": 1}).fillna(9)
                hit = hit.sort_values(["_priority"]).drop(columns=["_priority"])
                return hit.iloc[0]

    available = [c for c in search_cols if c in merged.columns]
    raise ValueError(
        f"Could not resolve index SNP '{user_text}'. "
        f"Searched columns: {available}"
    )


def _read_plink_ld_table(ld_path, index_snp):
    if not os.path.exists(ld_path):
        return {}

    ld = pd.read_csv(ld_path, sep=r"\s+")
    if ld.empty or "SNP_A" not in ld.columns or "SNP_B" not in ld.columns or "R2" not in ld.columns:
        return {}

    ld = ld.copy()
    ld["SNP_A"] = ld["SNP_A"].astype(str)
    ld["SNP_B"] = ld["SNP_B"].astype(str)
    ld["R2"] = pd.to_numeric(ld["R2"], errors="coerce")

    sub = ld[ld["SNP_A"] == str(index_snp)][["SNP_B", "R2"]].dropna()
    out = dict(zip(sub["SNP_B"], sub["R2"]))
    out[str(index_snp)] = 1.0
    return out


@st.cache_data(show_spinner=False)
def compute_ld_maps_with_plink(
    bfile_prefix,
    chrom,
    start,
    end,
    window_snps,
    idx1_ref,
    idx2_ref,
    idx3_ref,
    plink_path=str(BIN_DIR / "plink")
):
    plink_exec = _find_plink_exec(plink_path)
    bfile_prefix = _resolve_bfile_prefix(bfile_prefix)
    chrom = normalize_chrom(chrom)

    bim = load_reference_bim(bfile_prefix)
    ref_window = bim[
        (bim["CHR"] == chrom) &
        (bim["BP"] >= int(start)) &
        (bim["BP"] <= int(end))
    ].copy()

    ref_snps = set(ref_window["SNP"].tolist())
    extract_snps = sorted(set([str(x) for x in window_snps if pd.notna(x) and str(x) in ref_snps]))

    index_status = {
        "variant 1": idx1_ref is not None and str(idx1_ref) in ref_snps,
        "variant 2": idx2_ref is not None and str(idx2_ref) in ref_snps,
        "variant 3": idx3_ref is not None and str(idx3_ref) in ref_snps,
    }

    ld_maps = {"r2_1": {}, "r2_2": {}, "r2_3": {}}

    if len(extract_snps) == 0:
        return ld_maps, index_status, tuple()

    kb_span = max(1000, int((int(end) - int(start)) / 1000) + 100)

    with tempfile.TemporaryDirectory(prefix="plink_ld_") as tmpdir:
        extract_path = os.path.join(tmpdir, "extract.snplist")
        with open(extract_path, "w") as f:
            for snp in extract_snps:
                f.write(f"{snp}\n")

        index_inputs = [
            ("r2_1", idx1_ref),
            ("r2_2", idx2_ref),
            ("r2_3", idx3_ref),
        ]

        for r2_col, index_snp in index_inputs:
            if index_snp is None or str(index_snp) not in ref_snps:
                continue

            out_prefix = os.path.join(tmpdir, f"ld_{r2_col}")
            cmd = [
                plink_exec,
                "--threads", "1",
                "--bfile", bfile_prefix,
                "--chr", chrom,
                "--from-bp", str(int(start)),
                "--to-bp", str(int(end)),
                "--extract", extract_path,
                "--ld-snp", str(index_snp),
                "--r2",
                "--ld-window", "999999",
                "--ld-window-kb", str(kb_span),
                "--ld-window-r2", "0",
                "--out", out_prefix
            ]
            if chrom == "X":
                cmd.extend(["--allow-extra-chr"])

            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            if res.returncode != 0:
                log(f"PLINK failed for {index_snp}:\n{res.stdout}")
                continue

            ld_maps[r2_col] = _read_plink_ld_table(out_prefix + ".ld", index_snp)

    return ld_maps, index_status, tuple(sorted(ref_snps))


def attach_uploaded_ld_keys(df, ld_universe):
    """
    Uploaded-LD equivalent of attach_reference_snp_two_pass: set REF_SNP
    on each row to whichever identifier column first appears in the
    uploaded LD universe. Does not require a 1000G BIM.

    Match priority: DISPLAY_ID, rsid, SNP, uniqueid, posID, QUERY_UID,
    QUERY_UID_FLIP. Rows with no identifier in ld_universe get
    REF_SNP=NaN and render as grey/missing downstream.
    """
    out = dedup_columns(df.copy())

    if all(c in out.columns for c in ["CHR", "BP", "A1", "A2"]):
        chr_s = out["CHR"].map(normalize_chrom)
        bp_s = pd.to_numeric(out["BP"], errors="coerce").astype("Int64").astype(str)
        a1_s = out["A1"].astype(str).str.strip().str.upper()
        a2_s = out["A2"].astype(str).str.strip().str.upper()
        uid_fwd = chr_s + ":" + bp_s + ":" + a1_s + ":" + a2_s
        uid_rev = chr_s + ":" + bp_s + ":" + a2_s + ":" + a1_s
        if "uniqueid" not in out.columns:
            out["uniqueid"] = uid_fwd
        if "QUERY_UID" not in out.columns:
            out["QUERY_UID"] = uid_fwd
        if "QUERY_UID_FLIP" not in out.columns:
            out["QUERY_UID_FLIP"] = uid_rev

    universe = {str(x) for x in (ld_universe or ())}

    candidate_cols = ["DISPLAY_ID", "rsid", "SNP", "uniqueid", "posID", "QUERY_UID", "QUERY_UID_FLIP"]
    present_cols = [c for c in candidate_cols if c in out.columns]

    ref_snp = pd.Series([pd.NA] * len(out), index=out.index, dtype=object)
    for col in present_cols:
        col_str = out[col].astype(str).str.strip()
        mask = ref_snp.isna() & col_str.isin(universe)
        if mask.any():
            ref_snp.loc[mask] = col_str.loc[mask]

    out["REF_SNP"] = ref_snp
    return out


def _safe_idx_str(x):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    return s


def compute_ld_maps_from_uploaded_long(ld_long, window_snps, idx1_ref, idx2_ref, idx3_ref):
    """
    Build ld_maps from a standardized long-format LD table. Output
    structure matches compute_ld_maps_with_plink so the downstream
    ld_annot / merge_ld_annot pipeline is unchanged.
    """
    df = ld_long[["SNP_A", "SNP_B", "R2"]].copy()
    df["SNP_A"] = df["SNP_A"].astype(str)
    df["SNP_B"] = df["SNP_B"].astype(str)

    self_pairs = df[(df["SNP_A"] == df["SNP_B"]) & np.isclose(df["R2"], 1.0)]
    ld_universe = set(self_pairs["SNP_A"].astype(str))

    window_set = {str(x) for x in (window_snps or ()) if pd.notna(x)}
    ref_snps = tuple(sorted(ld_universe & window_set)) if window_set else tuple(sorted(ld_universe))

    def in_universe(idx):
        s = _safe_idx_str(idx)
        return s is not None and s in ld_universe

    index_status = {
        "variant 1": in_universe(idx1_ref),
        "variant 2": in_universe(idx2_ref),
        "variant 3": in_universe(idx3_ref),
    }

    def build_map(idx_ref):
        s = _safe_idx_str(idx_ref)
        if s is None or s not in ld_universe:
            return {}
        a = df[df["SNP_A"] == s][["SNP_B", "R2"]].rename(columns={"SNP_B": "OTHER"})
        b = df[df["SNP_B"] == s][["SNP_A", "R2"]].rename(columns={"SNP_A": "OTHER"})
        m = pd.concat([a, b], axis=0, ignore_index=True).dropna(subset=["OTHER", "R2"])
        m["OTHER"] = m["OTHER"].astype(str)
        m = m.groupby("OTHER", as_index=False)["R2"].max()
        out = dict(zip(m["OTHER"], m["R2"].astype(float)))
        out[s] = 1.0
        return out

    ld_maps = {
        "r2_1": build_map(idx1_ref),
        "r2_2": build_map(idx2_ref),
        "r2_3": build_map(idx3_ref),
    }
    return ld_maps, index_status, ref_snps


def compute_ld_maps_from_uploaded_matrix(ld_matrix, window_snps, idx1_ref, idx2_ref, idx3_ref):
    """
    Build ld_maps from a square-ish LD matrix (rows/cols = SNP IDs,
    cells = R^2). Output structure matches compute_ld_maps_with_plink.
    """
    rows = {str(x) for x in ld_matrix.index.tolist()}
    cols = {str(x) for x in ld_matrix.columns.tolist()}
    ld_universe = rows | cols

    window_set = {str(x) for x in (window_snps or ()) if pd.notna(x)}
    ref_snps = tuple(sorted(ld_universe & window_set)) if window_set else tuple(sorted(ld_universe))

    def in_universe(idx):
        s = _safe_idx_str(idx)
        return s is not None and s in ld_universe

    index_status = {
        "variant 1": in_universe(idx1_ref),
        "variant 2": in_universe(idx2_ref),
        "variant 3": in_universe(idx3_ref),
    }

    def extract_vec(idx_ref):
        s = _safe_idx_str(idx_ref)
        if s is None:
            return {}
        if s in rows:
            vec = ld_matrix.loc[s]
        elif s in cols:
            vec = ld_matrix[s]
        else:
            return {}
        if isinstance(vec, pd.DataFrame):
            vec = vec.iloc[0]
        vec = pd.to_numeric(vec, errors="coerce").dropna()
        vec.index = vec.index.astype(str)
        out = {k: float(v) for k, v in vec.items()}
        out[s] = 1.0
        return out

    ld_maps = {
        "r2_1": extract_vec(idx1_ref),
        "r2_2": extract_vec(idx2_ref),
        "r2_3": extract_vec(idx3_ref),
    }
    return ld_maps, index_status, ref_snps


def build_ld_annot_for_window(window_union_df, ld_maps, ref_snps, idx1_ref, idx2_ref, idx3_ref):
    out = window_union_df[["REF_SNP", "CHR", "BP"]].drop_duplicates("REF_SNP").copy()
    ref_set = set(ref_snps)

    out["in_ref"] = out["REF_SNP"].notna() & out["REF_SNP"].isin(ref_set)
    out["r2_1"] = out["REF_SNP"].map(ld_maps.get("r2_1", {}))
    out["r2_2"] = out["REF_SNP"].map(ld_maps.get("r2_2", {}))
    out["r2_3"] = out["REF_SNP"].map(ld_maps.get("r2_3", {}))

    if idx1_ref in ref_set:
        out.loc[out["REF_SNP"] == idx1_ref, "r2_1"] = 1.0
    if idx2_ref in ref_set:
        out.loc[out["REF_SNP"] == idx2_ref, "r2_2"] = 1.0
    if idx3_ref in ref_set:
        out.loc[out["REF_SNP"] == idx3_ref, "r2_3"] = 1.0

    return out[["REF_SNP", "r2_1", "r2_2", "r2_3", "in_ref"]]


def merge_ld_annot(df, ld_annot):
    df = dedup_columns(df)
    ld_annot = dedup_columns(ld_annot)

    base = df.drop(columns=["r2_1", "r2_2", "r2_3"], errors="ignore").copy()
    out = base.merge(ld_annot, on="REF_SNP", how="left")
    out["in_ref"] = out["in_ref"].fillna(False).astype(bool)
    return dedup_columns(out)


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
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
    bw_path=str(DATA_DIR / "recomb1000GAvg.bw"),
    n_recomb_bins=800,
    locusblend_mode="Three-index LocusBlend",
):
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

    if show_recomb and os.path.exists(bw_path):
        try:
            rec = _read_recomb_track_bw(chrom, bp_start, bp_end, bw_path=bw_path, n_bins=n_recomb_bins)
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


def update_progress(progress_bar, status_box, pct, msg):
    if progress_bar is not None:
        progress_bar.progress(pct)
    if status_box is not None:
        status_box.markdown(f"**{msg}**")
    log(msg)


def image_to_data_uri(path):
    mime = guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def set_streamlit_chrome_minimal():
    """Best-effort app chrome minimization.

    This reduces access to the Streamlit top-right menu in supported
    Streamlit versions. It must never trigger reruns and must never break
    the app if the option is unavailable.
    """
    try:
        st.set_option("client.toolbarMode", "minimal")
    except Exception:
        pass


def inject_locusblend_css():
    st.markdown(
        """
        <style>
        :root {
            color-scheme: light;
            --lb-bg: #ffffff;
            --lb-surface: #ffffff;
            --lb-sidebar-bg: #f8fafc;
            --lb-text: #111827;
            --lb-muted: #4b5563;
            --lb-border: #e5e7eb;
            --lb-input-bg: #ffffff;
            --lb-input-border: #d1d5db;
        }

        html, body, .stApp {
            background: var(--lb-bg) !important;
            color: var(--lb-text) !important;
            color-scheme: light !important;
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"] {
            background: var(--lb-bg) !important;
            color: var(--lb-text) !important;
        }

        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"] {
            background: var(--lb-sidebar-bg) !important;
            color: var(--lb-text) !important;
        }

        .block-container {
            background: var(--lb-bg) !important;
            color: var(--lb-text) !important;
        }

        h1, h2, h3, h4, h5, h6,
        p, li, label,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stCaptionContainer"],
        [data-testid="stWidgetLabel"],
        [data-testid="stExpander"],
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary {
            color: var(--lb-text) !important;
        }

        [data-testid="stCaptionContainer"],
        .stCaptionContainer {
            color: var(--lb-muted) !important;
        }

        input,
        textarea,
        select,
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea,
        [data-baseweb="select"] div,
        [data-baseweb="select"] span {
            background-color: var(--lb-input-bg) !important;
            color: var(--lb-text) !important;
        }

        [data-baseweb="input"],
        [data-baseweb="textarea"],
        [data-baseweb="select"] {
            background-color: var(--lb-input-bg) !important;
            color: var(--lb-text) !important;
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"],
        [data-testid="stMetric"],
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            color: var(--lb-text) !important;
        }

        [data-testid="stFileUploader"] {
            color: var(--lb-text) !important;
        }

        [data-testid="stFileUploader"] section {
            background: var(--lb-surface) !important;
            color: var(--lb-text) !important;
            border-color: var(--lb-border) !important;
        }

        #MainMenu {
            visibility: hidden !important;
        }

        [data-testid="stToolbar"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }

        [data-testid="stDecoration"] {
            display: none !important;
            visibility: hidden !important;
        }

        [data-testid="stDeployButton"] {
            display: none !important;
            visibility: hidden !important;
        }

        /* Force Streamlit dialogs and modal content to light mode. */
        [data-testid="stDialog"],
        [data-testid="stDialog"] *,
        div[role="dialog"],
        div[role="dialog"] * {
            background-color: #ffffff !important;
            color: #111827 !important;
            color-scheme: light !important;
        }

        [data-testid="stDialog"] button,
        div[role="dialog"] button {
            background-color: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #d1d5db !important;
        }

        /* Force expanders to remain readable in browser/Streamlit dark mode. */
        [data-testid="stExpander"],
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary *,
        [data-testid="stExpander"] div,
        [data-testid="stExpander"] p,
        [data-testid="stExpander"] label,
        [data-testid="stExpander"] span {
            background-color: #ffffff !important;
            color: #111827 !important;
            color-scheme: light !important;
        }

        /* Streamlit/BaseWeb controls, menus and dropdown popovers. */
        [data-baseweb="popover"],
        [data-baseweb="popover"] *,
        [data-baseweb="menu"],
        [data-baseweb="menu"] *,
        [data-baseweb="select"],
        [data-baseweb="select"] *,
        [data-baseweb="input"],
        [data-baseweb="input"] *,
        [data-baseweb="textarea"],
        [data-baseweb="textarea"] * {
            background-color: #ffffff !important;
            color: #111827 !important;
            color-scheme: light !important;
        }

        /* Inputs and buttons should remain visible, not hidden. */
        input,
        textarea,
        select,
        button {
            color-scheme: light !important;
        }

        button {
            color: #111827 !important;
        }

        /* Keep Streamlit alerts readable. */
        [data-testid="stAlert"],
        [data-testid="stAlert"] *,
        .stAlert,
        .stAlert * {
            color: #111827 !important;
        }

        /* Keep export/download controls readable. */
        [data-testid="stDownloadButton"],
        [data-testid="stDownloadButton"] *,
        [data-testid="stButton"],
        [data-testid="stButton"] * {
            color: #111827 !important;
        }

        /* --- LocusBlend 2.3.2: force buttons and number steppers to light mode --- */

        /* Normal Streamlit buttons */
        [data-testid="stButton"] button,
        [data-testid="stDownloadButton"] button,
        [data-testid="stFormSubmitButton"] button {
            background-color: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #d1d5db !important;
            box-shadow: none !important;
            color-scheme: light !important;
        }

        /* Streamlit primary buttons, including Update plot / Prepare export when primary */
        button[data-testid="baseButton-primary"],
        [data-testid="baseButton-primary"],
        [data-testid="stButton"] button[data-testid="baseButton-primary"],
        [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] {
            background-color: #ff4b4b !important;
            color: #ffffff !important;
            border: 1px solid #ff4b4b !important;
            box-shadow: none !important;
            color-scheme: light !important;
        }

        /* Hover states */
        [data-testid="stButton"] button:hover,
        [data-testid="stDownloadButton"] button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            background-color: #f9fafb !important;
            color: #111827 !important;
            border-color: #9ca3af !important;
        }

        button[data-testid="baseButton-primary"]:hover,
        [data-testid="baseButton-primary"]:hover,
        [data-testid="stButton"] button[data-testid="baseButton-primary"]:hover,
        [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"]:hover {
            background-color: #ff3333 !important;
            color: #ffffff !important;
            border-color: #ff3333 !important;
        }

        /* Disabled buttons should be readable, not black */
        [data-testid="stButton"] button:disabled,
        [data-testid="stDownloadButton"] button:disabled,
        [data-testid="stFormSubmitButton"] button:disabled,
        button:disabled,
        button[disabled] {
            background-color: #f3f4f6 !important;
            color: #6b7280 !important;
            border: 1px solid #d1d5db !important;
            opacity: 1 !important;
            box-shadow: none !important;
            color-scheme: light !important;
        }

        /* BaseWeb number input stepper buttons, including +/- controls */
        [data-baseweb="input"] button,
        [data-baseweb="input"] [role="button"],
        [data-testid="stNumberInput"] button,
        [data-testid="stNumberInput"] [role="button"] {
            background-color: #ffffff !important;
            color: #111827 !important;
            border-color: #d1d5db !important;
            box-shadow: none !important;
            color-scheme: light !important;
        }

        /* Icons inside number steppers */
        [data-baseweb="input"] button svg,
        [data-baseweb="input"] [role="button"] svg,
        [data-testid="stNumberInput"] button svg,
        [data-testid="stNumberInput"] [role="button"] svg {
            color: #111827 !important;
            fill: #111827 !important;
            stroke: #111827 !important;
        }

        /* Disabled number stepper buttons */
        [data-baseweb="input"] button:disabled,
        [data-baseweb="input"] [role="button"][aria-disabled="true"],
        [data-testid="stNumberInput"] button:disabled,
        [data-testid="stNumberInput"] [role="button"][aria-disabled="true"] {
            background-color: #f3f4f6 !important;
            color: #9ca3af !important;
            border-color: #d1d5db !important;
            opacity: 1 !important;
        }

        /* Icons inside disabled number steppers */
        [data-baseweb="input"] button:disabled svg,
        [data-baseweb="input"] [role="button"][aria-disabled="true"] svg,
        [data-testid="stNumberInput"] button:disabled svg,
        [data-testid="stNumberInput"] [role="button"][aria-disabled="true"] svg {
            color: #9ca3af !important;
            fill: #9ca3af !important;
            stroke: #9ca3af !important;
        }

        /* Keep selectbox controls light and readable */
        [data-baseweb="select"] > div,
        [data-baseweb="select"] div[role="button"],
        [data-baseweb="select"] svg {
            background-color: #ffffff !important;
            color: #111827 !important;
            fill: #111827 !important;
            stroke: #111827 !important;
            color-scheme: light !important;
        }

        /* Checkbox should remain visible on light background */
        [data-testid="stCheckbox"] label,
        [data-testid="stCheckbox"] span,
        [data-testid="stCheckbox"] div {
            color: #111827 !important;
            color-scheme: light !important;
        }

        .lb-sidebar-visual-card {
            container-type: inline-size;
            background: #ffffff !important;
            border: 1px solid var(--lb-border);
            border-radius: 12px;
            padding: 8px;
            margin: 0.25rem 0 0.75rem 0;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
            overflow: hidden;
            width: 100%;
            box-sizing: border-box;
        }

        .lb-sidebar-visual-row {
            display: flex;
            flex-direction: row;
            flex-wrap: nowrap;
            align-items: center;
            justify-content: center;
            gap: 8px;
            width: 100%;
            min-width: 0;
            box-sizing: border-box;
        }

        .lb-sidebar-visual-pane {
            min-width: 0;
            max-width: 100%;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            box-sizing: border-box;
        }

        .lb-sidebar-visual-pane--legend {
            flex: 1.08 1 0;
        }

        .lb-sidebar-visual-pane--diagram {
            flex: 0.92 1 0;
        }

        .lb-sidebar-visual-pane--single {
            flex: 1 1 auto;
        }

        .lb-sidebar-visual-img {
            display: block;
            width: 100%;
            max-width: 100%;
            min-width: 0;
            height: auto;
            max-height: 125px;
            object-fit: contain;
            box-sizing: border-box;
        }

        @container (max-width: 280px) {
            .lb-sidebar-visual-card {
                padding: 6px;
            }

            .lb-sidebar-visual-row {
                gap: 4px;
            }

            .lb-sidebar-visual-img {
                max-height: 115px;
            }
        }

        @container (max-width: 220px) {
            .lb-sidebar-visual-card {
                padding: 4px;
            }

            .lb-sidebar-visual-row {
                gap: 3px;
            }

            .lb-sidebar-visual-img {
                max-height: 100px;
            }
        }

        /* --- LocusBlend 2.3.3: exact Streamlit 1.50+ stBaseButton override --- */

        /* Exact Streamlit button selectors observed in Chrome DevTools.
           Do not target st-emotion-cache-* classes because they are generated. */

        button[data-testid="stBaseButton-secondary"],
        button[data-testid="stBaseButton-tertiary"],
        button[kind="secondary"],
        button[kind="tertiary"] {
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #d1d5db !important;
            box-shadow: none !important;
            opacity: 1 !important;
            color-scheme: light !important;
        }

        /* Text, spans, and icons inside secondary/tertiary buttons */
        button[data-testid="stBaseButton-secondary"] *,
        button[data-testid="stBaseButton-tertiary"] *,
        button[kind="secondary"] *,
        button[kind="tertiary"] * {
            color: #111827 !important;
            fill: #111827 !important;
            stroke: #111827 !important;
        }

        /* Primary Streamlit buttons */
        button[data-testid="stBaseButton-primary"],
        button[kind="primary"] {
            background: #ff4b4b !important;
            background-color: #ff4b4b !important;
            color: #ffffff !important;
            border: 1px solid #ff4b4b !important;
            box-shadow: none !important;
            opacity: 1 !important;
            color-scheme: light !important;
        }

        /* Text, spans, and icons inside primary buttons */
        button[data-testid="stBaseButton-primary"] *,
        button[kind="primary"] * {
            color: #ffffff !important;
            fill: #ffffff !important;
            stroke: #ffffff !important;
        }

        /* Generic stBaseButton fallback */
        button[data-testid^="stBaseButton"] {
            color-scheme: light !important;
            box-shadow: none !important;
        }

        /* Disabled buttons: make them light gray, not black */
        button[data-testid^="stBaseButton"]:disabled,
        button[data-testid^="stBaseButton"][disabled],
        button[data-testid^="stBaseButton"][aria-disabled="true"],
        button[kind]:disabled,
        button[kind][disabled],
        button[kind][aria-disabled="true"] {
            background: #f3f4f6 !important;
            background-color: #f3f4f6 !important;
            color: #6b7280 !important;
            border: 1px solid #d1d5db !important;
            opacity: 1 !important;
            box-shadow: none !important;
            cursor: not-allowed !important;
            color-scheme: light !important;
        }

        /* Disabled button inner text/icons */
        button[data-testid^="stBaseButton"]:disabled *,
        button[data-testid^="stBaseButton"][disabled] *,
        button[data-testid^="stBaseButton"][aria-disabled="true"] *,
        button[kind]:disabled *,
        button[kind][disabled] *,
        button[kind][aria-disabled="true"] * {
            color: #6b7280 !important;
            fill: #6b7280 !important;
            stroke: #6b7280 !important;
        }

        /* Hover states for enabled secondary/tertiary buttons */
        button[data-testid="stBaseButton-secondary"]:not(:disabled):hover,
        button[data-testid="stBaseButton-tertiary"]:not(:disabled):hover,
        button[kind="secondary"]:not(:disabled):hover,
        button[kind="tertiary"]:not(:disabled):hover {
            background: #f9fafb !important;
            background-color: #f9fafb !important;
            color: #111827 !important;
            border-color: #9ca3af !important;
        }

        /* Hover states for enabled primary buttons */
        button[data-testid="stBaseButton-primary"]:not(:disabled):hover,
        button[kind="primary"]:not(:disabled):hover {
            background: #ff3333 !important;
            background-color: #ff3333 !important;
            color: #ffffff !important;
            border-color: #ff3333 !important;
        }

        /* File uploader buttons and inner labels */
        [data-testid="stFileUploader"] button[data-testid^="stBaseButton"],
        [data-testid="stFileUploader"] button[kind],
        [data-testid="stFileUploader"] button {
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #d1d5db !important;
            box-shadow: none !important;
            opacity: 1 !important;
            color-scheme: light !important;
        }

        [data-testid="stFileUploader"] button[data-testid^="stBaseButton"] *,
        [data-testid="stFileUploader"] button[kind] *,
        [data-testid="stFileUploader"] button * {
            color: #111827 !important;
            fill: #111827 !important;
            stroke: #111827 !important;
        }

        /* File uploader dropzone and text */
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploadDropzone"] {
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #111827 !important;
            border-color: #e5e7eb !important;
            color-scheme: light !important;
        }

        [data-testid="stFileUploader"] section *,
        [data-testid="stFileUploaderDropzone"] *,
        [data-testid="stFileUploadDropzone"] * {
            color: #111827 !important;
        }

        /* Number input text field */
        [data-testid="stNumberInput"] input,
        [data-baseweb="input"] input {
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #111827 !important;
            border-color: #d1d5db !important;
            color-scheme: light !important;
        }

        /* Number input stepper controls.
           Streamlit/BaseWeb can use buttons, role=button, or aria-label wrappers. */
        [data-testid="stNumberInput"] button,
        [data-testid="stNumberInput"] button[data-testid^="stBaseButton"],
        [data-testid="stNumberInput"] button[kind],
        [data-testid="stNumberInput"] [role="button"],
        [data-testid="stNumberInput"] div[aria-label],
        [data-testid="stNumberInput"] span[aria-label],
        [data-baseweb="input"] button,
        [data-baseweb="input"] button[data-testid^="stBaseButton"],
        [data-baseweb="input"] button[kind],
        [data-baseweb="input"] [role="button"],
        [data-baseweb="input"] div[aria-label],
        [data-baseweb="input"] span[aria-label] {
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #111827 !important;
            border-color: #d1d5db !important;
            box-shadow: none !important;
            opacity: 1 !important;
            color-scheme: light !important;
        }

        /* Number input stepper icons */
        [data-testid="stNumberInput"] button *,
        [data-testid="stNumberInput"] button[data-testid^="stBaseButton"] *,
        [data-testid="stNumberInput"] button[kind] *,
        [data-testid="stNumberInput"] [role="button"] *,
        [data-testid="stNumberInput"] div[aria-label] *,
        [data-testid="stNumberInput"] span[aria-label] *,
        [data-baseweb="input"] button *,
        [data-baseweb="input"] button[data-testid^="stBaseButton"] *,
        [data-baseweb="input"] button[kind] *,
        [data-baseweb="input"] [role="button"] *,
        [data-baseweb="input"] div[aria-label] *,
        [data-baseweb="input"] span[aria-label] * {
            color: #111827 !important;
            fill: #111827 !important;
            stroke: #111827 !important;
        }

        /* Disabled number input steppers */
        [data-testid="stNumberInput"] button:disabled,
        [data-testid="stNumberInput"] button[disabled],
        [data-testid="stNumberInput"] button[aria-disabled="true"],
        [data-testid="stNumberInput"] [role="button"][aria-disabled="true"],
        [data-baseweb="input"] button:disabled,
        [data-baseweb="input"] button[disabled],
        [data-baseweb="input"] button[aria-disabled="true"],
        [data-baseweb="input"] [role="button"][aria-disabled="true"] {
            background: #f3f4f6 !important;
            background-color: #f3f4f6 !important;
            color: #9ca3af !important;
            border-color: #d1d5db !important;
            opacity: 1 !important;
        }

        /* Disabled number input stepper icons */
        [data-testid="stNumberInput"] button:disabled *,
        [data-testid="stNumberInput"] button[disabled] *,
        [data-testid="stNumberInput"] button[aria-disabled="true"] *,
        [data-testid="stNumberInput"] [role="button"][aria-disabled="true"] *,
        [data-baseweb="input"] button:disabled *,
        [data-baseweb="input"] button[disabled] *,
        [data-baseweb="input"] button[aria-disabled="true"] *,
        [data-baseweb="input"] [role="button"][aria-disabled="true"] * {
            color: #9ca3af !important;
            fill: #9ca3af !important;
            stroke: #9ca3af !important;
        }

        /* Download button */
        [data-testid="stDownloadButton"] button[data-testid^="stBaseButton"],
        [data-testid="stDownloadButton"] button[kind],
        [data-testid="stDownloadButton"] button {
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #d1d5db !important;
            box-shadow: none !important;
            opacity: 1 !important;
            color-scheme: light !important;
        }

        [data-testid="stDownloadButton"] button[data-testid^="stBaseButton"] *,
        [data-testid="stDownloadButton"] button[kind] *,
        [data-testid="stDownloadButton"] button * {
            color: #111827 !important;
            fill: #111827 !important;
            stroke: #111827 !important;
        }

        /* Radio buttons and checkbox text should remain readable */
        [data-testid="stRadio"] *,
        [data-testid="stCheckbox"] * {
            color-scheme: light !important;
        }

        [data-testid="stRadio"] label,
        [data-testid="stRadio"] span,
        [data-testid="stCheckbox"] label,
        [data-testid="stCheckbox"] span {
            color: #111827 !important;
        }

        /* --- LocusBlend 2.3.4: export button and checkbox repair --- */

        /* Repair normal/secondary Streamlit buttons, including Prepare export file. */
        button[data-testid="stBaseButton-secondary"],
        button[kind="secondary"],
        [data-testid="stButton"] button[data-testid="stBaseButton-secondary"],
        [data-testid="stButton"] button[kind="secondary"],
        [data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-secondary"],
        [data-testid="stFormSubmitButton"] button[kind="secondary"],
        [data-testid="stDownloadButton"] button[data-testid="stBaseButton-secondary"],
        [data-testid="stDownloadButton"] button[kind="secondary"] {
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #d1d5db !important;
            box-shadow: none !important;
            opacity: 1 !important;
            color-scheme: light !important;
        }

        /* Repair text inside secondary buttons.
           Do not force SVG fill/stroke here; broad SVG rules can break checkbox marks. */
        button[data-testid="stBaseButton-secondary"] span,
        button[kind="secondary"] span,
        [data-testid="stButton"] button[data-testid="stBaseButton-secondary"] span,
        [data-testid="stButton"] button[kind="secondary"] span,
        [data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-secondary"] span,
        [data-testid="stFormSubmitButton"] button[kind="secondary"] span,
        [data-testid="stDownloadButton"] button[data-testid="stBaseButton-secondary"] span,
        [data-testid="stDownloadButton"] button[kind="secondary"] span {
            color: #111827 !important;
        }

        /* Keep primary buttons readable if any remain. */
        button[data-testid="stBaseButton-primary"],
        button[kind="primary"],
        [data-testid="stButton"] button[data-testid="stBaseButton-primary"],
        [data-testid="stButton"] button[kind="primary"],
        [data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-primary"],
        [data-testid="stFormSubmitButton"] button[kind="primary"] {
            background: #ff4b4b !important;
            background-color: #ff4b4b !important;
            color: #ffffff !important;
            border: 1px solid #ff4b4b !important;
            box-shadow: none !important;
            opacity: 1 !important;
            color-scheme: light !important;
        }

        button[data-testid="stBaseButton-primary"] span,
        button[kind="primary"] span,
        [data-testid="stButton"] button[data-testid="stBaseButton-primary"] span,
        [data-testid="stButton"] button[kind="primary"] span,
        [data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-primary"] span,
        [data-testid="stFormSubmitButton"] button[kind="primary"] span {
            color: #ffffff !important;
        }

        /* Disabled buttons should be light gray and readable. */
        button[data-testid^="stBaseButton"]:disabled,
        button[data-testid^="stBaseButton"][disabled],
        button[data-testid^="stBaseButton"][aria-disabled="true"],
        button[kind]:disabled,
        button[kind][disabled],
        button[kind][aria-disabled="true"] {
            background: #f3f4f6 !important;
            background-color: #f3f4f6 !important;
            color: #6b7280 !important;
            border: 1px solid #d1d5db !important;
            opacity: 1 !important;
            box-shadow: none !important;
            color-scheme: light !important;
        }

        button[data-testid^="stBaseButton"]:disabled span,
        button[data-testid^="stBaseButton"][disabled] span,
        button[data-testid^="stBaseButton"][aria-disabled="true"] span,
        button[kind]:disabled span,
        button[kind][disabled] span,
        button[kind][aria-disabled="true"] span {
            color: #6b7280 !important;
        }

        /* Checkbox repair:
           Only control the label text. Do NOT override checkbox internal spans/SVGs/marks,
           because Streamlit/BaseWeb uses those to draw the checkmark. */
        [data-testid="stCheckbox"] label,
        [data-testid="stCheckbox"] label p,
        [data-testid="stCheckbox"] label span:not([data-baseweb]) {
            color: #111827 !important;
            color-scheme: light !important;
        }

        /* Restore checkbox box visibility without forcing the inner checkmark SVG. */
        [data-testid="stCheckbox"] [data-baseweb="checkbox"] {
            color-scheme: light !important;
        }

        /* Checkbox input should remain selectable and visible. */
        [data-testid="stCheckbox"] input[type="checkbox"] {
            accent-color: #ff4b4b !important;
        }

        /* Remove overly broad checkbox icon overrides from winning.
           This intentionally avoids fill/stroke rules on checkbox descendants. */
        [data-testid="stCheckbox"] svg {
            color: revert !important;
            fill: revert !important;
            stroke: revert !important;
        }

        /* File uploader Browse files button remains secondary-style. */
        [data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"],
        [data-testid="stFileUploader"] button[kind="secondary"] {
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #d1d5db !important;
            box-shadow: none !important;
            opacity: 1 !important;
        }

        [data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"] span,
        [data-testid="stFileUploader"] button[kind="secondary"] span {
            color: #111827 !important;
        }

        /* --- LocusBlend 2.4.1: markdown code and documentation readability --- */

        /* Inline markdown code: prevent black background / green text in browser dark mode. */
        [data-testid="stMarkdownContainer"] code,
        [data-testid="stMarkdownContainer"] p code,
        [data-testid="stMarkdownContainer"] li code,
        [data-testid="stMarkdownContainer"] td code,
        [data-testid="stMarkdownContainer"] th code,
        [data-testid="stExpander"] code,
        [data-testid="stExpander"] p code,
        [data-testid="stExpander"] li code,
        code {
            background: #f3f4f6 !important;
            background-color: #f3f4f6 !important;
            color: #111827 !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 4px !important;
            padding: 0.08rem 0.28rem !important;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace !important;
            font-size: 0.88em !important;
            white-space: break-spaces !important;
            color-scheme: light !important;
        }

        /* Code blocks should also stay light. */
        [data-testid="stMarkdownContainer"] pre,
        [data-testid="stMarkdownContainer"] pre code,
        [data-testid="stExpander"] pre,
        [data-testid="stExpander"] pre code,
        pre,
        pre code {
            background: #f8fafc !important;
            background-color: #f8fafc !important;
            color: #111827 !important;
            border-color: #e5e7eb !important;
            color-scheme: light !important;
        }

        /* Markdown tables in the User Guide should remain readable. */
        [data-testid="stMarkdownContainer"] table,
        [data-testid="stMarkdownContainer"] thead,
        [data-testid="stMarkdownContainer"] tbody,
        [data-testid="stMarkdownContainer"] tr,
        [data-testid="stMarkdownContainer"] th,
        [data-testid="stMarkdownContainer"] td {
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #111827 !important;
            border-color: #e5e7eb !important;
            color-scheme: light !important;
        }

        /* --- LocusBlend 2.4.1: compact sidebar sections --- */

        /* Reduce sidebar inner padding and vertical gaps. */
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-top: 1.0rem !important;
            padding-bottom: 1.0rem !important;
        }

        /* Keep sidebar markdown/caption spacing tighter. */
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            margin-bottom: 0.25rem !important;
        }

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] .stCaptionContainer {
            margin-top: 0.15rem !important;
            margin-bottom: 0.45rem !important;
            line-height: 1.35 !important;
        }

        /* Compact sidebar expanders while keeping them easy to select. */
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            margin-top: 0.25rem !important;
            margin-bottom: 0.45rem !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] details {
            border-radius: 8px !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            min-height: 2.35rem !important;
            padding-top: 0.45rem !important;
            padding-bottom: 0.45rem !important;
        }

        /* Reduce oversized blank gaps between Streamlit element containers in sidebar. */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.35rem !important;
        }

        [data-testid="stSidebar"] [data-testid="stElementContainer"] {
            margin-bottom: 0.20rem !important;
        }

        /* Make section helper text compact. */
        [data-testid="stSidebar"] p {
            line-height: 1.35 !important;
        }

        /* Keep form submit button close to final section. */
        [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] {
            margin-top: 0.35rem !important;
            margin-bottom: 0.35rem !important;
        }

        /* --- LocusBlend 2.5.1: checkbox and uploader control visibility --- */

        /* Make Streamlit/BaseWeb checkbox boxes and ticks visible in forced light mode. */
        [data-testid="stCheckbox"] [data-baseweb="checkbox"] > div,
        [data-testid="stCheckbox"] [data-baseweb="checkbox"] div[role="checkbox"],
        [data-testid="stCheckbox"] div[role="checkbox"] {
            background-color: #ffffff !important;
            border-color: #ff4b4b !important;
            color-scheme: light !important;
        }

        /* Checked checkbox state. */
        [data-testid="stCheckbox"] input[type="checkbox"]:checked + div,
        [data-testid="stCheckbox"] [aria-checked="true"],
        [data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"] {
            background-color: #ff4b4b !important;
            border-color: #ff4b4b !important;
        }

        /* Checkbox tick/check icon. Avoid broad rules on all checkbox descendants. */
        [data-testid="stCheckbox"] [aria-checked="true"] svg,
        [data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"] svg {
            color: #ffffff !important;
            fill: #ffffff !important;
            stroke: #ffffff !important;
        }

        /* Checkbox label remains readable. */
        [data-testid="stCheckbox"] label,
        [data-testid="stCheckbox"] label p,
        [data-testid="stCheckbox"] label span {
            color: #111827 !important;
        }

        /* Browser-native checkbox fallback. */
        [data-testid="stCheckbox"] input[type="checkbox"] {
            accent-color: #ff4b4b !important;
        }

        /* Uploaded file row and remove controls should not appear black in browser dark mode. */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] *,
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"],
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] * {
            background-color: #ffffff !important;
            color: #111827 !important;
            color-scheme: light !important;
        }

        /* Uploaded-file remove button / small icon buttons. */
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] button[data-testid^="stBaseButton"],
        [data-testid="stFileUploader"] button[kind],
        [data-testid="stFileUploader"] [role="button"] {
            background-color: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #d1d5db !important;
            box-shadow: none !important;
            opacity: 1 !important;
            color-scheme: light !important;
        }

        [data-testid="stFileUploader"] button svg,
        [data-testid="stFileUploader"] button *,
        [data-testid="stFileUploader"] [role="button"] svg,
        [data-testid="stFileUploader"] [role="button"] * {
            color: #111827 !important;
            fill: #111827 !important;
            stroke: #111827 !important;
        }

        /* If Streamlit renders a small dark delete/remove pill, force it light. */
        [data-testid="stFileUploader"] [aria-label*="remove" i],
        [data-testid="stFileUploader"] [aria-label*="delete" i],
        [data-testid="stFileUploader"] [title*="remove" i],
        [data-testid="stFileUploader"] [title*="delete" i] {
            background-color: #ffffff !important;
            color: #111827 !important;
            border-color: #d1d5db !important;
            color-scheme: light !important;
        }

        [data-testid="stFileUploader"] [aria-label*="remove" i] *,
        [data-testid="stFileUploader"] [aria-label*="delete" i] *,
        [data-testid="stFileUploader"] [title*="remove" i] *,
        [data-testid="stFileUploader"] [title*="delete" i] * {
            color: #111827 !important;
            fill: #111827 !important;
            stroke: #111827 !important;
        }

        /* --- LocusBlend 2.5.2: final uploaded-file remove-control fix --- */

        /* Uploaded file row containers. */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] > div,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] div,
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"],
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] > div,
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] div {
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #111827 !important;
            color-scheme: light !important;
        }

        /* Uploaded file row text and file-size text. */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] p,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] small,
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] span,
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] p,
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] small {
            color: #111827 !important;
        }

        /* Uploaded-file row buttons, including remove/delete controls. */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button[data-testid^="stBaseButton"],
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button[kind],
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] [role="button"],
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] button,
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] button[data-testid^="stBaseButton"],
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] button[kind],
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] [role="button"] {
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #d1d5db !important;
            border-radius: 8px !important;
            box-shadow: none !important;
            opacity: 1 !important;
            color-scheme: light !important;
        }

        /* Uploaded-file row button icons. */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button svg,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button *,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] [role="button"] svg,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] [role="button"] *,
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] button svg,
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] button *,
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] [role="button"] svg,
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] [role="button"] * {
            color: #111827 !important;
            fill: #111827 !important;
            stroke: #111827 !important;
        }

        /* Directly target common uploader remove/delete button labels. */
        [data-testid="stFileUploader"] button[aria-label*="remove" i],
        [data-testid="stFileUploader"] button[aria-label*="delete" i],
        [data-testid="stFileUploader"] button[aria-label*="clear" i],
        [data-testid="stFileUploader"] button[title*="remove" i],
        [data-testid="stFileUploader"] button[title*="delete" i],
        [data-testid="stFileUploader"] button[title*="clear" i],
        [data-testid="stFileUploader"] [role="button"][aria-label*="remove" i],
        [data-testid="stFileUploader"] [role="button"][aria-label*="delete" i],
        [data-testid="stFileUploader"] [role="button"][aria-label*="clear" i],
        [data-testid="stFileUploader"] [role="button"][title*="remove" i],
        [data-testid="stFileUploader"] [role="button"][title*="delete" i],
        [data-testid="stFileUploader"] [role="button"][title*="clear" i] {
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #d1d5db !important;
            border-radius: 8px !important;
            box-shadow: none !important;
            opacity: 1 !important;
            color-scheme: light !important;
        }

        /* Icons inside direct remove/delete targets. */
        [data-testid="stFileUploader"] button[aria-label*="remove" i] *,
        [data-testid="stFileUploader"] button[aria-label*="delete" i] *,
        [data-testid="stFileUploader"] button[aria-label*="clear" i] *,
        [data-testid="stFileUploader"] button[title*="remove" i] *,
        [data-testid="stFileUploader"] button[title*="delete" i] *,
        [data-testid="stFileUploader"] button[title*="clear" i] *,
        [data-testid="stFileUploader"] [role="button"][aria-label*="remove" i] *,
        [data-testid="stFileUploader"] [role="button"][aria-label*="delete" i] *,
        [data-testid="stFileUploader"] [role="button"][aria-label*="clear" i] *,
        [data-testid="stFileUploader"] [role="button"][title*="remove" i] *,
        [data-testid="stFileUploader"] [role="button"][title*="delete" i] *,
        [data-testid="stFileUploader"] [role="button"][title*="clear" i] * {
            color: #111827 !important;
            fill: #111827 !important;
            stroke: #111827 !important;
        }

        /* Some Streamlit versions render the file-row action as the last child.
        Keep this scoped to the uploaded file row only. */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] > div:last-child,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] > div:last-child *,
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] > div:last-child,
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] > div:last-child * {
            background-color: #ffffff !important;
            color: #111827 !important;
            border-color: #d1d5db !important;
            color-scheme: light !important;
        }

        /* Do not let the file row action inherit dark theme surfaces. */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] [data-baseweb],
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] [data-baseweb] {
            background-color: #ffffff !important;
            color: #111827 !important;
            color-scheme: light !important;
        }

        /* If this still fails, inspect the dark control in DevTools and add its stable data-testid or aria-label selector. Avoid st-emotion-cache-* classes. */

        /* --- LocusBlend 2.5.3: uploaded-file internal scrollbar fix --- */

        /* The remaining black vertical pill in uploaded-file rows is likely an
        internal scrollbar thumb, not a button. Keep this scoped to file uploader. */
        [data-testid="stFileUploader"],
        [data-testid="stFileUploader"] *,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] {
            scrollbar-color: #cbd5e1 #ffffff !important;
            scrollbar-width: thin !important;
            color-scheme: light !important;
        }

        /* WebKit / Chrome scrollbar track inside file uploader. */
        [data-testid="stFileUploader"]::-webkit-scrollbar,
        [data-testid="stFileUploader"] *::-webkit-scrollbar,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"]::-webkit-scrollbar,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] *::-webkit-scrollbar,
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"]::-webkit-scrollbar,
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] *::-webkit-scrollbar {
            width: 8px !important;
            height: 8px !important;
            background: #ffffff !important;
            background-color: #ffffff !important;
        }

        /* WebKit / Chrome scrollbar thumb inside file uploader. */
        [data-testid="stFileUploader"]::-webkit-scrollbar-thumb,
        [data-testid="stFileUploader"] *::-webkit-scrollbar-thumb,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"]::-webkit-scrollbar-thumb,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] *::-webkit-scrollbar-thumb,
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"]::-webkit-scrollbar-thumb,
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] *::-webkit-scrollbar-thumb {
            background: #cbd5e1 !important;
            background-color: #cbd5e1 !important;
            border: 2px solid #ffffff !important;
            border-radius: 999px !important;
        }

        /* WebKit / Chrome scrollbar corner inside file uploader. */
        [data-testid="stFileUploader"]::-webkit-scrollbar-corner,
        [data-testid="stFileUploader"] *::-webkit-scrollbar-corner,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"]::-webkit-scrollbar-corner,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] *::-webkit-scrollbar-corner,
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"]::-webkit-scrollbar-corner,
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] *::-webkit-scrollbar-corner {
            background: #ffffff !important;
            background-color: #ffffff !important;
        }

        /* Keep the uploaded file row surface light even when the scrollbar is present. */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] {
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #111827 !important;
            color-scheme: light !important;
        }

        /* Do not make the file row action black when the browser creates overlay scrollbars. */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] *,
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] * {
            color-scheme: light !important;
        }

        /* Optional: make uploader file rows less likely to create tiny internal scrollbars. */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
            overflow: visible !important;
        }

        /* Fallback for Streamlit file-row wrappers that create a tiny scrollable box. */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] > div,
        [data-testid="stFileUploader"] [data-testid="stUploadedFile"] > div {
            scrollbar-color: #cbd5e1 #ffffff !important;
            scrollbar-width: thin !important;
            color-scheme: light !important;
        }

        /* --- LocusBlend 2.5.4: exact uploaded-file delete button fix --- */

        /* DevTools-confirmed target:
        div[data-testid="stFileUploaderDeleteBtn"]
        > button[data-testid="stBaseButton-minimal"][aria-label^="Remove "] */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] {
            background: transparent !important;
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color-scheme: light !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        /* Exact remove button. */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button[data-testid="stBaseButton-minimal"],
        [data-testid="stFileUploader"] button[data-testid="stBaseButton-minimal"][aria-label^="Remove "],
        [data-testid="stFileUploader"] button[kind="minimal"][aria-label^="Remove "] {
            width: 24px !important;
            height: 24px !important;
            min-width: 24px !important;
            min-height: 24px !important;
            max-width: 24px !important;
            max-height: 24px !important;
            padding: 0 !important;
            margin: 0 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #64748b !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 999px !important;
            box-shadow: none !important;
            opacity: 1 !important;
            color-scheme: light !important;
        }

        /* Hover/focus state: keep light, not black. */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button:hover,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button:focus,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button:active,
        [data-testid="stFileUploader"] button[data-testid="stBaseButton-minimal"][aria-label^="Remove "]:hover,
        [data-testid="stFileUploader"] button[data-testid="stBaseButton-minimal"][aria-label^="Remove "]:focus,
        [data-testid="stFileUploader"] button[data-testid="stBaseButton-minimal"][aria-label^="Remove "]:active {
            background: #f8fafc !important;
            background-color: #f8fafc !important;
            color: #334155 !important;
            border: 1px solid #94a3b8 !important;
            box-shadow: none !important;
            outline: none !important;
        }

        /* Remove icon SVG sizing and color. */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] svg,
        [data-testid="stFileUploader"] button[data-testid="stBaseButton-minimal"][aria-label^="Remove "] svg,
        [data-testid="stFileUploader"] button[kind="minimal"][aria-label^="Remove "] svg {
            width: 14px !important;
            height: 14px !important;
            color: #64748b !important;
            fill: none !important;
            stroke: #64748b !important;
            background: transparent !important;
            background-color: transparent !important;
        }

        /* Remove icon path. Critical: do not give the path a white/black background box. */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] svg path,
        [data-testid="stFileUploader"] button[data-testid="stBaseButton-minimal"][aria-label^="Remove "] svg path,
        [data-testid="stFileUploader"] button[kind="minimal"][aria-label^="Remove "] svg path {
            color: #64748b !important;
            fill: none !important;
            stroke: #64748b !important;
            background: transparent !important;
            background-color: transparent !important;
        }

        /* Hover/focus icon color. */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button:hover svg,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button:hover svg path,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button:focus svg,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button:focus svg path,
        [data-testid="stFileUploader"] button[data-testid="stBaseButton-minimal"][aria-label^="Remove "]:hover svg,
        [data-testid="stFileUploader"] button[data-testid="stBaseButton-minimal"][aria-label^="Remove "]:hover svg path {
            color: #334155 !important;
            fill: none !important;
            stroke: #334155 !important;
        }

        /* Keep uploaded file row layout stable. */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
            align-items: center !important;
        }

        /* Avoid older broad rules making the delete icon into a dark block. */
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] *,
        [data-testid="stFileUploader"] button[data-testid="stBaseButton-minimal"][aria-label^="Remove "] * {
            box-shadow: none !important;
            text-shadow: none !important;
        }

        /* --- LocusBlend 2.6: WashU Medicine header --- */

        .lb-washu-header {
            width: 100%;
            background: #A51417 !important;
            background-color: #A51417 !important;
            color: #ffffff !important;
            border-radius: 0;
            margin: -0.5rem 0 1.25rem 0;
            padding: 0;
            box-shadow: none;
            color-scheme: light !important;
        }

        .lb-washu-header-inner {
            min-height: 44px;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            padding: 0.45rem 1.25rem;
        }

        .lb-washu-logo {
            display: block;
            height: 30px;
            max-width: 260px;
            object-fit: contain;
        }

        .lb-washu-logo-fallback {
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: 0;
            color: #ffffff !important;
        }

        @media (max-width: 700px) {
            .lb-washu-header {
                margin-top: -0.25rem;
                margin-bottom: 1rem;
            }

            .lb-washu-header-inner {
                min-height: 40px;
                padding: 0.4rem 0.85rem;
            }

            .lb-washu-logo {
                height: 25px;
                max-width: 220px;
            }
        }

        /* --- LocusBlend 2.6.1: full-width fixed WashU Medicine header --- */

        :root {
            --lb-washu-header-height: 54px;
            --lb-washu-red: #A51417;
        }

        /* Fixed global header across sidebar + main content. */
        .lb-washu-global-header {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            right: 0 !important;
            width: 100vw !important;
            height: var(--lb-washu-header-height) !important;
            z-index: 999990 !important;
            background: var(--lb-washu-red) !important;
            background-color: var(--lb-washu-red) !important;
            color: #ffffff !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            margin: 0 !important;
            padding: 0 !important;
            border: none !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            color-scheme: light !important;
        }

        /* Header content width. Keep logo aligned with main app content, but header background spans all. */
        .lb-washu-global-inner {
            width: 100% !important;
            height: var(--lb-washu-header-height) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            padding: 0 1.5rem !important;
            box-sizing: border-box !important;
        }

        /* Logo in fixed header. */
        .lb-washu-header-link {
            display: inline-flex !important;
            align-items: center !important;
            text-decoration: none !important;
            color: inherit !important;
        }

        .lb-washu-header-link:visited,
        .lb-washu-header-link:hover,
        .lb-washu-header-link:active {
            text-decoration: none !important;
            color: inherit !important;
        }

        .lb-washu-header-link img {
            display: block !important;
        }

        .lb-washu-global-header .lb-washu-logo {
            display: block !important;
            height: 32px !important;
            max-width: 280px !important;
            object-fit: contain !important;
        }

        /* Fallback text if local logo is unavailable. */
        .lb-washu-global-header .lb-washu-logo-fallback {
            font-size: 1.25rem !important;
            font-weight: 700 !important;
            color: #ffffff !important;
            letter-spacing: 0.01em !important;
        }

        /* Hide/neutralize old non-fixed 2.6 header container if still rendered. */
        .lb-washu-header {
            display: none !important;
        }

        /* Push the Streamlit app content below the fixed header. */
        [data-testid="stAppViewContainer"] {
            padding-top: var(--lb-washu-header-height) !important;
        }

        /* Push sidebar content below the fixed header. */
        [data-testid="stSidebar"] {
            padding-top: var(--lb-washu-header-height) !important;
        }

        /* Ensure sidebar background begins below header, while header still spans above it. */
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-top: 1rem !important;
        }

        /* Main block should not add another huge top gap. */
        [data-testid="stMain"] .block-container {
            padding-top: 2rem !important;
        }

        /* Streamlit's own top header can otherwise create a blank strip.
           Keep it transparent and visually minimized without removing app controls. */
        [data-testid="stHeader"] {
            background: transparent !important;
            height: 0 !important;
            min-height: 0 !important;
        }

        /* Keep Streamlit sidebar controls selectable above the banner if present. */
        [data-testid="stSidebarCollapseButton"] {
            z-index: 1000000 !important;
        }

        /* Responsive header. */
        @media (max-width: 700px) {
            :root {
                --lb-washu-header-height: 48px;
            }

            .lb-washu-global-inner {
                padding: 0 1rem !important;
            }

            .lb-washu-global-header .lb-washu-logo {
                height: 27px !important;
                max-width: 230px !important;
            }

            [data-testid="stMain"] .block-container {
                padding-top: 1.5rem !important;
            }
        }

        /* --- LocusBlend 2.6.3: keep sidebar expanded for review/demo --- */

        /* Do not attempt to restyle Streamlit's collapsed/reopen button.
        For this review build, prevent users from collapsing the sidebar.
        This avoids the known issue where the sidebar can be hard to reopen. */

        /* Hide only the native "close/collapse sidebar" control when the sidebar is expanded.
        Keep the selector narrow and scoped to the sidebar header. */
        [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebar"] [data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"] {
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }

        /* Keep the sidebar itself visible and normal. */
        [data-testid="stSidebar"] {
            visibility: visible !important;
            opacity: 1 !important;
        }

        /* Preserve the app-wide WashU header from app.2.6.1.
        Do not touch native sidebar reopen controls here. */

        /* --- LocusBlend 2.6.4: remove empty sidebar header gap --- */

        /* In this review/demo build, the sidebar collapse button is intentionally hidden.
        Streamlit's sidebar header container then becomes empty but still occupies
        space above the Legend. Collapse only that empty header area. */
        [data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
            height: 0 !important;
            min-height: 0 !important;
            max-height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            overflow: hidden !important;
        }

        /* Keep the actual collapse button hidden, as in app.2.6.3. */
        [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebar"] [data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"] {
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }

        /* Do not change the global WashU header or Streamlit main/header layers here. */

        /* --- LocusBlend 2.6.5: sidebar app title --- */

        /* Keep the native Streamlit sidebar collapse mechanics untouched here.
           This build adds an app-level sidebar title instead of trying to restyle
           native sidebar controls. */

        /* Use the sidebar top area for an intentional app title. */
        .lb-sidebar-app-header {
            margin: 0 0 0.85rem 0 !important;
            padding: 0.85rem 0.9rem !important;
            border-radius: 10px !important;
            background: #ffffff !important;
            background-color: #ffffff !important;
            border: 1px solid #e5e7eb !important;
            color: #111827 !important;
            box-shadow: none !important;
            color-scheme: light !important;
        }

        .lb-sidebar-app-title {
            font-size: 1.05rem !important;
            font-weight: 800 !important;
            line-height: 1.2 !important;
            color: #111827 !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        .lb-sidebar-app-subtitle {
            font-size: 0.78rem !important;
            font-weight: 500 !important;
            line-height: 1.25 !important;
            color: #6b7280 !important;
            margin-top: 0.25rem !important;
            padding: 0 !important;
        }

        /* Reduce only the visible top spacing inside sidebar user content.
           Do not touch native sidebar buttons. */
        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            padding-top: 0.75rem !important;
        }

        /* Keep the empty Streamlit sidebar header compact as in app.2.6.4. */
        [data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
            height: 0 !important;
            min-height: 0 !important;
            max-height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            overflow: hidden !important;
        }

        /* Keep the sidebar collapse button hidden for this review/demo build. */
        [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebar"] [data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"] {
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }

        /* --- LocusBlend 2.6.6: cleaner sidebar top label --- */

        /* Replace the heavy sidebar title card from 2.6.5 with a compact label. */
        .lb-sidebar-app-header {
            display: none !important;
        }

        /* Compact sidebar label with a WashU-red accent. */
        .lb-sidebar-app-label {
            margin: 0.25rem 0 1.1rem 0 !important;
            padding: 0.15rem 0 0.15rem 0.7rem !important;
            border-left: 4px solid #A51417 !important;
            background: transparent !important;
            color: #111827 !important;
            box-shadow: none !important;
            color-scheme: light !important;
        }

        .lb-sidebar-app-label span {
            font-size: 0.95rem !important;
            font-weight: 800 !important;
            line-height: 1.2 !important;
            color: #111827 !important;
            letter-spacing: 0.01em !important;
        }

        /* Reduce only the visible top spacing inside sidebar user content.
           Do not touch native sidebar buttons. */
        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            padding-top: 0.65rem !important;
        }

        /* Keep the empty Streamlit sidebar header compact. */
        [data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
            height: 0 !important;
            min-height: 0 !important;
            max-height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            overflow: hidden !important;
        }

        /* Keep the sidebar collapse button hidden for this review/demo build. */
        [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebar"] [data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"] {
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }

        /* --- LocusBlend 2.6.7: top update button hint --- */

        .lb-sidebar-update-hint {
            margin: 0.25rem 0 0.9rem 0 !important;
            padding: 0 !important;
            font-size: 0.76rem !important;
            line-height: 1.25 !important;
            color: #6b7280 !important;
        }

        /* --- LocusBlend 2.6.10: top-level Visualizer / Documentation switcher --- */

        [data-testid="stRadio"] {
            color-scheme: light !important;
        }

        /* Keep the page switcher visually compact. */
        .lb-doc-note {
            color: #6b7280 !important;
            font-size: 0.9rem !important;
            line-height: 1.45 !important;
        }

        /* Documentation page readability. */
        .lb-doc-section {
            margin-top: 1.25rem !important;
            margin-bottom: 1.25rem !important;
        }

        .lb-doc-section h2,
        .lb-doc-section h3 {
            color: #111827 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_app_header():
    """Render a compact app-level label at the top of the sidebar."""
    st.sidebar.markdown(
        """
<div class="lb-sidebar-app-label">
  <span>LocusBlend Controls</span>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_visual_strip():
    legend_overlay_path = ASSET_DIR / "legend_overlay.png"
    drawing3_path = ASSET_DIR / "Drawing3.png"

    items = []
    if legend_overlay_path.exists():
        items.append(("legend", "LD legend", legend_overlay_path))
    if drawing3_path.exists():
        items.append(("diagram", "LocusBlend diagram", drawing3_path))

    if not items:
        return

    single = len(items) == 1
    html_chunks = []
    for kind, alt, path in items:
        src = image_to_data_uri(path)
        safe_alt = html_lib.escape(alt, quote=True)
        pane_class = "lb-sidebar-visual-pane--single" if single else f"lb-sidebar-visual-pane--{kind}"
        html_chunks.append(
            f'''
            <div class="lb-sidebar-visual-pane {pane_class}">
                <img src="{src}" alt="{safe_alt}" class="lb-sidebar-visual-img" />
            </div>
            '''
        )

    visual_html = f"""
    <!doctype html>
    <html>
    <head>
    <style>
    html, body {{
        margin: 0;
        padding: 0;
        background: transparent;
        overflow: hidden;
        color-scheme: light;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    .lb-sidebar-visual-card {{
        width: 100%;
        box-sizing: border-box;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 8px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        overflow: hidden;
    }}

    .lb-sidebar-visual-row {{
        display: flex;
        flex-direction: row;
        flex-wrap: nowrap;
        align-items: center;
        justify-content: center;
        gap: 8px;
        width: 100%;
        min-width: 0;
        box-sizing: border-box;
    }}

    .lb-sidebar-visual-pane {{
        min-width: 0;
        max-width: 100%;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        box-sizing: border-box;
    }}

    .lb-sidebar-visual-pane--legend {{
        flex: 1.08 1 0;
    }}

    .lb-sidebar-visual-pane--diagram {{
        flex: 0.92 1 0;
    }}

    .lb-sidebar-visual-pane--single {{
        flex: 1 1 auto;
    }}

    .lb-sidebar-visual-img {{
        display: block;
        width: 100%;
        max-width: 100%;
        min-width: 0;
        height: auto;
        max-height: 125px;
        object-fit: contain;
        box-sizing: border-box;
    }}

    @media (max-width: 280px) {{
        .lb-sidebar-visual-card {{
            padding: 6px;
        }}

        .lb-sidebar-visual-row {{
            gap: 4px;
        }}

        .lb-sidebar-visual-img {{
            max-height: 115px;
        }}
    }}

    @media (max-width: 220px) {{
        .lb-sidebar-visual-card {{
            padding: 4px;
        }}

        .lb-sidebar-visual-row {{
            gap: 3px;
        }}

        .lb-sidebar-visual-img {{
            max-height: 100px;
        }}
    }}
    </style>
    </head>
    <body>
        <div class="lb-sidebar-visual-card">
            <div class="lb-sidebar-visual-row">
                {''.join(html_chunks)}
            </div>
        </div>
    </body>
    </html>
    """

    st.sidebar.markdown("### LD legend")
    with st.sidebar:
        components.html(visual_html, height=165, scrolling=False)


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


def clone_plotly_figure(fig):
    """Return a detached copy of a Plotly figure so export layout changes do not mutate cached figures."""
    if fig is None:
        return None
    try:
        return go.Figure(fig.to_dict())
    except Exception:
        return copy.deepcopy(fig)


def _load_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:
        raise RuntimeError(
            "Pillow is required to compose PNG/PDF exports. Install it with: pip install pillow"
        ) from e
    return Image, ImageDraw, ImageFont


def _get_resample_filter():
    Image, _, _ = _load_pillow()
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    if hasattr(Image, "LANCZOS"):
        return Image.LANCZOS
    return Image.BICUBIC


def _pil_font(size, bold=False):
    _, _, ImageFont = _load_pillow()
    candidates = ["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf"]
    for font_name in candidates:
        try:
            return ImageFont.truetype(font_name, int(size))
        except Exception:
            pass
    return ImageFont.load_default()


def _text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0, 0), str(text), font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        try:
            return draw.textsize(str(text), font=font)
        except Exception:
            return len(str(text)) * 7, 12


def _draw_wrapped_text(draw, text, xy, max_width, font, fill, line_spacing=4):
    x, y = xy
    max_width = max(1, int(max_width))

    def draw_line(line_text, line_y):
        draw.text((x, line_y), line_text, font=font, fill=fill)
        _, line_h = _text_size(draw, line_text or "Ag", font)
        return line_y + line_h + line_spacing

    for raw_line in str(text or "").splitlines():
        words = raw_line.split()
        if not words:
            y = draw_line("", y)
            continue

        line = ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            candidate_w, _ = _text_size(draw, candidate, font)
            if candidate_w <= max_width or not line:
                line = candidate
            else:
                y = draw_line(line, y)
                line = word
        if line:
            y = draw_line(line, y)
    return y


def _format_export_context(context):
    context = dict(context or {})
    lines = []

    mode = context.get("mode")
    compare_mode = context.get("compare_mode")
    if mode:
        lines.append(f"Mode: {mode}")
    if compare_mode:
        lines.append(f"Compare: {compare_mode}")

    chrom = context.get("chromosome")
    center_bp = context.get("center_bp")
    window_kb = context.get("window_kb")
    locus_parts = []
    if chrom not in (None, ""):
        locus_parts.append(f"chr{chrom}")
    if center_bp not in (None, ""):
        try:
            locus_parts.append(f"center {int(center_bp):,} bp")
        except Exception:
            locus_parts.append(f"center {center_bp} bp")
    if window_kb not in (None, ""):
        try:
            locus_parts.append(f"+/- {int(window_kb):,} kb")
        except Exception:
            locus_parts.append(f"+/- {window_kb} kb")
    if locus_parts:
        lines.append("Locus: " + ", ".join(locus_parts))

    title_top = context.get("title_top")
    title_bottom = context.get("title_bottom")
    title_parts = [str(v) for v in [title_top, title_bottom] if v not in (None, "")]
    if title_parts:
        lines.append("Sources: " + " / ".join(title_parts))

    highlight_genes = str(context.get("highlight_genes", "") or "").strip()
    if highlight_genes:
        lines.append(f"Highlighted genes: {highlight_genes}")

    ld_status_caption = str(context.get("ld_status_caption", "") or "").strip()
    if ld_status_caption:
        lines.append(ld_status_caption)

    return "\n".join(lines)


def render_plotly_figure_to_png_bytes(fig, width_px, height_px, scale=1):
    """Render a cloned Plotly figure to PNG bytes using kaleido.

    This must not mutate the cached session_state figure.
    """
    if fig is None:
        raise ValueError("No Plotly figure is available for export.")

    width_px = max(1, int(width_px))
    height_px = max(1, int(height_px))

    fig_copy = clone_plotly_figure(fig)
    fig_copy.update_layout(
        width=width_px,
        height=height_px,
        autosize=False,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )

    try:
        return pio.to_image(
            fig_copy,
            format="png",
            width=width_px,
            height=height_px,
            scale=scale,
        )
    except Exception as e:
        raise RuntimeError(
            "Static export failed. Plotly image export requires kaleido. "
            "Depending on the installed kaleido version, Chrome may also be required on the server."
        ) from e


def open_pil_image_from_bytes(data):
    Image, _, _ = _load_pillow()
    return Image.open(BytesIO(data)).convert("RGBA")


def open_pil_asset(path):
    if path is None:
        return None
    try:
        path = Path(path)
        if not path.exists():
            return None
        Image, _, _ = _load_pillow()
        return Image.open(path).convert("RGBA")
    except Exception:
        return None


def resize_contained(img, max_w, max_h):
    if img is None:
        return None
    max_w = max(1, int(max_w))
    max_h = max(1, int(max_h))
    src_w, src_h = img.size
    if src_w <= 0 or src_h <= 0:
        return img.resize((1, 1), _get_resample_filter())
    scale = min(max_w / src_w, max_h / src_h)
    target_w = max(1, int(round(src_w * scale)))
    target_h = max(1, int(round(src_h * scale)))
    return img.resize((target_w, target_h), _get_resample_filter())


def paste_contained(canvas, img, box, align="center"):
    if canvas is None or img is None:
        return None

    x, y, w, h = [int(v) for v in box]
    fitted = resize_contained(img, w, h)
    if fitted is None:
        return None

    if align == "left":
        paste_x = x
    elif align == "right":
        paste_x = x + max(0, w - fitted.width)
    else:
        paste_x = x + max(0, (w - fitted.width) // 2)
    paste_y = y + max(0, (h - fitted.height) // 2)

    if fitted.mode == "RGBA":
        canvas.paste(fitted, (paste_x, paste_y), fitted)
    else:
        canvas.paste(fitted, (paste_x, paste_y))
    return paste_x, paste_y, fitted.width, fitted.height


def draw_export_header(canvas, draw, context, include_legends, header_box):
    x, y, w, h = [int(v) for v in header_box]
    pad = max(12, int(h * 0.10))
    border = "#d1d5db"
    fill = "#f8fafc"
    text = "#111827"
    muted = "#374151"

    try:
        draw.rounded_rectangle([x, y, x + w, y + h], radius=max(8, h // 18), fill=fill, outline=border, width=1)
    except Exception:
        draw.rectangle([x, y, x + w, y + h], fill=fill, outline=border, width=1)

    title_font = _pil_font(max(18, min(44, h // 7)), bold=True)
    body_font = _pil_font(max(11, min(24, h // 15)), bold=False)

    legend_imgs = []
    if include_legends:
        for asset_name in ["legend_overlay.png", "Drawing3.png"]:
            img = open_pil_asset(ASSET_DIR / asset_name)
            if img is not None:
                legend_imgs.append(img)

    inner_w = max(1, w - 2 * pad)
    legend_w = int(inner_w * 0.36) if legend_imgs else 0
    legend_gap = max(8, pad // 2)
    text_w = inner_w - legend_w - (legend_gap if legend_imgs else 0)
    text_x = x + pad
    text_y = y + pad

    draw.text((text_x, text_y), "LocusBlend export", font=title_font, fill=text)
    _, title_h = _text_size(draw, "LocusBlend export", title_font)
    context_text = _format_export_context(context)
    if context_text:
        _draw_wrapped_text(
            draw=draw,
            text=context_text,
            xy=(text_x, text_y + title_h + max(6, pad // 3)),
            max_width=max(1, text_w),
            font=body_font,
            fill=muted,
            line_spacing=max(3, h // 70),
        )

    if legend_imgs:
        legend_x = x + w - pad - legend_w
        legend_y = y + pad
        legend_h = max(1, h - 2 * pad)
        item_gap = max(6, legend_w // 36)
        item_w = max(1, (legend_w - item_gap * (len(legend_imgs) - 1)) // len(legend_imgs))
        for i, img in enumerate(legend_imgs):
            item_x = legend_x + i * (item_w + item_gap)
            paste_contained(canvas, img, (item_x, legend_y, item_w, legend_h))


def build_locusblend_export_image(
    locus_fig,
    compare_fig,
    export_context=None,
    output_width_in=8.5,
    output_height_in=11.0,
    dpi=300,
    include_legends=True,
):
    Image, ImageDraw, _ = _load_pillow()

    output_width_in = float(output_width_in)
    output_height_in = float(output_height_in)
    dpi = int(dpi)
    if not (3 <= output_width_in <= 40 and 3 <= output_height_in <= 40):
        raise ValueError("Export page dimensions must be between 3 and 40 inches.")
    if not (72 <= dpi <= 600):
        raise ValueError("Export DPI must be between 72 and 600.")

    canvas_w = int(output_width_in * dpi)
    canvas_h = int(output_height_in * dpi)
    if canvas_w * canvas_h > 80_000_000:
        raise ValueError("The requested export is too large. Reduce page size or DPI and try again.")

    canvas = Image.new("RGB", (canvas_w, canvas_h), "#ffffff")
    draw = ImageDraw.Draw(canvas)

    margin = max(30, int(0.35 * dpi))
    gap = max(16, int(0.12 * dpi))
    usable_w = canvas_w - 2 * margin
    usable_h = canvas_h - 2 * margin
    if usable_w <= 0 or usable_h <= 0:
        raise ValueError("Export page is too small for the requested margins.")

    header_h = int(1.15 * dpi) if include_legends else int(0.55 * dpi)
    header_h = max(header_h, 170 if include_legends else 95)
    header_h = min(header_h, max(60, int(usable_h * 0.34)))

    header_box = (margin, margin, usable_w, header_h)
    draw_export_header(canvas, draw, export_context or {}, include_legends, header_box)

    plot_top = margin + header_h + gap
    plot_h = canvas_h - plot_top - margin
    if plot_h <= gap + 2:
        raise ValueError("Export page is too small for both cached plots.")

    compare_mode = str((export_context or {}).get("compare_mode", ""))
    if compare_mode == "Single blended compare plot":
        locus_h = int(plot_h * 0.64)
    else:
        locus_h = int(plot_h * 0.62)
    compare_h = plot_h - locus_h - gap
    locus_h = max(1, locus_h)
    compare_h = max(1, compare_h)

    locus_png = render_plotly_figure_to_png_bytes(locus_fig, usable_w, locus_h, scale=1)
    locus_img = open_pil_image_from_bytes(locus_png)
    locus_box = (margin, plot_top, usable_w, locus_h)
    paste_contained(canvas, locus_img, locus_box)

    compare_y = plot_top + locus_h + gap
    compare_box = (margin, compare_y, usable_w, compare_h)
    if compare_mode == "Single blended compare plot":
        compare_render_size = max(1, min(usable_w, compare_h))
        compare_png = render_plotly_figure_to_png_bytes(
            compare_fig,
            compare_render_size,
            compare_render_size,
            scale=1,
        )
    else:
        compare_png = render_plotly_figure_to_png_bytes(compare_fig, usable_w, compare_h, scale=1)
    compare_img = open_pil_image_from_bytes(compare_png)
    paste_contained(canvas, compare_img, compare_box)

    return canvas.convert("RGB")


def make_locusblend_export_bytes(
    output_format,
    width_in,
    height_in,
    dpi,
    include_legends,
):
    locus_fig = st.session_state.get("last_locus_fig")
    compare_fig = st.session_state.get("last_compare_fig")
    export_context = st.session_state.get("last_export_context", {})
    compare_mode = st.session_state.get("last_compare_mode", export_context.get("compare_mode", ""))

    if locus_fig is None or compare_fig is None:
        raise ValueError("No cached plot is available yet. Select Update plot first.")

    export_context = dict(export_context or {})
    export_context["compare_mode"] = compare_mode

    canvas = build_locusblend_export_image(
        locus_fig=locus_fig,
        compare_fig=compare_fig,
        export_context=export_context,
        output_width_in=float(width_in),
        output_height_in=float(height_in),
        dpi=int(dpi),
        include_legends=bool(include_legends),
    )

    buf = BytesIO()
    fmt = str(output_format).upper()

    if fmt == "PNG":
        canvas.save(buf, format="PNG")
        return buf.getvalue(), "image/png", "locusblend_export.png"

    if fmt == "PDF":
        canvas.convert("RGB").save(buf, format="PDF", resolution=int(dpi))
        return buf.getvalue(), "application/pdf", "locusblend_export.pdf"

    raise ValueError(f"Unsupported export format: {output_format}")


def render_export_controls():
    if "last_locus_fig" not in st.session_state or "last_compare_fig" not in st.session_state:
        return

    st.markdown("### Export / Print current figure")
    with st.expander("Export settings", expanded=False):
        st.caption(
            "Exports the currently rendered cached plot. "
            "Sidebar edits are not applied until you select Update plot."
        )

        output_format = st.selectbox(
            "Output format",
            ["PNG", "PDF"],
            index=0,
            key="export_output_format",
        )

        paper_preset = st.selectbox(
            "Page size",
            [
                "US Letter portrait (8.5 x 11 in)",
                "US Letter landscape (11 x 8.5 in)",
                "Custom",
            ],
            index=0,
            key="export_paper_preset",
        )

        if paper_preset == "US Letter portrait (8.5 x 11 in)":
            default_w, default_h = 8.5, 11.0
        elif paper_preset == "US Letter landscape (11 x 8.5 in)":
            default_w, default_h = 11.0, 8.5
        else:
            default_w = float(st.session_state.get("export_custom_width_in", 8.5))
            default_h = float(st.session_state.get("export_custom_height_in", 11.0))

        if paper_preset == "Custom":
            col_w, col_h = st.columns(2)
            with col_w:
                width_in = st.number_input(
                    "Width (in)",
                    min_value=4.0,
                    max_value=30.0,
                    value=float(default_w),
                    step=0.25,
                    key="export_custom_width_in",
                )
            with col_h:
                height_in = st.number_input(
                    "Height (in)",
                    min_value=4.0,
                    max_value=30.0,
                    value=float(default_h),
                    step=0.25,
                    key="export_custom_height_in",
                )
        else:
            width_in, height_in = default_w, default_h
            st.caption(f"Output size: {width_in:g} x {height_in:g} inches")

        col_dpi, col_leg = st.columns(2)
        with col_dpi:
            dpi = st.number_input(
                "DPI",
                min_value=72,
                max_value=600,
                value=300,
                step=25,
                key="export_dpi",
            )
        with col_leg:
            _export_legend_default = bool(st.session_state.get("export_include_legends", True))
            _export_legend_choice = st.selectbox(
                "Export legend images",
                options=["Include", "Do not include"],
                index=0 if _export_legend_default else 1,
                key="export_include_legends_choice",
                help="Include or omit the LD legend images in the exported PNG/PDF.",
            )
            include_legends = _export_legend_choice == "Include"
            st.session_state["export_include_legends"] = include_legends

        current_signature = (
            output_format,
            paper_preset,
            float(width_in),
            float(height_in),
            int(dpi),
            bool(include_legends),
            id(st.session_state.get("last_locus_fig")),
            id(st.session_state.get("last_compare_fig")),
        )

        prepare_export = st.button(
            "Prepare export file",
            type="secondary",
            key="prepare_locusblend_export",
        )

        if prepare_export:
            try:
                data, mime, filename = make_locusblend_export_bytes(
                    output_format=output_format,
                    width_in=float(width_in),
                    height_in=float(height_in),
                    dpi=int(dpi),
                    include_legends=bool(include_legends),
                )
                st.session_state["last_export_bytes"] = data
                st.session_state["last_export_mime"] = mime
                st.session_state["last_export_filename"] = filename
                st.session_state["last_export_signature"] = current_signature
                st.session_state.pop("last_export_error", None)
                st.success(f"Export prepared: {filename}")
            except Exception as e:
                st.session_state["last_export_error"] = str(e)
                st.session_state.pop("last_export_bytes", None)
                st.session_state.pop("last_export_mime", None)
                st.session_state.pop("last_export_filename", None)
                st.session_state.pop("last_export_signature", None)

        if st.session_state.get("last_export_error"):
            st.error(st.session_state["last_export_error"])

        if (
            st.session_state.get("last_export_bytes") is not None
            and st.session_state.get("last_export_signature") == current_signature
        ):
            st.download_button(
                "Download export",
                data=st.session_state["last_export_bytes"],
                file_name=st.session_state.get("last_export_filename", "locusblend_export.png"),
                mime=st.session_state.get("last_export_mime", "application/octet-stream"),
                key="download_locusblend_export",
                use_container_width=True,
            )
        elif st.session_state.get("last_export_bytes") is not None:
            st.info("Export options changed. Select Prepare export file again.")


def render_locusblend_welcome_card():
    """Render a simple non-modal welcome panel.

    This replaces the old automatic welcome dialog. It must not change
    session_state, must not trigger recomputation, and must not open a modal.
    """
    with st.container(border=True):
        st.markdown("### Welcome to LocusBlend")
        st.markdown(
            """
LocusBlend visualizes GWAS summary statistics with multi-variant linkage
disequilibrium (LD) coloring, gene-track annotation, and locus compare plots.

**Basic workflow:**

1. Choose a visualization mode at the top of the page.
2. Configure data, LD source, locus, index variants, and display settings in the sidebar.
3. Select **Update plot** to apply changes.

Sidebar edits are intentionally not applied until **Update plot** is selected.
"""
        )

        c1, c2, c3 = st.columns(3)
        c1.markdown("**Modes**  \nStandard single-index, two-index, or three-index LocusBlend")
        c2.markdown("**Data**  \nUse default datasets or upload CSV, TSV, or TXT summary-statistic files")
        c3.markdown("**Output**  \nInteract with plots or export the current cached figure")


def render_washu_header():
    """Render a fixed full-width WashU Medicine-style app header."""
    logo_path = ASSET_DIR / "washulogo.png"
    try:
        logo_uri = image_to_data_uri(logo_path) if logo_path.exists() else None
    except Exception:
        logo_uri = None

    if logo_uri:
        logo_html = f'<img src="{logo_uri}" alt="WashU Medicine" class="lb-washu-logo" />'
    else:
        logo_html = '<span class="lb-washu-logo-fallback">WashU Medicine</span>'

    st.markdown(
        f"""
<div class="lb-washu-global-header" role="banner">
  <a class="lb-washu-header-link" href="https://medicine.washu.edu/" target="_blank" rel="noopener noreferrer" aria-label="WashU Medicine home">
    <div class="lb-washu-global-inner">
      {logo_html}
    </div>
  </a>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_documentation_page():
    """Render the full LocusBlend user guide in the main content area."""
    st.markdown("## Documentation")
    st.markdown(
        '<div class="lb-doc-note">Use this page as the persistent LocusBlend reference while keeping the visualizer focused on plots and controls.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
## Overview

**LocusBlend** visualizes GWAS summary statistics with multi-variant linkage
disequilibrium (LD) coloring, gene-track annotation, and locus compare plots.

The app supports:

* **Standard locus zoom**: one index variant
* **Two-index LocusBlend**: two index variants with blended LD coloring
* **Three-index LocusBlend**: three index variants with blended LD coloring

## Quick start

1. Select a mode at the top of the page.
2. Open the sidebar sections and configure:

   * datasets
   * LD source
   * locus window
   * index variants
   * plot display
   * gene track
   * locus compare
3. Select **Update plot**.
4. Use hover / zoom / pan in Plotly, or use **Export / Print current figure**.

## Important interaction model

Changing sidebar widgets reruns the Streamlit script, but it does **not**
recompute LD or rebuild the figures immediately.

The plot updates only when:

* **Update plot** is selected, or
* the app is loaded for the first time and no cached figure exists yet.

This design prevents expensive LD / PLINK computation from running after every
small parameter edit.

## Input file requirements

Supported uploaded summary-statistic file formats: `.csv`, `.tsv`, `.txt`,
`.csv.gz`, `.tsv.gz`, and `.txt.gz`.

Required genome build: GRCh38 / hg38. LocusBlend does not perform liftover. If
your summary-statistic file uses GRCh37 / hg19 or hg18 coordinates, lift it over
to GRCh38 / hg38 before upload. Otherwise internal 1000G LD matching, gene
annotation, locus windows, and index-variant matching may be incorrect or fail.

Required columns:

* `CHR`
* `BP`
* `P`
* `A1`
* `A2`

Recommended columns:

* `rsid`
* `BETA`
* `SE`
* `A1FREQ`
* `N`

Accepted aliases:

| Concept            | Accepted column names                               |
| ------------------ | --------------------------------------------------- |
| Chromosome         | `CHR`, `chr`, `#chr`, `chrom`, `chromosome`         |
| Base-pair position | `BP`, `bp`, `pos`, `position`, `base_pair_location` |
| P value            | `P`, `p`, `pval`, `pvalue`, `P-value`               |
| Variant ID         | `rsid`, `RSID`, `SNP`, `MarkerName`, `ID`           |
| Effect allele      | `A1`, `EA`, `effect_allele`, `ALLELE1`              |
| Other allele       | `A2`, `NEA`, `other_allele`, `ALLELE0`              |
| Effect size        | `BETA`, `beta`, `Effect`, `estimate`                |
| Standard error     | `SE`, `StdErr`, `stderr`                            |
| Allele frequency   | `A1FREQ`, `EAF`, `MAF`, `freq`                      |
| Sample size        | `N`, `n`, `samplesize`                              |

Example:

| CHR |       BP | rsid       |    P | A1 | A2 | BETA |   SE |
| --- | -------: | ---------- | ---: | -- | -- | ---: | ---: |
| 14  | 73238768 | rs11159021 | 1e-6 | A  | G  | 0.12 | 0.03 |

Supported chromosomes are **1-22 and X**. Chromosome X may be provided in
uploaded data as `X`, `chrX`, `23`, or `chr23`; the app normalizes these values
to `X` internally. X-based allele IDs use the same `CHR:BP:A1:A2` pattern as
autosomes, for example `X:13189:A:G`.

## LD source options

### Internal 1000G reference

No upload is required. Select one of the five 1000 Genomes super-populations:
AFR, AMR, EAS, EUR, or SAS. The selected ancestry determines which
chromosome-specific PLINK binary reference panel is used for LD calculation.
The default is EUR.

Uploaded summary-statistic coordinates must use GRCh38 / hg38; no liftover is
performed. Each selected ancestry requires local PLINK files in the app `data/`
directory using this prefix pattern:

`1000g_{AFR|AMR|EAS|EUR|SAS}_hg38_high_coverage_Illumina.filtered.SNV_INDEL_SV_phased_panel_include_MHC_ch{chrom}`

### Uploaded LD matrix

Use a square R2 matrix with SNP IDs as both row and column names. Cells should
contain pairwise R2 values from 0 to 1.

Example:

| SNP | rs1 | rs2 | rs3 |
| --- | --: | --: | --: |
| rs1 | 1.0 | 0.7 | 0.1 |
| rs2 | 0.7 | 1.0 | 0.3 |
| rs3 | 0.1 | 0.3 | 1.0 |

### Uploaded LD long table

Required columns:

* `SNP_A`
* `SNP_B`
* `R2`

Accepted aliases:

* `SNP_A`: `SNP_A`, `SNP1`, `ID1`
* `SNP_B`: `SNP_B`, `SNP2`, `ID2`
* `R2`: `R2`, `r2`, `rsq`

The long table must include self-pair rows where `SNP_A == SNP_B` and `R2 == 1`.
These rows allow the app to distinguish variants present in the LD reference
from variants that are missing.

Example:

| SNP_A | SNP_B |  R2 |
| ----- | ----- | --: |
| rs1   | rs1   | 1.0 |
| rs2   | rs2   | 1.0 |
| rs1   | rs2   | 0.7 |

## Locus settings

Use the sidebar to choose:

* chromosome (1-22 or X)
* center base-pair position
* window size in kb

The window is interpreted as `center +/- window_kb`.

## Index variants

Manual input:

* Enter one, two, or three variant IDs depending on the active mode.

Auto-select:

* The app can select independent index variants from either dataset 1 (top) or
  dataset 2 (bottom) using LD clumping.
* Auto-selection uses LD clumping at r2 = 0.01.
* The number of selected variants depends on the active mode.

## Plot display

Use plot display settings to control:

* dataset 1 (top) and dataset 2 (bottom) plot titles
* y-axis limits
* combined plot height
* vertical spacing
* recombination-rate display
* recombination y-axis maximum

## Gene track

Gene display modes:

* `protein_coding`
* `all`

Gene highlighting:

* Enter comma- or semicolon-separated gene names.
* Matching is case-insensitive.
* Highlighted genes are drawn more prominently in the gene track.

Example:

`PSEN1, PAPLN, HEATR4`

## Locus compare

The locus compare panel supports:

* **Three separate compare plots**: one panel per active index variant
* **Single blended compare plot**: all variants overlaid in one blended plot

The number of active panels depends on the active mode:

* Standard: variant 1
* Two-index: variants 1 and 2
* Three-index: variants 1, 2, and 3

## Export / Print current figure

The export panel exports the currently rendered cached plot. It does not apply
pending sidebar edits.

Supported formats:

* PNG
* PDF

Default page size:

* US Letter portrait, 8.5 x 11 inches

You can also choose:

* US Letter landscape
* Custom page size
* DPI
* whether to include the legend images

If you change export options after preparing a file, select **Prepare export
file** again before downloading.

## Troubleshooting

**I changed a parameter but the plot did not change.**
Select **Update plot**. Sidebar edits are intentionally not applied immediately.

**I changed LD source but the plot did not change.**
First select **Apply LD source**, then select **Update plot**.

**A variant is shown as a grey X.**
The variant was not found in the active LD reference or uploaded LD universe.

**Uploaded LD long table fails.**
Check that self-pair rows are included: `SNP_A == SNP_B`, `R2 == 1`.

**Export fails.**
Server-side export requires Plotly static image export dependencies. The same
Python environment that runs Streamlit needs `plotly`, `kaleido`, and `pillow`.
Kaleido v1 may also require server-side Chrome or Chromium.
"""
    )


def render_page_switcher():
    """Render a lightweight top-level page switcher."""
    return st.radio(
        "Page view",
        options=["Visualizer", "Documentation"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="page_view",
    )


def load_locusblend_page_icon():
    icon_path = ASSET_DIR / "WashU-SHIELD-Red_RGB.png"
    if icon_path.exists():
        try:
            from PIL import Image

            return Image.open(icon_path)
        except Exception:
            pass
    return "🧬"


st.set_page_config(
    page_title="LocusBlend",
    page_icon=load_locusblend_page_icon(),
    layout="wide",
    initial_sidebar_state="expanded",
)
set_streamlit_chrome_minimal()
inject_locusblend_css()
render_washu_header()
st.title("LocusBlend")
st.caption("Flexible multi-index regional visualization of genomic association signals")
page_view = render_page_switcher()

if page_view == "Documentation":
    render_documentation_page()
    st.stop()

render_locusblend_welcome_card()

st.session_state.setdefault("active_locusblend_mode", "Three-index LocusBlend")

_locusblend_modes = ["Standard locus zoom", "Two-index LocusBlend", "Three-index LocusBlend"]

if (
    "pending_locusblend_mode" not in st.session_state
    or st.session_state["pending_locusblend_mode"] not in _locusblend_modes
):
    st.session_state["pending_locusblend_mode"] = st.session_state["active_locusblend_mode"]

pending_mode = st.radio(
    "Mode",
    _locusblend_modes,
    horizontal=True,
    key="pending_locusblend_mode",
    label_visibility="collapsed",
)

if st.session_state["pending_locusblend_mode"] != st.session_state["active_locusblend_mode"]:
    st.info("Select **Update plot** to apply the selected mode.")

active_mode = st.session_state["active_locusblend_mode"]

# The old automatic welcome dialog was replaced by the non-modal
# welcome card and the top-level Documentation view.
progress_bar = None
status_box = None

try:
    render_sidebar_app_header()
    render_sidebar_visual_strip()

    with st.sidebar.expander("1. Data upload", expanded=False):
        st.markdown("Upload dataset 1 (top) and dataset 2 (bottom) summary-statistic files, or use the default example datasets.")
        st.caption("Full input file format details are in the top Documentation view.")
        uploaded_top = st.file_uploader(
            "Upload dataset 1 (top) summary-statistic file",
            type=["csv", "tsv", "txt", "gz"],
            key="uploaded_top",
        )
        st.caption(
            "Required: CHR, BP, P, A1, A2  -  Recommended: rsid, BETA, SE  -  "
            "Genome build: GRCh38 / hg38 only; no liftover is performed.  -  "
            "Supported: .csv, .tsv, .txt, .csv.gz, .tsv.gz, .txt.gz"
        )
        uploaded_bottom = st.file_uploader(
            "Upload dataset 2 (bottom) summary-statistic file",
            type=["csv", "tsv", "txt", "gz"],
            key="uploaded_bottom",
        )
        st.caption(
            "Required: CHR, BP, P, A1, A2  -  Recommended: rsid, BETA, SE  -  "
            "Genome build: GRCh38 / hg38 only; no liftover is performed.  -  "
            "Supported: .csv, .tsv, .txt, .csv.gz, .tsv.gz, .txt.gz"
        )

    with st.sidebar.expander("2. LD source", expanded=False):
        st.markdown("Choose the active LD source.")
        st.caption("Full LD format details are in the top Documentation view.")
        _ld_source_options = [
            "Use internal 1000G reference",
            "Upload LD matrix",
            "Upload LD long table",
        ]
        # Sticky/applied state — used by the downstream pipeline
        st.session_state.setdefault("active_ld_source", _ld_source_options[0])
        st.session_state.setdefault("active_ld_matrix_bytes", None)
        st.session_state.setdefault("active_ld_matrix_name", "")
        st.session_state.setdefault("active_ld_long_bytes", None)
        st.session_state.setdefault("active_ld_long_name", "")
        if "internal_1000g_ancestry" not in st.session_state:
            st.session_state["internal_1000g_ancestry"] = INTERNAL_1000G_DEFAULT_ANCESTRY
        else:
            normalized_ancestry = normalize_internal_1000g_ancestry(
                st.session_state.get("internal_1000g_ancestry")
            )
            if normalized_ancestry != st.session_state.get("internal_1000g_ancestry"):
                st.session_state["internal_1000g_ancestry"] = normalized_ancestry
        # Pending UI state — bound to the radio widget; only copies to active on Apply
        st.session_state.setdefault("pending_ld_source", st.session_state["active_ld_source"])

        pending_ld_source = st.radio(
            "LD source",
            _ld_source_options,
            index=_ld_source_options.index(st.session_state["pending_ld_source"])
                if st.session_state["pending_ld_source"] in _ld_source_options else 0,
            key="pending_ld_source",
            label_visibility="collapsed",
            captions=[
                "Embedded 1000G PLINK reference (default)",
                "Named R² matrix, SNP IDs as row and column names",
                "Long-format pairwise table with SNP_A, SNP_B, R2",
            ],
        )

        _matrix_widget = None
        _long_widget = None
        if pending_ld_source == "Use internal 1000G reference":
            internal_1000g_ancestry = st.selectbox(
                "1000G ancestry",
                options=INTERNAL_1000G_ANCESTRY_OPTIONS,
                index=INTERNAL_1000G_ANCESTRY_OPTIONS.index(
                    st.session_state["internal_1000g_ancestry"]
                ),
                key="internal_1000g_ancestry",
                format_func=format_internal_1000g_ancestry_option,
                help=(
                    "Select the 1000 Genomes super-population used for LD calculation. "
                    "Uploaded coordinates must be GRCh38 / hg38."
                ),
            )
        elif pending_ld_source == "Upload LD matrix":
            _matrix_widget = st.file_uploader(
                "Upload LD matrix",
                type=["tsv", "csv", "txt", "ld"],
                key="uploaded_ld_matrix",
            )
            st.caption("Square R² matrix with SNP IDs as row and column names.")
        elif pending_ld_source == "Upload LD long table":
            _long_widget = st.file_uploader(
                "Upload LD long table",
                type=["tsv", "csv", "txt", "ld"],
                key="uploaded_ld_long",
            )
            st.caption("Columns: SNP_A, SNP_B, R2. Include self-pairs with R2=1.")

        _apply_clicked = st.button("Apply LD source", key="apply_ld_source")
        if _apply_clicked:
            if pending_ld_source == "Upload LD matrix" and _matrix_widget is None:
                st.error(
                    "LD matrix file is required for 'Upload LD matrix'. "
                    "Please upload a file first, then select 'Apply LD source'."
                )
            elif pending_ld_source == "Upload LD long table" and _long_widget is None:
                st.error(
                    "LD long table file is required for 'Upload LD long table'. "
                    "Please upload a file first, then select 'Apply LD source'."
                )
            else:
                st.session_state["active_ld_source"] = pending_ld_source
                if pending_ld_source == "Upload LD matrix" and _matrix_widget is not None:
                    st.session_state["active_ld_matrix_bytes"] = _matrix_widget.getvalue()
                    st.session_state["active_ld_matrix_name"] = _matrix_widget.name
                if pending_ld_source == "Upload LD long table" and _long_widget is not None:
                    st.session_state["active_ld_long_bytes"] = _long_widget.getvalue()
                    st.session_state["active_ld_long_name"] = _long_widget.name
                if pending_ld_source == "Use internal 1000G reference":
                    _ancestry = normalize_internal_1000g_ancestry(st.session_state["internal_1000g_ancestry"])
                    st.success(f"LD source applied: Internal 1000G reference ({_ancestry})")
                else:
                    _applied_name = (
                        st.session_state["active_ld_matrix_name"]
                        if pending_ld_source == "Upload LD matrix"
                        else st.session_state["active_ld_long_name"]
                    )
                    st.success(f"LD source applied: {pending_ld_source} ({_applied_name})")

    ld_source = st.session_state["active_ld_source"]
    active_internal_1000g_ancestry = normalize_internal_1000g_ancestry(
        st.session_state.get("internal_1000g_ancestry", INTERNAL_1000G_DEFAULT_ANCESTRY)
    )
    if ld_source == "Use internal 1000G reference":
        st.sidebar.caption(f"Active LD source: Internal 1000G reference ({active_internal_1000g_ancestry})")
    else:
        st.sidebar.caption(f"Active LD source: {ld_source}")

    update_progress(progress_bar, status_box, 2, "Preparing app...")

    if uploaded_top is not None:
        update_progress(progress_bar, status_box, 5, f"Reading dataset 1 (top): {uploaded_top.name}")
        df_top = load_locus_csv_uploaded(uploaded_top.getvalue(), uploaded_top.name)
        top_source_name = uploaded_top.name
    else:
        update_progress(progress_bar, status_box, 5, "Reading default dataset 1 (top)...")
        df_top = load_locus_csv(str(DATA_DIR / "merged_female_withld_PSEN1.csv"))
        top_source_name = "merged_female_withld_PSEN1.csv"

    if uploaded_bottom is not None:
        update_progress(progress_bar, status_box, 10, f"Reading dataset 2 (bottom): {uploaded_bottom.name}")
        df_bottom = load_locus_csv_uploaded(uploaded_bottom.getvalue(), uploaded_bottom.name)
        bottom_source_name = uploaded_bottom.name
    else:
        update_progress(progress_bar, status_box, 10, "Reading default dataset 2 (bottom)...")
        df_bottom = load_locus_csv(str(DATA_DIR / "Nedelec_monocytes_PSEN1_example.csv"))
        bottom_source_name = "Nedelec_monocytes_PSEN1_example.csv"

    st.session_state.setdefault("chrom", "14")
    st.session_state.setdefault("index_selection_method", "Manual input")
    st.session_state.setdefault("bp", 73238768)
    st.session_state.setdefault("window_kb", 500)
    st.session_state.setdefault("rsid", "rs11159021")
    st.session_state.setdefault("rsid_2", "rs3742825")
    st.session_state.setdefault("rsid_3", "rs74986264")
    st.session_state.setdefault("title_top", "Alzheimer's disease Female GWAS")
    st.session_state.setdefault("title_bottom", "Monocyte PSEN1 eQTL")
    st.session_state.setdefault("show_recomb", True)
    st.session_state.setdefault("combined_height", 980)
    st.session_state.setdefault("vertical_spacing", 0.04)
    st.session_state.setdefault("recomb_max", 100)
    st.session_state.setdefault("gtf_path", str(DATA_DIR / "gencode.v49.annotation.chr14.gtf.gz"))
    st.session_state.setdefault("gene_display_mode", "protein_coding")
    st.session_state.setdefault("gene_track_gap", 30000)
    st.session_state.setdefault("highlight_genes", "")
    st.session_state.setdefault("compare_size", 560)
    st.session_state.setdefault("locuscompare_mode", "Three separate compare plots")
    st.session_state.setdefault(
        "ld_bfile_prefix",
        str(get_internal_1000g_prefix("14", INTERNAL_1000G_DEFAULT_ANCESTRY))
    )
    st.session_state.setdefault("plink_path", str(BIN_DIR / "plink"))

    upload_signature = make_uploaded_dataset_signature(uploaded_top, uploaded_bottom)
    if upload_signature is not None:
        upload_sync_result = infer_uploaded_locus_sync(
            df_top,
            df_bottom,
            uploaded_top,
            uploaded_bottom,
        )
        apply_uploaded_dataset_sync_if_new(
            upload_signature,
            upload_sync_result,
            df_top=df_top,
            df_bottom=df_bottom,
            uploaded_top=uploaded_top,
            uploaded_bottom=uploaded_bottom,
        )

    active_upload_sync_result = st.session_state.get("last_upload_sync_result")
    if (
        upload_signature is not None
        and st.session_state.get("last_uploaded_dataset_signature") == upload_signature
        and active_upload_sync_result
    ):
        _upload_sync_message = format_upload_sync_message(active_upload_sync_result)
        if _upload_sync_message:
            if active_upload_sync_result.get("status") == "warning":
                st.sidebar.warning(_upload_sync_message)
            else:
                st.sidebar.info(_upload_sync_message)

    current_bp = int(st.session_state["bp"])
    current_window_bp = int(st.session_state["window_kb"] * 1000)

    update_progress(progress_bar, status_box, 15, "Estimating default y-axis ranges...")

    _est_chrom = normalize_chrom(st.session_state.get("chrom", "14"))
    window_top = df_top[
        chrom_mask(df_top, _est_chrom)
        & (df_top["BP"] >= current_bp - current_window_bp)
        & (df_top["BP"] <= current_bp + current_window_bp)
    ].copy()
    if len(window_top) == 0:
        window_top = df_top.loc[chrom_mask(df_top, _est_chrom)].copy()
        if len(window_top) == 0:
            window_top = df_top.copy()
    default_max_ylim_top = int(np.ceil((-np.log10(window_top["P"])).max() + 1))

    window_bottom = df_bottom[
        chrom_mask(df_bottom, _est_chrom)
        & (df_bottom["BP"] >= current_bp - current_window_bp)
        & (df_bottom["BP"] <= current_bp + current_window_bp)
    ].copy()
    if len(window_bottom) == 0:
        window_bottom = df_bottom.loc[chrom_mask(df_bottom, _est_chrom)].copy()
        if len(window_bottom) == 0:
            window_bottom = df_bottom.copy()
    default_max_ylim_bottom = int(np.ceil((-np.log10(window_bottom["P"])).max() + 1))

    if "max_ylim_top" not in st.session_state:
        st.session_state["max_ylim_top"] = default_max_ylim_top
    if "max_ylim_bottom" not in st.session_state:
        st.session_state["max_ylim_bottom"] = default_max_ylim_bottom

    with st.sidebar.form("plot_controls"):
        submitted_top = False

        with st.expander("3. Locus window", expanded=False):
            _chrom_options = get_supported_chromosomes()
            _current_chrom = normalize_chrom(st.session_state.get("chrom", "14"))
            if not is_supported_chrom(_current_chrom):
                _current_chrom = "14"
            st.session_state["chrom"] = _current_chrom
            chrom = st.selectbox(
                "Chromosome",
                _chrom_options,
                index=_chrom_options.index(_current_chrom),
                key="chrom",
            )
            bp = st.number_input("Center position (bp)", value=st.session_state["bp"], step=1, key="bp")
            window_kb = st.number_input("Window size (+/- kb)", value=st.session_state["window_kb"], step=50, min_value=1, key="window_kb")
            window_bp = int(window_kb * 1000)

        with st.expander("4. Index variants", expanded=False):
            _index_method_aliases = {
                "Auto-select by LD clumping from " + "top " + "dataset": "Auto-select by LD clumping from dataset 1 (top)",
                "Auto-select by LD clumping from " + "bottom " + "dataset": "Auto-select by LD clumping from dataset 2 (bottom)",
            }
            if st.session_state.get("index_selection_method") in _index_method_aliases:
                st.session_state["index_selection_method"] = _index_method_aliases[
                    st.session_state["index_selection_method"]
                ]
            _index_methods = [
                "Manual input",
                "Auto-select by LD clumping from dataset 1 (top)",
                "Auto-select by LD clumping from dataset 2 (bottom)",
            ]
            _index_method_idx = _index_methods.index(st.session_state["index_selection_method"]) \
                if st.session_state["index_selection_method"] in _index_methods else 0
            index_selection_method = st.selectbox(
                "Index selection method",
                _index_methods,
                index=_index_method_idx,
                key="index_selection_method",
            )
            st.caption(
                "Manual: use the variant IDs below. "
                "Auto: the app selects the most significant independent variants "
                "within the locus window using LD clumping at r² = 0.01."
            )

            st.markdown("##### Manual index inputs")
            if index_selection_method != "Manual input":
                st.caption("Manual index fields are ignored when auto-selection is enabled.")
            rsid = st.text_input("Index variant 1", value=st.session_state["rsid"], key="rsid")
            _pending_mode = st.session_state.get("pending_locusblend_mode", "Three-index LocusBlend")
            if _pending_mode != "Standard locus zoom":
                rsid_2 = st.text_input("Index variant 2", value=st.session_state["rsid_2"], key="rsid_2")
            else:
                rsid_2 = ""
            if _pending_mode == "Three-index LocusBlend":
                rsid_3 = st.text_input("Index variant 3", value=st.session_state["rsid_3"], key="rsid_3")
            else:
                rsid_3 = ""

        with st.expander("5. Plot display", expanded=False):
            title_top = st.text_input("Dataset 1 (top) plot title", value=st.session_state["title_top"], key="title_top")
            title_bottom = st.text_input("Dataset 2 (bottom) plot title", value=st.session_state["title_bottom"], key="title_bottom")
            max_ylim_top = st.number_input("Y-axis max (top)", value=int(st.session_state["max_ylim_top"]), step=1, key="max_ylim_top")
            max_ylim_bottom = st.number_input("Y-axis max (bottom)", value=int(st.session_state["max_ylim_bottom"]), step=1, key="max_ylim_bottom")
            combined_height = st.number_input("Combined plot height", value=int(st.session_state["combined_height"]), step=40, key="combined_height")
            vertical_spacing = st.number_input("Vertical spacing", value=float(st.session_state["vertical_spacing"]), step=0.01, format="%.3f", key="vertical_spacing")
            _recomb_default = bool(st.session_state.get("show_recomb", True))
            _recomb_choice = st.selectbox(
                "Recombination rate",
                options=["Show", "Hide"],
                index=0 if _recomb_default else 1,
                key="show_recomb_choice",
                help="Show or hide the recombination-rate track on locus zoom plots.",
            )
            show_recomb = _recomb_choice == "Show"
            st.session_state["show_recomb"] = show_recomb
            recomb_max = st.number_input("Recombination Y max", value=int(st.session_state["recomb_max"]), step=10, key="recomb_max")

        with st.expander("6. Gene track", expanded=False):
            gene_display_mode = st.selectbox(
                "Gene display mode",
                ["protein_coding", "all"],
                index=["protein_coding", "all"].index(st.session_state["gene_display_mode"]),
                key="gene_display_mode"
            )
            gene_track_gap = st.number_input("Gene track min gap (bp)", value=int(st.session_state["gene_track_gap"]), step=5000, key="gene_track_gap")
            st.text_input("Highlight genes", value=st.session_state.get("highlight_genes", ""), placeholder="PSEN1, PAPLN, HEATR4", key="highlight_genes")
            st.caption("Comma or semicolon separated gene names (case-insensitive)")

        with st.expander("7. Locus compare", expanded=False):
            compare_size = st.number_input("Compare plot size", value=int(st.session_state["compare_size"]), step=20, key="compare_size")
            _locuscompare_options = ["Three separate compare plots", "Single blended compare plot"]
            locuscompare_mode = st.selectbox(
                "Locus compare mode",
                _locuscompare_options,
                index=_locuscompare_options.index(st.session_state["locuscompare_mode"])
                    if st.session_state["locuscompare_mode"] in _locuscompare_options else 0,
                key="locuscompare_mode"
            )
            st.caption("Three separate: one panel per index variant | Single blended: all variants overlaid in one plot")

        submitted_bottom = st.form_submit_button(
            "Update plot",
            use_container_width=True,
            type="primary",
        )

        submitted = submitted_top or submitted_bottom

    plink_path = st.session_state["plink_path"]
    ld_bfile_prefix = st.session_state["ld_bfile_prefix"]
    gtf_path = st.session_state["gtf_path"]

    st.caption(f"Dataset 1 (top) source: {top_source_name} | Dataset 2 (bottom) source: {bottom_source_name}")

    # Widget changes (radio, file uploader, number inputs) trigger Streamlit
    # reruns but should not trigger the LD pipeline or figure rebuild — those
    # only happen when the user selects the "Update plot" button, or on the
    # very first load (no cached figure yet).
    should_compute = submitted or "last_locus_fig" not in st.session_state

    if should_compute:
        st.session_state["active_locusblend_mode"] = st.session_state["pending_locusblend_mode"]
        progress_bar = st.progress(0)
        status_box = st.empty()
        update_progress(progress_bar, status_box, 5, "Preparing LD pipeline...")

        active_mode = st.session_state["active_locusblend_mode"]
        active_internal_1000g_ancestry = normalize_internal_1000g_ancestry(
            st.session_state.get("internal_1000g_ancestry", INTERNAL_1000G_DEFAULT_ANCESTRY)
        )
        ld_labels = get_ld_reference_labels(ld_source, active_internal_1000g_ancestry)

        selected_chrom = normalize_chrom(chrom)
        if not is_supported_chrom(selected_chrom):
            raise ValueError(
                f"Unsupported chromosome {chrom}. Supported chromosomes are 1-22 and X."
            )

        # Validate selected chromosome data presence
        _top_mask = chrom_mask(df_top, selected_chrom)
        _bottom_mask = chrom_mask(df_bottom, selected_chrom)
        if not _top_mask.any() and not _bottom_mask.any():
            raise ValueError(
                f"No rows found for chromosome {selected_chrom} in either dataset. "
                "Check the chromosome selector or upload data for this chromosome."
            )

        # Filter to selected chromosome for the LD pipeline
        df_top_chr = df_top.loc[_top_mask].copy()
        df_bottom_chr = df_bottom.loc[_bottom_mask].copy()

        # Resolve references dynamically
        if ld_source == "Use internal 1000G reference":
            ld_bfile_prefix = get_internal_bfile_prefix_for_chrom(selected_chrom, active_internal_1000g_ancestry)
        gtf_path = get_gtf_path_for_chrom(selected_chrom)

        chrom_for_ld = selected_chrom
        ld_bp_start = int(bp - window_bp)
        ld_bp_end = int(bp + window_bp)

        ld_long_table = None
        ld_matrix_table = None
        ld_uploaded_universe = None

        if ld_source == "Upload LD long table":
            _long_bytes = st.session_state.get("active_ld_long_bytes")
            _long_name = st.session_state.get("active_ld_long_name") or "ld_long"
            if _long_bytes is None:
                raise ValueError(
                    "Active LD source is 'Upload LD long table' but no LD file has been applied. "
                    "Upload a file and select 'Apply LD source' in the sidebar."
                )
            update_progress(progress_bar, status_box, 16, f"Parsing uploaded LD long table: {_long_name}")
            ld_long_table, ld_uploaded_universe = read_uploaded_ld_long(_long_bytes, _long_name)
        elif ld_source == "Upload LD matrix":
            _matrix_bytes = st.session_state.get("active_ld_matrix_bytes")
            _matrix_name = st.session_state.get("active_ld_matrix_name") or "ld_matrix"
            if _matrix_bytes is None:
                raise ValueError(
                    "Active LD source is 'Upload LD matrix' but no LD file has been applied. "
                    "Upload a file and select 'Apply LD source' in the sidebar."
                )
            update_progress(progress_bar, status_box, 16, f"Parsing uploaded LD matrix: {_matrix_name}")
            ld_matrix_table, ld_uploaded_universe = read_uploaded_ld_matrix(_matrix_bytes, _matrix_name)

        if ld_source == "Use internal 1000G reference":
            update_progress(progress_bar, status_box, 18, f"Computing LD from 1000G {active_internal_1000g_ancestry} with PLINK...")
            bim_ref = load_reference_bim(ld_bfile_prefix)
            df_top_ref = attach_reference_snp_two_pass(df_top_chr, bim_ref)
            df_bottom_ref = attach_reference_snp_two_pass(df_bottom_chr, bim_ref)
        else:
            update_progress(progress_bar, status_box, 18, "Matching summary stats against uploaded LD universe...")
            df_top_ref = attach_uploaded_ld_keys(df_top_chr, ld_uploaded_universe)
            df_bottom_ref = attach_uploaded_ld_keys(df_bottom_chr, ld_uploaded_universe)

        # Index variant resolution — manual or auto
        idx1_label_for_plot = rsid
        idx2_label_for_plot = rsid_2
        idx3_label_for_plot = rsid_3
        auto_index_table = None
        auto_index_summary = ""

        if index_selection_method == "Manual input":
            idx1_row = resolve_index_variant_from_input(df_top_ref, df_bottom_ref, rsid)
            idx1_ref = idx1_row["REF_SNP"]

            idx2_ref = None
            idx3_ref = None

            if active_mode != "Standard locus zoom" and str(rsid_2).strip() != "":
                idx2_row = resolve_index_variant_from_input(df_top_ref, df_bottom_ref, rsid_2)
                idx2_ref = idx2_row["REF_SNP"]

            if active_mode == "Three-index LocusBlend" and str(rsid_3).strip() != "":
                idx3_row = resolve_index_variant_from_input(df_top_ref, df_bottom_ref, rsid_3)
                idx3_ref = idx3_row["REF_SNP"]

        else:
            update_progress(progress_bar, status_box, 19, "Running auto index selection by LD clumping...")
            required_n = get_required_n_indices(active_mode)

            if index_selection_method == "Auto-select by LD clumping from dataset 1 (top)":
                _source_ref = df_top_ref
                _source_label = "dataset 1 (top)"
            else:
                _source_ref = df_bottom_ref
                _source_label = "dataset 2 (bottom)"

            auto_selected, auto_summary = auto_select_index_variants_by_clumping(
                df_source_ref=_source_ref,
                selected_chrom=selected_chrom,
                center_bp=bp,
                window_bp=window_bp,
                required_n=required_n,
                active_ld_source=ld_source,
                bfile_prefix=ld_bfile_prefix if ld_source == "Use internal 1000G reference" else None,
                plink_path=plink_path,
                ld_long_table=ld_long_table,
                ld_matrix_table=ld_matrix_table,
                clump_r2=0.01,
            )

            idx1_ref = auto_selected.iloc[0]["REF_SNP"]
            idx1_label_for_plot = str(auto_selected.iloc[0]["DISPLAY_ID"])
            idx2_ref = None
            idx3_ref = None
            idx2_label_for_plot = ""
            idx3_label_for_plot = ""

            if required_n >= 2 and len(auto_selected) >= 2:
                idx2_ref = auto_selected.iloc[1]["REF_SNP"]
                idx2_label_for_plot = str(auto_selected.iloc[1]["DISPLAY_ID"])
            if required_n >= 3 and len(auto_selected) >= 3:
                idx3_ref = auto_selected.iloc[2]["REF_SNP"]
                idx3_label_for_plot = str(auto_selected.iloc[2]["DISPLAY_ID"])

            auto_index_table = auto_selected.copy()
            auto_index_table["_source"] = _source_label
            auto_index_summary = (
                f"Auto-selected from the {_source_label} within "
                f"chr{selected_chrom}:{auto_summary['start_bp']}-{auto_summary['end_bp']} "
                f"using LD clumping at r² = {auto_summary['clump_r2']}. "
                f"({auto_summary['n_candidates']:,} candidates → "
                f"{auto_summary['n_selected']} independent variants)"
            )

        top_window = df_top_ref[
            chrom_mask(df_top_ref, selected_chrom)
            & (df_top_ref["BP"] >= ld_bp_start)
            & (df_top_ref["BP"] <= ld_bp_end)
        ][["REF_SNP", "CHR", "BP"]].copy()

        bottom_window = df_bottom_ref[
            chrom_mask(df_bottom_ref, selected_chrom)
            & (df_bottom_ref["BP"] >= ld_bp_start)
            & (df_bottom_ref["BP"] <= ld_bp_end)
        ][["REF_SNP", "CHR", "BP"]].copy()

        window_union = pd.concat([top_window, bottom_window], ignore_index=True).drop_duplicates("REF_SNP")
        window_snps = tuple(sorted(window_union["REF_SNP"].dropna().astype(str).unique().tolist()))

        if ld_source == "Use internal 1000G reference":
            ld_maps, index_status, ref_snps = compute_ld_maps_with_plink(
                bfile_prefix=ld_bfile_prefix,
                chrom=chrom_for_ld,
                start=ld_bp_start,
                end=ld_bp_end,
                window_snps=window_snps,
                idx1_ref=idx1_ref,
                idx2_ref=idx2_ref,
                idx3_ref=idx3_ref,
                plink_path=plink_path
            )
        elif ld_source == "Upload LD long table":
            ld_maps, index_status, ref_snps = compute_ld_maps_from_uploaded_long(
                ld_long=ld_long_table,
                window_snps=window_snps,
                idx1_ref=idx1_ref,
                idx2_ref=idx2_ref,
                idx3_ref=idx3_ref,
            )
        else:
            ld_maps, index_status, ref_snps = compute_ld_maps_from_uploaded_matrix(
                ld_matrix=ld_matrix_table,
                window_snps=window_snps,
                idx1_ref=idx1_ref,
                idx2_ref=idx2_ref,
                idx3_ref=idx3_ref,
            )

        ld_annot = build_ld_annot_for_window(
            window_union_df=window_union,
            ld_maps=ld_maps,
            ref_snps=ref_snps,
            idx1_ref=idx1_ref,
            idx2_ref=idx2_ref,
            idx3_ref=idx3_ref
        )

        df_top_plot = merge_ld_annot(df_top_ref, ld_annot)
        df_bottom_plot = merge_ld_annot(df_bottom_ref, ld_annot)

        _src = ld_labels["source_name"]
        _status_parts = [
            f"variant 1: {('found in ' + _src) if index_status['variant 1'] else ('not in ' + _src + ' -> grey X')}",
        ]
        if active_mode != "Standard locus zoom":
            _status_parts.append(
                f"variant 2: {('found in ' + _src) if index_status['variant 2'] else ('not in ' + _src + ' -> grey X')}",
            )
        if active_mode == "Three-index LocusBlend":
            _status_parts.append(
                f"variant 3: {('found in ' + _src) if index_status['variant 3'] else ('not in ' + _src + ' -> grey X')}",
            )
        ld_status_caption = " | ".join(_status_parts)

        _ref_info = ""
        if ld_source == "Use internal 1000G reference":
            _ref_info = f"{active_internal_1000g_ancestry} {format_chrom_label(selected_chrom)}"
        ld_status_caption += f"  |  Chromosome: {format_chrom_label(selected_chrom)}"
        if _ref_info:
            ld_status_caption += f"  |  Ref: 1000G {_ref_info}"

        update_progress(progress_bar, status_box, 25, "Building top locus plot...")
        fig_top, n_top, bp_start_top, bp_end_top = get_plotly_locus_py(
            max_ylim=max_ylim_top,
            bp=bp,
            window_bp=window_bp,
            merged_female_withld=df_top_plot,
            merged_df=df_top_plot,
            idx1_ref=idx1_ref,
            idx2_ref=idx2_ref,
            idx3_ref=idx3_ref,
            idx1_label=idx1_label_for_plot,
            idx2_label=idx2_label_for_plot,
            idx3_label=idx3_label_for_plot,
            y_label=title_top,
            ld_labels=ld_labels,
            chrom=selected_chrom,
            show_recomb=show_recomb,
            bw_path=str(DATA_DIR / "recomb1000GAvg.bw"),
            locusblend_mode=active_mode,
        )

        update_progress(progress_bar, status_box, 40, "Building bottom locus plot...")
        fig_bottom, n_bottom, bp_start_bottom, bp_end_bottom = get_plotly_locus_py(
            max_ylim=max_ylim_bottom,
            bp=bp,
            window_bp=window_bp,
            merged_female_withld=df_bottom_plot,
            merged_df=df_bottom_plot,
            idx1_ref=idx1_ref,
            idx2_ref=idx2_ref,
            idx3_ref=idx3_ref,
            idx1_label=idx1_label_for_plot,
            idx2_label=idx2_label_for_plot,
            idx3_label=idx3_label_for_plot,
            y_label=title_bottom,
            ld_labels=ld_labels,
            chrom=selected_chrom,
            show_recomb=show_recomb,
            bw_path=str(DATA_DIR / "recomb1000GAvg.bw"),
            locusblend_mode=active_mode,
        )

        update_progress(progress_bar, status_box, 55, f"Loading gene track ({gene_display_mode})...")
        gene_bp_start = min(bp_start_top, bp_start_bottom)
        gene_bp_end = max(bp_end_top, bp_end_bottom)

        if gtf_path.endswith(".parquet"):
            genes_df = load_genes_from_table(
                chrom=selected_chrom,
                start=gene_bp_start,
                end=gene_bp_end,
                parquet_path=gtf_path,
                gene_display_mode=gene_display_mode
            )
        else:
            genes_df = load_genes_from_gtf(
                gtf_path=gtf_path,
                chrom=selected_chrom,
                start=gene_bp_start,
                end=gene_bp_end,
                gene_display_mode=gene_display_mode
            )

        update_progress(progress_bar, status_box, 70, "Assembling shared locus figure...")
        locus_fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=vertical_spacing,
            row_heights=[0.36, 0.36, 0.28],
            specs=[
                [{"secondary_y": True}],
                [{"secondary_y": True}],
                [{"secondary_y": False}]
            ],
            subplot_titles=(title_top, title_bottom, f"GENCODE gene track ({gene_display_mode})")
        )

        for tr in fig_top.data:
            tr_json = tr.to_plotly_json()
            tr_json.pop("xaxis", None)
            tr_json.pop("yaxis", None)
            if tr_json.get("type") == "scattergl":
                tr2 = go.Scattergl(**tr_json)
            else:
                tr2 = go.Scatter(**tr_json)
            is_secondary = getattr(tr, "mode", None) == "lines"
            locus_fig.add_trace(tr2, row=1, col=1, secondary_y=is_secondary)

        for tr in fig_bottom.data:
            tr_json = tr.to_plotly_json()
            tr_json.pop("xaxis", None)
            tr_json.pop("yaxis", None)
            if tr_json.get("type") == "scattergl":
                tr2 = go.Scattergl(**tr_json)
            else:
                tr2 = go.Scatter(**tr_json)
            is_secondary = getattr(tr, "mode", None) == "lines"
            locus_fig.add_trace(tr2, row=2, col=1, secondary_y=is_secondary)

        _raw_highlight = st.session_state.get("highlight_genes", "")
        highlight_names = {g.strip().lower() for g in re.split(r"[,;\s]+", _raw_highlight) if g.strip()}

        locus_fig, gene_track_rows = add_gene_track_to_subplot(
            locus_fig,
            genes_df,
            row=3,
            col=1,
            min_gap=gene_track_gap,
            highlight_names=highlight_names
        )

        locus_fig.update_yaxes(
            title_text="-log<sub>10</sub>(P)",
            range=list(fig_top.layout.yaxis.range),
            zeroline=False,
            row=1,
            col=1,
            secondary_y=False
        )
        locus_fig.update_yaxes(
            title_text="Recombination rate",
            range=[0, recomb_max],
            autorange=False,
            zeroline=False,
            showgrid=False,
            row=1,
            col=1,
            secondary_y=True
        )

        locus_fig.update_yaxes(
            title_text="-log<sub>10</sub>(P)",
            range=list(fig_bottom.layout.yaxis.range),
            zeroline=False,
            row=2,
            col=1,
            secondary_y=False
        )
        locus_fig.update_yaxes(
            title_text="Recombination rate",
            range=[0, recomb_max],
            autorange=False,
            zeroline=False,
            showgrid=False,
            row=2,
            col=1,
            secondary_y=True
        )

        locus_fig.update_yaxes(
            title_text="Genes",
            range=[-gene_track_rows + 0.5, 0.8],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            row=3,
            col=1
        )

        locus_fig.update_xaxes(
            range=[gene_bp_start / 1e6, gene_bp_end / 1e6],
            showticklabels=False,
            zeroline=False,
            row=1,
            col=1
        )
        locus_fig.update_xaxes(
            range=[gene_bp_start / 1e6, gene_bp_end / 1e6],
            showticklabels=False,
            zeroline=False,
            row=2,
            col=1
        )
        locus_fig.update_xaxes(
            range=[gene_bp_start / 1e6, gene_bp_end / 1e6],
            title_text=format_chrom_axis_title(selected_chrom),
            zeroline=False,
            row=3,
            col=1
        )

        if hasattr(locus_fig.layout, "xaxis2"):
            locus_fig.layout.xaxis2.matches = "x"
        if hasattr(locus_fig.layout, "xaxis3"):
            locus_fig.layout.xaxis3.matches = "x"

        if hasattr(locus_fig.layout, "yaxis2"):
            locus_fig.layout.yaxis2.update(
                range=[0, recomb_max],
                autorange=False,
                tickmode="linear",
                dtick=20
            )
        if hasattr(locus_fig.layout, "yaxis4"):
            locus_fig.layout.yaxis4.update(
                range=[0, recomb_max],
                autorange=False,
                tickmode="linear",
                dtick=20
            )

        locus_fig.update_layout(
            height=combined_height,
            dragmode="zoom",
            showlegend=False,
            margin=dict(l=60, r=60, b=45, t=60),
        )
        locus_fig = apply_locusblend_plot_theme(locus_fig)

        _active_compare_signals = ["variant 1"]
        if active_mode != "Standard locus zoom":
            _active_compare_signals.append("variant 2")
        if active_mode == "Three-index LocusBlend":
            _active_compare_signals.append("variant 3")

        if locuscompare_mode == "Single blended compare plot":
            update_progress(progress_bar, status_box, 85, "Building single blended compare plot...")
            compare_fig, compare_blended_count = build_single_blended_locuscompare(
                df_top=df_top_plot,
                df_bottom=df_bottom_plot,
                idx1_ref=idx1_ref,
                idx2_ref=idx2_ref,
                idx3_ref=idx3_ref,
                idx1_label=idx1_label_for_plot,
                idx2_label=idx2_label_for_plot,
                idx3_label=idx3_label_for_plot,
                title_top=title_top,
                title_bottom=title_bottom,
                compare_size=compare_size,
                ld_labels=ld_labels,
                locusblend_mode=active_mode,
            )
            compare_counts = None
        else:
            update_progress(progress_bar, status_box, 85, "Building compare plots...")
            compare_fig, compare_counts = build_compare_figure_triptych(
                df_top=df_top_plot,
                df_bottom=df_bottom_plot,
                idx1_label=idx1_label_for_plot,
                idx2_label=idx2_label_for_plot,
                idx3_label=idx3_label_for_plot,
                title_top=title_top,
                title_bottom=title_bottom,
                compare_size=compare_size,
                ld_labels=ld_labels,
                active_signals=_active_compare_signals,
            )
            compare_blended_count = None

        compare_fig = apply_locusblend_plot_theme(compare_fig)
        compare_fig = apply_locus_compare_safe_autoscale(compare_fig)

        update_progress(progress_bar, status_box, 95, "Rendering figures...")

        n_flip_top = int(df_top_plot["coding_flipped"].fillna(False).sum()) if "coding_flipped" in df_top_plot.columns else 0
        n_flip_bottom = int(df_bottom_plot["coding_flipped"].fillna(False).sum()) if "coding_flipped" in df_bottom_plot.columns else 0

        if locuscompare_mode == "Single blended compare plot":
            compare_summary = f"Compare points (blended): {compare_blended_count:,}"
        else:
            _compare_parts = [f"v1: {compare_counts.get('variant 1', 0):,}"]
            if active_mode != "Standard locus zoom":
                _compare_parts.append(f"v2: {compare_counts.get('variant 2', 0):,}")
            if active_mode == "Three-index LocusBlend":
                _compare_parts.append(f"v3: {compare_counts.get('variant 3', 0):,}")
            compare_summary = f"Compare points ({'/'.join(_compare_parts)})"

        summary_text = (
            f"Dataset 1 SNP count: {n_top:,} | "
            f"Dataset 2 SNP count: {n_bottom:,} | "
            f"Genes shown ({gene_display_mode}): {len(genes_df):,} | "
            f"{compare_summary} | "
            f"{ld_labels['summary_label']}: {len(ref_snps):,} | "
            f"Coding flipped (dataset 1/dataset 2): {n_flip_top:,}/{n_flip_bottom:,}"
        )

        for _export_key in [
            "last_export_bytes",
            "last_export_filename",
            "last_export_mime",
            "last_export_signature",
            "last_export_error",
        ]:
            st.session_state.pop(_export_key, None)

        st.session_state["last_export_context"] = {
            "mode": active_mode,
            "compare_mode": locuscompare_mode,
            "chromosome": selected_chrom,
            "center_bp": int(bp),
            "window_kb": int(window_kb),
            "title_top": str(title_top),
            "title_bottom": str(title_bottom),
            "highlight_genes": str(st.session_state.get("highlight_genes", "")),
            "ld_status_caption": str(ld_status_caption),
        }

        st.session_state["last_locus_fig"] = locus_fig
        st.session_state["last_compare_fig"] = compare_fig
        st.session_state["last_compare_mode"] = locuscompare_mode
        st.session_state["last_compare_size"] = int(compare_size)
        st.session_state["last_summary_text"] = summary_text
        st.session_state["last_summary_metrics"] = {
            "n_top": n_top,
            "n_bottom": n_bottom,
            "n_genes": len(genes_df),
            "gene_mode": gene_display_mode,
            "compare_summary": compare_summary,
            "n_ref_snps": len(ref_snps),
            "summary_label": ld_labels["summary_label"],
            "n_flip_top": n_flip_top,
            "n_flip_bottom": n_flip_bottom,
        }
        if index_selection_method != "Manual input":
            st.session_state["last_auto_index_table"] = auto_index_table
            st.session_state["last_auto_index_caption"] = auto_index_summary
        else:
            st.session_state["last_auto_index_table"] = None
            st.session_state["last_auto_index_caption"] = ""
        st.session_state["last_index_selection_method"] = index_selection_method
        st.session_state["last_ld_status_caption"] = ld_status_caption
        st.session_state["last_progress_success"] = "Plots updated successfully."
        st.session_state["has_rendered_once"] = True

        update_progress(progress_bar, status_box, 100, "Done.")
        status_box.success(st.session_state["last_progress_success"])

    # Display block — always runs, reads from sticky session_state. On
    # widget-only reruns (radio, uploader, params edits) this redraws the
    # previously cached figures without re-running any heavy pipeline work.
    st.caption(st.session_state["last_ld_status_caption"])

    config = {
        "toImageButtonOptions": {
            "format": "png",  # or "svg"
            "filename": "locus_plot",
            "width": 1800,
            "height": 1200,
            "scale": 3,
        }
    }

    st.plotly_chart(st.session_state["last_locus_fig"], use_container_width=True, config=config)

    st.markdown("### Locus compare")
    _last_compare_mode = st.session_state["last_compare_mode"]
    _last_compare_size = int(st.session_state["last_compare_size"])
    _compare_fig = st.session_state["last_compare_fig"]
    compare_config = get_locus_compare_plotly_config(config)

    if _last_compare_mode == "Single blended compare plot":
        _plot_size = _last_compare_size
        _compare_fig_display = clone_plotly_figure(_compare_fig)
        _compare_fig_display.update_layout(
            width=_plot_size,
            height=_plot_size,
            autosize=False,
            margin=dict(l=70, r=40, t=50, b=70),
        )
        blended_config = {
            "responsive": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "locus_compare_blended",
                "width": _plot_size,
                "height": _plot_size,
                "scale": 3,
            },
        }
        blended_config = get_locus_compare_plotly_config(blended_config)
        _left, _center, _right = st.columns([1, 2, 1])
        with _center:
            st.plotly_chart(
                _compare_fig_display,
                use_container_width=False,
                config=blended_config,
            )
    else:
        st.plotly_chart(_compare_fig, use_container_width=True, config=compare_config)

    # Auto index table
    if st.session_state.get("last_auto_index_table") is not None:
        _auto_tbl = st.session_state["last_auto_index_table"]
        _auto_cap = st.session_state.get("last_auto_index_caption", "")
        if _auto_cap:
            st.caption(_auto_cap)
        st.dataframe(
            _auto_tbl.rename(columns={
                "rank": "Rank",
                "DISPLAY_ID": "Variant",
                "REF_SNP": "REF_SNP",
                "CHR": "CHR",
                "BP": "BP",
                "P": "P",
                "_source": "Source",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # Summary display
    if "last_summary_metrics" in st.session_state:
        m = st.session_state["last_summary_metrics"]
        _index_method = st.session_state.get("last_index_selection_method", "Manual input")
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**Dataset 1 SNPs**<br><span style='font-size:1.3em'>{m['n_top']:,}</span>", unsafe_allow_html=True)
            c2.markdown(f"**Dataset 2 SNPs**<br><span style='font-size:1.3em'>{m['n_bottom']:,}</span>", unsafe_allow_html=True)
            c3.markdown(f"**Genes** ({m['gene_mode']})<br><span style='font-size:1.3em'>{m['n_genes']:,}</span>", unsafe_allow_html=True)
            c4.markdown(f"**{m['summary_label']}**<br><span style='font-size:1.3em'>{m['n_ref_snps']:,}</span>", unsafe_allow_html=True)
            st.caption(
                f"Index selection: {_index_method}  |  "
                f"{m['compare_summary']}  |  "
                f"Coding flipped (dataset 1/dataset 2): {m['n_flip_top']:,}/{m['n_flip_bottom']:,}"
            )
    elif "last_summary_text" in st.session_state:
        st.write(st.session_state["last_summary_text"])

    render_export_controls()

    if not should_compute:
        st.info("Using previously rendered plots. Select Update plot to apply new settings.")

except Exception as e:
    _progress_bar = locals().get("progress_bar", None)
    _status_box = locals().get("status_box", None)

    if _progress_bar is not None:
        try:
            _progress_bar.empty()
        except Exception:
            pass

    if _status_box is not None:
        try:
            _status_box.empty()
        except Exception:
            pass

    log(traceback.format_exc())
    log(f"APP FAILED: {repr(e)}")
    st.exception(e)
