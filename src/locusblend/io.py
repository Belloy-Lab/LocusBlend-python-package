"""Input parsing and normalization helpers (Streamlit-independent).

Extracted from ``app.2.8.12.py``:

* ``dedup_columns`` and ``_clean_locus_df`` (column aliases and value cleanup)
* uploaded summary-statistic format detection and parsing
  (``.csv``, ``.tsv``, ``.txt``, ``.csv.gz``, ``.tsv.gz``, ``.txt.gz``)
* chromosome normalization / masking / sorting / labeling helpers
* locus dataset summaries and the y-axis recommendation helper
* upload-driven chromosome/BP sync inference (pure part of the pipeline)

Behavior is intentionally identical to the baseline. The only changes are
mechanical: the Streamlit cache decorators are dropped, and the shared ``log``
helper lives here because several extracted modules use it.
"""

from __future__ import annotations

import hashlib
import os
import time
from io import BytesIO

import numpy as np
import pandas as pd

# --- extracted from app.2.8.12.py (verbatim) ---


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dedup_columns(df):
    return df.loc[:, ~pd.Index(df.columns).duplicated()].copy()


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


def load_locus_csv(path):
    log(f"loading csv from path: {path}")
    df = pd.read_csv(path)
    print("RAW COLUMNS:", [repr(c) for c in df.columns], flush=True)
    df = _clean_locus_df(df, source_name=path)
    log(f"{path} loaded, shape={df.shape}")
    return df


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


def format_upload_sync_message(sync_result):
    if not sync_result:
        return ""
    return str(sync_result.get("message", ""))


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
