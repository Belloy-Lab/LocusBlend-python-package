"""LD reference handling: PLINK bfiles, PLINK LD, and uploaded LD tables.

Extracted from ``app.2.8.12.py``: PLINK executable discovery, PLINK bfile
prefix handling, BIM loading, PLINK ``--r2`` LD calculation, uploaded LD
long-table/matrix parsing, uploaded-LD key matching, and LD map/annotation
construction. Thresholds and behavior are unchanged.

Changes required to remove the app coupling (documented, no algorithm change):

* Streamlit cache decorators are dropped; callers that still want caching can
  wrap these functions themselves.
* ``_resolve_bfile_prefix`` / ``load_reference_bim`` no longer search the app's
  ``data/`` and script directories implicitly; search directories are an
  explicit parameter (a prefix's own parent directory is always included).
* ``_find_plink_exec`` / ``compute_ld_maps_with_plink`` no longer default to a
  bundled ``bin/plink``: PLINK comes from an explicit path, the
  ``LOCUSBLEND_PLINK`` environment variable, or the system ``PATH``. PLINK is
  not bundled with this package.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .io import dedup_columns, log, normalize_chrom
from .reference import PLINK_ENV_VAR

# --- extracted from app.2.8.12.py (verbatim) ---


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


def _normalize_bfile_prefix(p):
    p = str(p).strip()
    for suf in [".bed", ".bim", ".fam"]:
        if p.endswith(suf):
            p = p[:-len(suf)]
    return p


def _resolve_bfile_prefix(bfile_prefix, search_dirs=None):
    """Return a PLINK bfile prefix for which .bed/.bim/.fam all exist.

    app.2.8.12.py implicitly searched the app's ``data/`` and script
    directories. Here the search directories are explicit; the parent
    directory of a supplied prefix is always included. No other behavior
    (candidate order, deduplication, error message) changed.
    """
    raw = str(bfile_prefix).strip()
    prefix = _normalize_bfile_prefix(raw)

    dirs = []
    for d in (search_dirs or ()):
        p = Path(d)
        if p not in dirs:
            dirs.append(p)
    if prefix:
        parent = Path(prefix).parent
        if str(parent) not in ("", ".") and parent not in dirs:
            dirs.append(parent)

    candidates = []
    if prefix:
        candidates.append(prefix)
        for d in dirs:
            candidates.append(str(d / os.path.basename(prefix)))

    for d in dirs:
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
    for d in dirs:
        debug_files.extend(sorted(glob.glob(str(d / "*"))))
    raise FileNotFoundError(
        "Could not resolve a valid PLINK bfile prefix. "
        f"Input was: {raw!r}. "
        f"Searched directories: {[str(d) for d in dirs]}. "
        f"Available files include: {[os.path.basename(x) for x in debug_files[:50]]}"
    )


def _find_plink_exec(plink_path=None):
    """Return a usable PLINK executable.

    app.2.8.12.py defaulted to a ``bin/plink`` bundled next to the script.
    PLINK is not bundled with this package: an explicit path,
    ``LOCUSBLEND_PLINK``, or the system PATH is used instead.
    """
    candidates = []
    if plink_path:
        candidates.append(str(plink_path).strip())
    env_plink = os.environ.get(PLINK_ENV_VAR, "").strip()
    if env_plink:
        candidates.append(env_plink)
    candidates.extend(["./plink", "plink"])
    for c in candidates:
        if c and (os.path.exists(c) or shutil.which(c)):
            return c
    raise FileNotFoundError(
        "PLINK executable not found. LocusBlend does not bundle PLINK. "
        "Install PLINK and either put it on your PATH, set the "
        f"{PLINK_ENV_VAR} environment variable to its full path, or pass "
        "plink_path='/full/path/to/plink'. PLINK is required for LD computation "
        "and for automatic index-variant selection against the internal 1000G "
        "reference."
    )


def load_reference_bim(bfile_prefix, search_dirs=None):
    bfile_prefix = _resolve_bfile_prefix(bfile_prefix, search_dirs=search_dirs)
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


def compute_ld_maps_with_plink(
    bfile_prefix,
    chrom,
    start,
    end,
    window_snps,
    idx1_ref,
    idx2_ref,
    idx3_ref,
    plink_path=None
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
