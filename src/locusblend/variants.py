"""Variant/reference matching and index-variant selection.

Extracted from ``app.2.8.12.py``: BIM UID construction, two-pass allele
matching (forward / flipped), index-SNP resolution from user input, clumping
candidate preparation, PLINK clumping, greedy clumping for uploaded LD, and
automatic index-variant selection. Algorithms are unchanged.

Streamlit coupling removed:

* ``auto_select_index_variants_by_clumping`` used to read the active mode from
  Streamlit session state for its error message; the mode label is now an
  explicit parameter (``active_locusblend_mode``) with the baseline's fallback
  value.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import pandas as pd

from .io import chrom_mask, dedup_columns, log, normalize_chrom
from .ld import _find_plink_exec, _resolve_bfile_prefix

# --- extracted from app.2.8.12.py (verbatim) ---


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
    active_locusblend_mode=None,
):
    """Dispatch to PLINK or greedy clumping, returning (selected_rows, summary_dict).

    ``active_locusblend_mode`` replaces the baseline's read of the Streamlit
    session state (key ``active_locusblend_mode``, fallback ``"selected"``) in
    the "not enough independent index variants" error message. It is only used
    for that message; no selection behavior depends on it.
    """
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
            f"The '{active_locusblend_mode or 'selected'}' mode "
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
