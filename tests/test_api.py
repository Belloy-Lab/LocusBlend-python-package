"""Public API orchestration tests.

No real 1000G/GENCODE files and no PLINK binary are available in this
repository, so the reference lookups and the PLINK/gene steps are replaced with
small fakes. Everything else (format detection, column normalization, locus
inference, two-pass reference matching, LD annotation merging, figure assembly,
result population) runs for real on synthetic data.
"""

import gzip
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import locusblend
from locusblend import api
from locusblend.config import (
    MODE_STANDARD,
    MODE_THREE_INDEX,
    MODE_TWO_INDEX,
    normalize_auto_index_source,
    normalize_public_mode,
)
from locusblend.ld import _find_plink_exec
from locusblend.models import IndexVariant, LocusBlendResult
from locusblend.reference import ReferenceManager

CENTER_BP = 73_238_768
N_SNPS = 60
COMPLEMENT = {"A": "G", "C": "T", "G": "A", "T": "C"}


# ----------------------------------------------------------------------
# synthetic inputs (both datasets share one CHR/BP/allele backbone so a single
# fake BIM can reference-match them, exactly like a real 1000G panel)
# ----------------------------------------------------------------------
def make_backbone(n=N_SNPS):
    rng = np.random.default_rng(42)
    bp = np.sort(rng.integers(CENTER_BP - 200_000, CENTER_BP + 200_000, n))
    bp[n // 2] = CENTER_BP
    a1 = rng.choice(list("ACGT"), n)
    a2 = np.array([COMPLEMENT[base] for base in a1])
    return pd.DataFrame({"CHR": ["14"] * n, "BP": bp, "A1": a1, "A2": a2})


def make_bim():
    """BIM-like frame; SNP ids follow dataset 1's rsid naming."""
    bim = make_backbone()
    bim.insert(1, "SNP", [f"rsone{i:03d}" for i in range(len(bim))])
    bim.insert(2, "CM", 0.0)
    return bim[["CHR", "SNP", "CM", "BP", "A1", "A2"]]


def make_dataset(tag="one", chrom="14", n=N_SNPS):
    """Synthetic summary statistics with a known best-P row at ``CENTER_BP``."""
    rng = np.random.default_rng(0 if tag == "one" else 1)
    df = make_backbone(n)
    if chrom != "14":
        df["CHR"] = chrom
    df["P"] = 10 ** (-rng.uniform(0.5, 6.0, n))
    df.loc[df.index[n // 2], "P"] = 1e-12
    df["rsid"] = [f"rs{tag}{i:03d}" for i in range(n)]
    df["BETA"] = rng.normal(0, 0.1, n)
    df["SE"] = rng.uniform(0.01, 0.05, n)
    df["dataset_tag"] = tag
    return df[["CHR", "BP", "P", "A1", "A2", "rsid", "BETA", "SE", "dataset_tag"]]


def make_genes():
    return pd.DataFrame(
        {
            "seqname": ["chr14", "chr14"],
            "start": [CENTER_BP - 5_000, CENTER_BP + 5_000],
            "end": [CENTER_BP - 1_000, CENTER_BP + 9_000],
            "strand": ["+", "-"],
            "gene_id": ["ENSG1", "ENSG2"],
            "gene_name": ["PSEN1", "PAPLN"],
            "gene_type": ["protein_coding", "protein_coding"],
        }
    )


# ----------------------------------------------------------------------
# fake reference / PLINK / gene layer
#
# The reference *layout* is real: a small fake reference tree is created at the
# paths ReferenceManager resolves, so path resolution, ancestry handling and
# validation all run for real. Only the heavy steps (BIM read, PLINK LD,
# GENCODE parse, optionally the PLINK lookup and clumping) are replaced.
# ----------------------------------------------------------------------
def patch_reference_layer(
    monkeypatch,
    tmp_path,
    patch_auto_select=True,
    patch_plink=True,
    patch_manager=True,
    ancestry="EUR",
    chrom="14",
):
    """Patch the heavy pipeline steps; return recorded call information."""
    calls = SimpleNamespace(
        compute_ld=[],
        auto_select=[],
        genes=[],
        load_reference_bim=[],
        reference_dir=tmp_path,
        ancestry=ancestry,
        bfile_prefix=None,
        gtf_path=None,
    )

    # real reference layout: 1000g/<ancestry>/, gencode/, recombination/
    reference = ReferenceManager(reference_dir=tmp_path, ancestry=ancestry)
    prefix = reference.bfile_prefix(chrom)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".bed", ".bim", ".fam"):
        Path(str(prefix) + suffix).write_text("", encoding="utf-8")
    gtf_path = reference.gtf_path(chrom)
    gtf_path.parent.mkdir(parents=True, exist_ok=True)
    gtf_path.write_text("", encoding="utf-8")
    calls.bfile_prefix = str(prefix)
    calls.gtf_path = str(gtf_path)

    # the public API resolves its reference manager against this fake tree
    # (``patch_manager=False`` keeps the real resolution so tests can exercise
    # the reference_dir / environment-variable priority)
    if patch_manager:
        monkeypatch.setattr(
            api,
            "ReferenceManager",
            lambda reference_dir=None, ancestry="EUR", **kwargs: ReferenceManager(
                reference_dir=tmp_path, ancestry=ancestry
            ),
        )

    def fake_load_reference_bim(bfile_prefix, search_dirs=None):
        calls.load_reference_bim.append(bfile_prefix)
        return make_bim()

    monkeypatch.setattr(api, "load_reference_bim", fake_load_reference_bim)

    def fake_compute_ld_maps_with_plink(
        bfile_prefix, chrom, start, end, window_snps, idx1_ref, idx2_ref, idx3_ref, plink_path=None
    ):
        calls.compute_ld.append(
            {
                "bfile_prefix": bfile_prefix,
                "chrom": chrom,
                "start": start,
                "end": end,
                "window_snps": tuple(window_snps),
                "idx1_ref": idx1_ref,
                "idx2_ref": idx2_ref,
                "idx3_ref": idx3_ref,
                "plink_path": plink_path,
            }
        )
        window = list(window_snps)
        ld_maps = {"r2_1": {}, "r2_2": {}, "r2_3": {}}
        for column, index_ref in (("r2_1", idx1_ref), ("r2_2", idx2_ref), ("r2_3", idx3_ref)):
            if index_ref is None or pd.isna(index_ref):
                continue
            ld_maps[column] = {str(snp): 0.8 for snp in window[:5]}
            ld_maps[column][str(index_ref)] = 1.0
        ref_snps = tuple(sorted(set(window)))
        index_status = {
            "variant 1": _in_window(idx1_ref, ref_snps),
            "variant 2": _in_window(idx2_ref, ref_snps),
            "variant 3": _in_window(idx3_ref, ref_snps),
        }
        return ld_maps, index_status, ref_snps

    monkeypatch.setattr(api, "compute_ld_maps_with_plink", fake_compute_ld_maps_with_plink)

    if patch_plink:
        monkeypatch.setattr(api, "_find_plink_exec", lambda plink_path=None: "plink-fake")

    if patch_auto_select:

        def fake_auto_select(**kwargs):
            calls.auto_select.append(kwargs)
            source = kwargs["df_source_ref"]
            head = source.head(kwargs["required_n"])
            table = pd.DataFrame(
                {
                    "rank": range(1, len(head) + 1),
                    "DISPLAY_ID": head["rsid"].astype(str).tolist(),
                    "REF_SNP": head["REF_SNP"].tolist(),
                    "CHR": head["CHR"].tolist(),
                    "BP": head["BP"].tolist(),
                    "P": head["P"].tolist(),
                }
            )
            summary = {
                "n_candidates": len(source),
                "n_selected": len(table),
                "clump_r2": kwargs["clump_r2"],
                "selected_chrom": kwargs["selected_chrom"],
                "start_bp": int(kwargs["center_bp"] - kwargs["window_bp"]),
                "end_bp": int(kwargs["center_bp"] + kwargs["window_bp"]),
            }
            return table, summary

        monkeypatch.setattr(api, "auto_select_index_variants_by_clumping", fake_auto_select)

    def fake_load_genes_from_gtf(
        gtf_path, chrom, start, end, gene_display_mode="protein_coding", chunksize=200000
    ):
        calls.genes.append(
            {
                "gtf_path": gtf_path,
                "chrom": chrom,
                "start": start,
                "end": end,
                "gene_display_mode": gene_display_mode,
            }
        )
        genes = make_genes()
        if gene_display_mode == "protein_coding":
            genes = genes[genes["gene_type"] == "protein_coding"].copy()
        return genes

    monkeypatch.setattr(api, "load_genes_from_gtf", fake_load_genes_from_gtf)

    return calls


def _in_window(index_ref, ref_snps):
    if index_ref is None or pd.isna(index_ref):
        return False
    return str(index_ref) in set(ref_snps)


def _trace_texts(trace):
    """Return a trace's hover texts as a list of strings (may be None/array)."""
    text = getattr(trace, "text", None)
    if text is None:
        return []
    if isinstance(text, str):
        return [text]
    try:
        return [str(item) for item in text]
    except TypeError:
        return [str(text)]


@pytest.fixture
def fake_reference(tmp_path, monkeypatch):
    return patch_reference_layer(monkeypatch, tmp_path)


def _write_tsv_gz(df, path):
    text = df.to_csv(sep="\t", index=False)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(text)


# ----------------------------------------------------------------------
# 1. public import
# ----------------------------------------------------------------------
def test_public_plot_is_importable_and_callable():
    assert callable(locusblend.plot)
    assert locusblend.plot is api.plot


# ----------------------------------------------------------------------
# 2. mode normalization
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "public,expected",
    [
        ("standard", MODE_STANDARD),
        ("two", MODE_TWO_INDEX),
        ("three", MODE_THREE_INDEX),
        (" Three ", MODE_THREE_INDEX),
    ],
)
def test_normalize_public_mode(public, expected):
    assert normalize_public_mode(public) == expected


@pytest.mark.parametrize("bad", ["four", "", None, "locus", "three-index"])
def test_normalize_public_mode_rejects_unknown(bad):
    with pytest.raises(ValueError):
        normalize_public_mode(bad)


@pytest.mark.parametrize("bad", ["top", "bottom", "", None, "dataset3"])
def test_normalize_auto_index_source_rejects_unknown(bad):
    with pytest.raises(ValueError):
        normalize_auto_index_source(bad)


@pytest.mark.parametrize(
    "mode,expected_n",
    [("standard", 1), ("two", 2), ("three", 3)],
)
def test_mode_selects_required_number_of_index_variants(fake_reference, mode, expected_n):
    result = locusblend.plot(
        make_dataset("one"),
        make_dataset("two"),
        reference_dir="ignored",
        mode=mode,
    )
    assert result.metadata["required_n_indices"] == expected_n
    assert len(result.index_variants) == expected_n
    assert len(result.index_reference_snps) == expected_n
    assert result.metadata["mode_label"] == normalize_public_mode(mode)


def test_plot_rejects_unsupported_mode_before_work(fake_reference):
    with pytest.raises(ValueError):
        locusblend.plot(make_dataset("one"), make_dataset("two"), reference_dir="x", mode="four")
    assert fake_reference.compute_ld == []


# ----------------------------------------------------------------------
# 3./4. DataFrame and file-path inputs
# ----------------------------------------------------------------------
def test_dataframe_inputs_default_titles(fake_reference):
    result = locusblend.plot(
        make_dataset("one"),
        make_dataset("two"),
        reference_dir="ignored",
        mode="standard",
    )
    assert isinstance(result, LocusBlendResult)
    assert result.metadata["config"].title_top == "Dataset 1"
    assert result.metadata["config"].title_bottom == "Dataset 2"


def test_file_path_inputs_use_filename_titles(fake_reference, tmp_path):
    path1 = tmp_path / "trait1.tsv.gz"
    path2 = tmp_path / "trait2.csv"
    _write_tsv_gz(make_dataset("one"), path1)
    make_dataset("two").to_csv(path2, index=False)

    result = locusblend.plot(
        path1,
        path2,
        reference_dir="ignored",
        mode="standard",
        chrom="14",
        center_bp=CENTER_BP,
    )

    assert result.metadata["config"].title_top == "trait1"
    assert result.metadata["config"].title_bottom == "trait2"
    assert result.chromosome == "14"
    assert result.center_bp == CENTER_BP
    assert "dataset_tag" in result.dataset1_processed.columns


def test_unsupported_input_type_raises(fake_reference):
    with pytest.raises(TypeError):
        locusblend.plot(123, make_dataset("two"), reference_dir="ignored")


# ----------------------------------------------------------------------
# 5. locus inference
# ----------------------------------------------------------------------
def test_locus_inference_when_chrom_and_bp_omitted(fake_reference):
    result = locusblend.plot(
        make_dataset("one"), make_dataset("two"), reference_dir="ignored", mode="standard"
    )

    assert result.chromosome == "14"
    # choose_sync_center_bp picks the best valid P row (forced at CENTER_BP)
    assert result.center_bp == CENTER_BP
    assert result.window_kb == 500
    assert result.window_start_bp == CENTER_BP - 500_000
    assert result.window_end_bp == CENTER_BP + 500_000


def test_explicit_locus_is_used_as_supplied(fake_reference):
    result = locusblend.plot(
        make_dataset("one"),
        make_dataset("two"),
        reference_dir="ignored",
        mode="standard",
        chrom="chr14",
        center_bp=CENTER_BP + 1234,
        window_kb=250,
    )
    assert result.chromosome == "14"
    assert result.center_bp == CENTER_BP + 1234
    assert result.window_kb == 250
    assert result.metadata["ld_window_start_bp"] == CENTER_BP + 1234 - 250_000
    assert result.metadata["ld_window_end_bp"] == CENTER_BP + 1234 + 250_000


def test_unsupported_chromosome_raises(fake_reference):
    with pytest.raises(ValueError) as excinfo:
        locusblend.plot(
            make_dataset("one"),
            make_dataset("two"),
            reference_dir="ignored",
            chrom="chrY",
            center_bp=CENTER_BP,
        )
    assert "Unsupported chromosome" in str(excinfo.value)


def test_locus_inference_failure_raises(fake_reference):
    with pytest.raises(ValueError) as excinfo:
        locusblend.plot(
            make_dataset("one", chrom="chrY"),
            make_dataset("two", chrom="chrY"),
            reference_dir="ignored",
        )
    assert "Could not infer a chromosome" in str(excinfo.value)


def test_missing_chromosome_rows_raise(fake_reference):
    with pytest.raises(ValueError) as excinfo:
        locusblend.plot(
            make_dataset("one", chrom="15"),
            make_dataset("two", chrom="15"),
            reference_dir="ignored",
            chrom="14",
            center_bp=CENTER_BP,
        )
    assert "No rows found for chromosome 14" in str(excinfo.value)


def test_center_bp_inference_failure_raises(fake_reference):
    # chromosome 15 rows exist (presence check passes) but no usable BP value,
    # so choose_sync_center_bp cannot infer a center
    ds1 = make_dataset("one", chrom="15")
    ds2 = make_dataset("two", chrom="15")
    ds1["BP"] = np.nan
    ds2["BP"] = np.nan

    with pytest.raises(ValueError) as excinfo:
        locusblend.plot(
            ds1,
            ds2,
            reference_dir="ignored",
            chrom="15",
        )
    assert "Could not infer a center BP" in str(excinfo.value)


# ----------------------------------------------------------------------
# 6. index variants: manual validation and resolution
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "mode,count",
    [("standard", 2), ("two", 1), ("two", 3), ("three", 2)],
)
def test_manual_index_count_must_match_mode(fake_reference, mode, count):
    variants = [f"rsone{i:03d}" for i in range(count)]
    with pytest.raises(ValueError) as excinfo:
        locusblend.plot(
            make_dataset("one"),
            make_dataset("two"),
            reference_dir="ignored",
            mode=mode,
            index_variants=variants,
            chrom="14",
            center_bp=CENTER_BP,
        )
    assert "requires exactly" in str(excinfo.value)


def test_manual_single_string_index_variant(fake_reference):
    ds1 = make_dataset("one")
    manual = str(ds1["rsid"].iloc[7])

    result = locusblend.plot(
        ds1,
        make_dataset("two"),
        reference_dir="ignored",
        mode="standard",
        index_variants=manual,
        chrom="14",
        center_bp=CENTER_BP,
    )

    assert result.index_variants == (manual,)
    assert result.index_reference_snps == (manual,)
    assert result.metadata["index_selection_method"] == "Manual input"
    # manual input must not touch the clumping path
    assert fake_reference.auto_select == []
    assert result.metadata["config"].index_variants == (manual,)


def test_manual_multiple_index_variants(fake_reference):
    ds1 = make_dataset("one")
    variants = [str(ds1["rsid"].iloc[i]) for i in (3, 4, 5)]

    result = locusblend.plot(
        ds1,
        make_dataset("two"),
        reference_dir="ignored",
        mode="three",
        index_variants=variants,
        chrom="14",
        center_bp=CENTER_BP,
    )

    assert result.index_variants == tuple(variants)
    assert result.index_reference_snps == tuple(variants)
    assert [record.rank for record in result.selected_index_variants] == [1, 2, 3]
    assert all(record.display_id in variants for record in result.selected_index_variants)


def test_manual_unknown_variant_raises(fake_reference):
    with pytest.raises(ValueError) as excinfo:
        locusblend.plot(
            make_dataset("one"),
            make_dataset("two"),
            reference_dir="ignored",
            mode="standard",
            index_variants="rs-not-present",
            chrom="14",
            center_bp=CENTER_BP,
        )
    assert "Could not resolve index SNP" in str(excinfo.value)


def test_manual_index_without_reference_match_falls_back(fake_reference):
    """Manual index variant present in the data but absent from the 1000G BIM."""
    ds1 = make_dataset("one")
    ds2 = make_dataset("two")
    extra_bp = int(ds1["BP"].max()) + 1  # position not covered by the fake BIM
    extra = pd.DataFrame(
        [
            {
                "CHR": "14",
                "BP": extra_bp,
                "P": 1e-7,
                "A1": "A",
                "A2": "G",
                "rsid": "rs-not-in-1000g",
                "BETA": 0.1,
                "SE": 0.02,
                "dataset_tag": "one",
            }
        ]
    )
    ds1 = pd.concat([ds1, extra], ignore_index=True)

    result = locusblend.plot(
        ds1,
        ds2,
        reference_dir="ignored",
        mode="standard",
        index_variants="rs-not-in-1000g",
        chrom="14",
        center_bp=CENTER_BP,
    )

    # unresolved 1000G reference id is a plain None, not pandas.NA/NaN
    assert result.index_reference_snps == (None,)
    assert result.selected_index_variants[0].ref_snp is None
    assert result.index_variants == ("rs-not-in-1000g",)
    assert result.ld_status["in_reference"]["variant 1"] is False

    # the input row itself is kept and marked as missing-reference
    matched = result.dataset1_processed[
        result.dataset1_processed["rsid"] == "rs-not-in-1000g"
    ].iloc[0]
    assert pd.isna(matched["REF_SNP"])
    assert bool(matched["in_ref"]) is False

    # plotting falls back to DISPLAY_ID and draws the missing-reference grey X
    index_traces = [
        tr
        for tr in result.locus_figure.data
        if any("rs-not-in-1000g" in text for text in _trace_texts(tr))
    ]
    assert index_traces, "expected an index marker trace for the unresolved variant"
    assert all(tr.marker.symbol == "x" for tr in index_traces)
    assert any(
        "Index SNP not found in 1000G EUR LD" in text
        for tr in index_traces
        for text in _trace_texts(tr)
    )


# ----------------------------------------------------------------------
# 7. auto-index source selection
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "source,expected_tag",
    [("dataset1", "one"), ("dataset2", "two")],
)
def test_auto_index_source_selection(fake_reference, source, expected_tag):
    result = locusblend.plot(
        make_dataset("one"),
        make_dataset("two"),
        reference_dir="ignored",
        mode="two",
        auto_index_source=source,
        chrom="14",
        center_bp=CENTER_BP,
    )

    assert len(fake_reference.auto_select) == 1
    used = fake_reference.auto_select[0]["df_source_ref"]
    assert set(used["dataset_tag"]) == {expected_tag}

    assert len(result.index_variants) == 2
    assert all(label.startswith(f"rs{expected_tag}") for label in result.index_variants)
    assert result.metadata["auto_index_source"] == source

    # clumping parameters follow the baseline defaults
    call = fake_reference.auto_select[0]
    assert call["clump_r2"] == 0.01
    assert call["required_n"] == 2
    assert call["active_ld_source"] == "Use internal 1000G reference"
    assert call["bfile_prefix"] == fake_reference.bfile_prefix
    assert call["active_locusblend_mode"] == MODE_TWO_INDEX
    assert call["window_bp"] == 500_000
    assert call["selected_chrom"] == "14"


def test_invalid_auto_index_source_raises(fake_reference):
    with pytest.raises(ValueError):
        locusblend.plot(
            make_dataset("one"),
            make_dataset("two"),
            reference_dir="ignored",
            auto_index_source="dataset3",
        )


# ----------------------------------------------------------------------
# 8. populated result
# ----------------------------------------------------------------------
def test_result_is_populated(fake_reference):
    result = locusblend.plot(
        make_dataset("one"),
        make_dataset("two"),
        reference_dir="ignored",
        ancestry="eur",
        mode="three",
        chrom="14",
        center_bp=CENTER_BP,
        highlight_genes="PSEN1, PAPLN",
        output=None,
    )

    assert result.locus_figure is not None
    assert result.compare_figure is not None
    assert result.locus_figure is result.locus_fig
    assert result.compare_figure is result.compare_fig

    assert result.chromosome == "14"
    assert result.center_bp == CENTER_BP
    assert result.window_kb == 500
    assert result.mode == "three"
    assert result.ancestry == "EUR"
    assert result.window_start_bp == CENTER_BP - 500_000
    assert result.window_end_bp == CENTER_BP + 500_000

    assert len(result.index_variants) == 3
    assert len(result.index_reference_snps) == 3
    assert len(result.selected_index_variants) == 3
    assert all(isinstance(v, IndexVariant) for v in result.selected_index_variants)

    assert isinstance(result.dataset1_processed, pd.DataFrame)
    assert isinstance(result.dataset2_processed, pd.DataFrame)
    for column in [
        "REF_SNP",
        "REF_MATCH",
        "REF_MATCH_TYPE",
        "NEED_FLIP",
        "r2_1",
        "r2_2",
        "r2_3",
        "in_ref",
    ]:
        assert column in result.dataset1_processed.columns
        assert column in result.dataset2_processed.columns

    assert result.genes is not None and len(result.genes) == 2
    assert result.ld_status["ancestry"] == "EUR"
    assert result.ld_status["in_reference"] == {
        "variant 1": True,
        "variant 2": True,
        "variant 3": True,
    }
    assert "Chromosome: chr14" in result.ld_status["caption"]
    assert result.n_window_snps == len(fake_reference.compute_ld[0]["window_snps"])

    assert result.metadata["reference_dir"] == str(fake_reference.reference_dir)
    assert result.metadata["bfile_prefix"] == fake_reference.bfile_prefix
    assert result.metadata["plink_executable"] == "plink-fake"
    assert result.metadata["reference_validation"]["ok"] is True
    assert result.metadata["reference_validation"]["chromosome"] == "14"
    assert result.metadata["config"].gene_display_mode == "protein_coding"
    assert result.metadata["n_genes"] == 2
    assert result.metadata["compare_mode"] == "Three separate compare plots"
    assert set(result.metadata["compare_counts"]) == {"variant 1", "variant 2", "variant 3"}
    assert "Dataset 1 SNP count" in result.metadata["summary_text"]
    assert result.metadata["output_written"] is False
    assert result.metadata["output_path"] is None

    # gene track got the inferred gene window and the protein-coding default
    genes_call = fake_reference.genes[0]
    assert genes_call["chrom"] == "14"
    assert genes_call["gene_display_mode"] == "protein_coding"
    assert genes_call["start"] <= CENTER_BP <= genes_call["end"]

    # LD was computed on the requested window with the matched index SNPs
    ld_call = fake_reference.compute_ld[0]
    assert ld_call["chrom"] == "14"
    assert ld_call["start"] == CENTER_BP - 500_000
    assert ld_call["end"] == CENTER_BP + 500_000
    assert ld_call["idx1_ref"] == result.index_reference_snps[0]


def test_result_exposes_three_row_locus_figure_and_compare(fake_reference):
    result = locusblend.plot(
        make_dataset("one"),
        make_dataset("two"),
        reference_dir="ignored",
        mode="two",
        chrom="14",
        center_bp=CENTER_BP,
    )

    # 3-row combined figure: two locus rows + gene track row
    assert result.locus_figure.layout.height == 980
    assert len(result.locus_figure.layout.annotations) >= 3
    subplot_titles = [ann.text for ann in result.locus_figure.layout.annotations[:3]]
    assert subplot_titles[0] == "Dataset 1"
    assert subplot_titles[1] == "Dataset 2"
    assert subplot_titles[2] == "GENCODE gene track (protein_coding)"

    # compare figure carries only the active panels for the selected mode
    assert result.compare_figure is not None
    assert result.metadata["compare_counts"].get("variant 3", 0) == 0


def test_gene_display_mode_all_is_forwarded(fake_reference):
    result = locusblend.plot(
        make_dataset("one"),
        make_dataset("two"),
        reference_dir="ignored",
        mode="standard",
        chrom="14",
        center_bp=CENTER_BP,
        gene_display_mode="all",
    )
    assert fake_reference.genes[0]["gene_display_mode"] == "all"
    assert result.metadata["config"].gene_display_mode == "all"


def test_gene_display_mode_validation(fake_reference):
    with pytest.raises(ValueError):
        locusblend.plot(
            make_dataset("one"),
            make_dataset("two"),
            reference_dir="ignored",
            gene_display_mode="all_genes",
        )


def test_recombination_missing_is_best_effort(fake_reference):
    result = locusblend.plot(
        make_dataset("one"),
        make_dataset("two"),
        reference_dir="ignored",
        mode="standard",
        chrom="14",
        center_bp=CENTER_BP,
        show_recombination=True,
    )
    assert result.metadata["recombination_bw_path"] is None
    assert any("Recombination BigWig not found" in w for w in result.warnings)
    # the figure is still produced (three rows: two loci + gene track)
    assert len(result.locus_figure.layout.annotations) >= 3


# ----------------------------------------------------------------------
# 9./10. output handling
# ----------------------------------------------------------------------
def test_output_none_writes_nothing(fake_reference, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = locusblend.plot(
        make_dataset("one"),
        make_dataset("two"),
        reference_dir="ignored",
        mode="standard",
        chrom="14",
        center_bp=CENTER_BP,
        output=None,
    )
    assert result.metadata["output_written"] is False
    assert list(tmp_path.glob("**/*.png")) == []


def test_png_output_uses_export_helper(fake_reference, tmp_path, monkeypatch):
    written = []

    def fake_render(fig, width_px, height_px, scale=1):
        written.append((fig, int(width_px), int(height_px), scale))
        return b"\x89PNG\r\n\x1a\nfake-export"

    monkeypatch.setattr(api, "render_plotly_figure_to_png_bytes", fake_render)

    output = tmp_path / "nested" / "dir" / "locusblend.png"
    result = locusblend.plot(
        make_dataset("one"),
        make_dataset("two"),
        reference_dir="ignored",
        mode="standard",
        chrom="14",
        center_bp=CENTER_BP,
        output=output,
    )

    assert output.is_file()
    assert output.read_bytes() == b"\x89PNG\r\n\x1a\nfake-export"
    assert len(written) == 1
    rendered_fig, width_px, height_px, scale = written[0]
    assert rendered_fig is result.locus_figure
    assert width_px == api.DEFAULT_PNG_WIDTH_PX
    assert height_px == 980
    assert scale == 1
    assert result.metadata["output_written"] is True
    assert result.metadata["output_path"] == str(output)


# ----------------------------------------------------------------------
# PLINK discovery / requirement
# ----------------------------------------------------------------------
def test_find_plink_exec_missing_binary_is_actionable(monkeypatch):
    monkeypatch.delenv("LOCUSBLEND_PLINK", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(FileNotFoundError) as excinfo:
        _find_plink_exec("definitely-not-a-real-plink-binary")

    message = str(excinfo.value)
    assert "PLINK executable not found" in message
    assert "LOCUSBLEND_PLINK" in message
    assert "plink_path" in message


def test_auto_index_without_plink_raises_file_not_found(tmp_path, monkeypatch):
    """PLINK is required: preflight uses real discovery and fails early."""
    calls = patch_reference_layer(monkeypatch, tmp_path, patch_auto_select=False, patch_plink=False)
    monkeypatch.delenv("LOCUSBLEND_PLINK", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(FileNotFoundError) as excinfo:
        locusblend.plot(
            make_dataset("one"),
            make_dataset("two"),
            reference_dir="ignored",
            mode="standard",
            chrom="14",
            center_bp=CENTER_BP,
        )

    assert "PLINK executable not found" in str(excinfo.value)
    # failed before any expensive work: no BIM read, no LD, no gene parsing
    assert calls.load_reference_bim == []
    assert calls.compute_ld == []
    assert calls.genes == []


# ----------------------------------------------------------------------
# Pass 3: environment preflight
# ----------------------------------------------------------------------
def test_plot_rejects_invalid_ancestry_before_loading_datasets(tmp_path, monkeypatch):
    """Ancestry typos must raise instead of silently using EUR."""
    monkeypatch.delenv("LOCUSBLEND_REFERENCE_DIR", raising=False)

    with pytest.raises(ValueError) as excinfo:
        locusblend.plot(
            "not-a-dataframe-or-path.gz",
            "also-not-a-dataset.gz",
            reference_dir=tmp_path,
            ancestry="ABC",
        )

    message = str(excinfo.value)
    assert "Unsupported ancestry 'ABC'" in message
    assert "EUR" in message


def test_plot_accepts_mixed_case_ancestry(fake_reference):
    result = locusblend.plot(
        make_dataset("one"),
        make_dataset("two"),
        reference_dir="ignored",
        ancestry="eUr",
        mode="standard",
        chrom="14",
        center_bp=CENTER_BP,
    )
    assert result.ancestry == "EUR"
    assert result.ld_status["ancestry"] == "EUR"


def test_plot_missing_reference_dir_fails_early_without_datasets(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCUSBLEND_REFERENCE_DIR", raising=False)

    with pytest.raises(ValueError) as excinfo:
        locusblend.plot(
            123,  # invalid input type: must never be reached
            456,
            reference_dir=None,
        )

    message = str(excinfo.value)
    assert "reference_dir is not configured" in message
    assert "LOCUSBLEND_REFERENCE_DIR" in message


def test_plot_nonexistent_reference_dir_fails_early(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCUSBLEND_REFERENCE_DIR", raising=False)

    with pytest.raises(ValueError) as excinfo:
        locusblend.plot(
            make_dataset("one"),
            make_dataset("two"),
            reference_dir=tmp_path / "does_not_exist",
        )

    assert "reference_dir does not exist" in str(excinfo.value)


def test_plot_missing_chromosome_reference_files_fails_before_ld(fake_reference):
    Path(fake_reference.bfile_prefix + ".bed").unlink()

    with pytest.raises(FileNotFoundError) as excinfo:
        locusblend.plot(
            make_dataset("one"),
            make_dataset("two"),
            reference_dir="ignored",
            mode="standard",
            chrom="14",
            center_bp=CENTER_BP,
        )

    message = str(excinfo.value)
    assert "PLINK reference files for chromosome 14 are missing" in message
    assert ".bed" in message
    # no expensive work happened
    assert fake_reference.load_reference_bim == []
    assert fake_reference.compute_ld == []
    assert fake_reference.genes == []


def test_plot_missing_gencode_fails_before_ld(fake_reference):
    Path(fake_reference.gtf_path).unlink()

    with pytest.raises(FileNotFoundError) as excinfo:
        locusblend.plot(
            make_dataset("one"),
            make_dataset("two"),
            reference_dir="ignored",
            mode="standard",
            chrom="14",
            center_bp=CENTER_BP,
        )

    assert "GENCODE annotation for chromosome 14 is missing" in str(excinfo.value)
    assert fake_reference.load_reference_bim == []
    assert fake_reference.compute_ld == []


def test_plot_missing_ancestry_directory_fails_early(tmp_path, monkeypatch):
    # fake tree with EUR, but the call asks for SAS
    patch_reference_layer(monkeypatch, tmp_path, ancestry="EUR")

    with pytest.raises(FileNotFoundError) as excinfo:
        locusblend.plot(
            make_dataset("one"),
            make_dataset("two"),
            reference_dir="ignored",
            ancestry="SAS",
            mode="standard",
            chrom="14",
            center_bp=CENTER_BP,
        )

    assert "SAS reference directory is missing" in str(excinfo.value)


def test_plot_missing_recombination_is_not_fatal(fake_reference):
    result = locusblend.plot(
        make_dataset("one"),
        make_dataset("two"),
        reference_dir="ignored",
        mode="standard",
        chrom="14",
        center_bp=CENTER_BP,
        show_recombination=True,
    )
    assert result.metadata["reference_validation"]["ok"] is True
    assert result.metadata["reference_validation"]["warnings"]
    assert any("Recombination BigWig not found" in w for w in result.warnings)

    # not requested -> no warning noise, still no error
    quiet = locusblend.plot(
        make_dataset("one"),
        make_dataset("two"),
        reference_dir="ignored",
        mode="standard",
        chrom="14",
        center_bp=CENTER_BP,
        show_recombination=False,
    )
    assert quiet.warnings == ()
    assert quiet.metadata["reference_validation"]["warnings"]
