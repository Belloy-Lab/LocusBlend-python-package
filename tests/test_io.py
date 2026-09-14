"""Column/chromosome normalization behavior of the extracted IO helpers."""

import io
import gzip

import pandas as pd
import pytest

from locusblend.io import (
    _clean_locus_df,
    chrom_mask,
    chrom_sort_key,
    format_chrom_axis_title,
    format_chrom_label,
    get_supported_chromosomes,
    get_uploaded_summary_file_kind,
    is_supported_chrom,
    normalize_chrom,
    read_summary_stats_file,
)


def _messy_summary_frame():
    # Aliases taken from the baseline's accepted column list.
    return pd.DataFrame(
        {
            "chr": ["chr14", "14", "chr23", "X"],
            "base_pair_location": [73238768, 73239000, 13189, 13189],
            "markername": ["rs11159021", "rs3742825", "rs74986264", "rs0001"],
            "pval": [1e-6, 0.01, 1e-9, 0.5],
            "ALLELE1": ["a", "g", "t", "c"],
            "ALLELE0": ["g", "a", "c", "t"],
            "Effect": [0.12, -0.05, 0.3, 0.01],
            "StdErr": [0.03, 0.02, 0.05, 0.01],
            "EAF": [0.2, 0.6, 0.1, 0.4],
            "sample_size": [1000, 1200, 900, 800],
        },
        index=[0, 1, 2, 3],
    )


def test_clean_locus_df_normalizes_columns_and_aliases():
    df = _clean_locus_df(_messy_summary_frame(), source_name="test")

    for col in ["CHR", "BP", "P", "rsid", "A1", "A2", "BETA", "SE", "A1FREQ", "N"]:
        assert col in df.columns, col

    # chr prefix stripped; like the baseline, "23" stays "23" at this stage and
    # is only mapped to "X" by the chromosome helpers below
    assert df["CHR"].tolist() == ["14", "14", "23", "X"]

    # numeric coercion and allele upper-casing
    assert df["BP"].tolist() == [73238768, 73239000, 13189, 13189]
    assert df["A1"].tolist() == ["A", "G", "T", "C"]
    assert df["A2"].tolist() == ["G", "A", "C", "T"]
    assert df["P"].dtype.kind == "f"

    # derived helper columns
    assert df["posID"].tolist()[0] == "14:73238768"
    assert df["uniqueid"].tolist()[0] == "14:73238768:A:G"
    assert df["DISPLAY_ID"].tolist() == ["rs11159021", "rs3742825", "rs74986264", "rs0001"]


def test_chromosome_23_and_x_uid_pattern():
    df = _clean_locus_df(_messy_summary_frame(), source_name="test")

    # downstream helpers treat 23 and X as chromosome X
    assert [normalize_chrom(c) for c in df["CHR"]] == ["14", "14", "X", "X"]
    assert chrom_mask(df, "X").tolist() == [False, False, True, True]

    # X-based allele IDs use the same CHR:BP:A1:A2 pattern as autosomes
    assert df.loc[df["CHR"] == "X", "uniqueid"].tolist() == ["X:13189:C:T"]


def test_clean_locus_df_requires_core_columns():
    with pytest.raises(ValueError):
        _clean_locus_df(pd.DataFrame({"CHR": ["1"], "BP": [1]}), source_name="test")


def test_dedup_columns_is_addressed_in_clean_locus_df():
    raw = pd.DataFrame(
        [[1, 2, 3, 4, 5, 6]],
        columns=["CHR", "BP", "P", "A1", "A2", "A1"],
    )
    df = _clean_locus_df(raw, source_name="test")
    assert list(df.columns).count("A1") == 1
    assert list(df.columns).count("A2") == 1


@pytest.mark.parametrize(
    "value,expected",
    [
        ("14", "14"),
        (14, "14"),
        ("chr14", "14"),
        ("14.0", "14"),
        ("X", "X"),
        ("x", "X"),
        ("chrX", "X"),
        (23, "X"),
        ("23", "X"),
        ("chr23", "X"),
        (None, ""),
    ],
)
def test_normalize_chrom(value, expected):
    assert normalize_chrom(value) == expected


def test_chromosome_helpers():
    assert get_supported_chromosomes() == [str(i) for i in range(1, 23)] + ["X"]
    assert is_supported_chrom("chrX") is True
    assert is_supported_chrom("chrY") is False
    assert chrom_sort_key("X") == 23
    assert chrom_sort_key("chrX") == 23
    assert chrom_sort_key("2") < chrom_sort_key("10") < chrom_sort_key("X")
    assert format_chrom_label(23) == "chrX"
    assert format_chrom_axis_title("chrX") == "Chromosome X (Mb)"


def test_chrom_mask_matches_normalized_chromosomes():
    df = pd.DataFrame({"CHR": ["14", "chr14", 23, "X", "1"]})
    assert chrom_mask(df, "chr14").tolist() == [True, True, False, False, False]
    assert chrom_mask(df, 23).tolist() == [False, False, True, True, False]
    assert chrom_mask(df, "X").tolist() == [False, False, True, True, False]


def test_chrom_mask_requires_chr_column():
    with pytest.raises(KeyError):
        chrom_mask(pd.DataFrame({"BP": [1]}), "1")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("data.csv", "csv"),
        ("data.TSV", "tsv"),
        ("data.txt", "txt"),
        ("data.csv.gz", "csv.gz"),
        ("data.tsv.gz", "tsv.gz"),
        ("data.txt.gz", "txt.gz"),
        ("data.xlsx", None),
        ("", None),
        (None, None),
        ("/tmp/upload/case.tsv.gz", "tsv.gz"),
    ],
)
def test_get_uploaded_summary_file_kind(name, expected):
    assert get_uploaded_summary_file_kind(name) == expected


def _buffer(text, name):
    buf = io.BytesIO(text.encode("utf-8"))
    buf.name = name
    return buf


def test_read_summary_stats_file_csv_tsv_and_txt():
    body_comma = "CHR,BP,P,A1,A2\n14,100,1e-6,A,G\n"
    body_tab = "CHR\tBP\tP\tA1\tA2\n14\t100\t1e-6\tA\tG\n"
    body_space = "CHR BP P A1 A2\n14 100 1e-6 A G\n"

    assert list(read_summary_stats_file(_buffer(body_comma, "a.csv")).columns) == [
        "CHR", "BP", "P", "A1", "A2",
    ]
    assert list(read_summary_stats_file(_buffer(body_tab, "a.tsv")).columns) == [
        "CHR", "BP", "P", "A1", "A2",
    ]
    assert list(read_summary_stats_file(_buffer(body_space, "a.txt")).columns) == [
        "CHR", "BP", "P", "A1", "A2",
    ]


def test_read_summary_stats_file_gzip():
    body = "CHR\tBP\tP\tA1\tA2\n14\t100\t1e-6\tA\tG\n"
    buf = io.BytesIO(gzip.compress(body.encode("utf-8")))
    buf.name = "a.tsv.gz"
    df = read_summary_stats_file(buf)
    assert df.shape == (1, 5)


def test_read_summary_stats_file_rejects_unsupported_format():
    with pytest.raises(ValueError):
        read_summary_stats_file(_buffer("CHR,BP\n1,2\n", "a.xlsx"))
