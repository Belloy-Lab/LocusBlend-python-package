"""ReferenceManager path resolution against a temporary fake reference tree."""

from pathlib import Path

import pytest

from locusblend.reference import (
    INTERNAL_1000G_ANCESTRIES,
    INTERNAL_1000G_DEFAULT_ANCESTRY,
    ReferenceManager,
    default_recombination_bw_path,
    format_internal_1000g_ancestry_option,
    get_gtf_path_for_chrom,
    get_internal_1000g_prefix,
    normalize_internal_1000g_ancestry,
)

BFILE_TEMPLATE = (
    "1000g_{ancestry}_hg38_high_coverage_Illumina.filtered."
    "SNV_INDEL_SV_phased_panel_include_MHC_ch{chrom}"
)


def _make_reference_tree(root: Path, ancestry="EUR", chrom="14") -> Path:
    (root / "1000g" / ancestry).mkdir(parents=True, exist_ok=True)
    (root / "gencode").mkdir(parents=True, exist_ok=True)
    (root / "recombination").mkdir(parents=True, exist_ok=True)

    prefix = root / "1000g" / ancestry / BFILE_TEMPLATE.format(ancestry=ancestry, chrom=chrom)
    for suffix in (".bed", ".bim", ".fam"):
        Path(str(prefix) + suffix).write_text("", encoding="utf-8")

    (root / "gencode" / f"gencode.v49.annotation.chr{chrom}.gtf.gz").write_text("", encoding="utf-8")
    (root / "recombination" / "recomb1000GAvg.bw").write_text("", encoding="utf-8")
    return prefix


def test_ancestry_normalization_and_labels():
    assert INTERNAL_1000G_DEFAULT_ANCESTRY == "EUR"
    assert set(INTERNAL_1000G_ANCESTRIES) == {"AFR", "AMR", "EAS", "EUR", "SAS"}
    assert normalize_internal_1000g_ancestry("eur") == "EUR"
    assert normalize_internal_1000g_ancestry("sas") == "SAS"
    assert normalize_internal_1000g_ancestry("bogus") == "EUR"
    assert normalize_internal_1000g_ancestry(None) == "EUR"
    assert format_internal_1000g_ancestry_option("AFR") == "AFR - African"


def test_bfile_prefix_and_verified_path(tmp_path):
    prefix = _make_reference_tree(tmp_path)
    ref = ReferenceManager(reference_dir=tmp_path, ancestry="EUR")

    # no existence check: plain path construction
    assert ref.bfile_prefix("chr14") == prefix
    assert ref.bfile_prefix(14) == prefix

    # existence check: same string the baseline returned
    assert ref.get_bfile_prefix("14") == str(prefix)
    assert get_internal_1000g_prefix("14", "EUR", reference_dir=tmp_path) == prefix


def test_bfile_prefix_missing_chromosome_reports_all_three_files(tmp_path):
    _make_reference_tree(tmp_path, chrom="14")
    ref = ReferenceManager(reference_dir=tmp_path, ancestry="EUR")

    with pytest.raises(FileNotFoundError) as excinfo:
        ref.get_bfile_prefix("8")

    message = str(excinfo.value)
    assert "chromosome 8 were not found" in message
    assert message.count(".bed") == 1
    assert message.count(".bim") == 1
    assert message.count(".fam") == 1


def test_missing_chromosome_x_uses_baseline_hint(tmp_path):
    _make_reference_tree(tmp_path, chrom="14")
    ref = ReferenceManager(reference_dir=tmp_path, ancestry="EUR")

    with pytest.raises(FileNotFoundError) as excinfo:
        ref.get_bfile_prefix("X")

    message = str(excinfo.value)
    assert "chromosome X were not found" in message
    assert "Select an ancestry with chrX support or upload your own LD reference." in message


def test_gtf_paths_and_chromosome_x_fallback(tmp_path):
    _make_reference_tree(tmp_path, chrom="14")
    ref = ReferenceManager(reference_dir=tmp_path, ancestry="EUR")

    chrom_gtf = tmp_path / "gencode" / "gencode.v49.annotation.chr14.gtf.gz"
    assert ref.get_gtf_path("chr14") == str(chrom_gtf)
    assert ref.gtf_path(14) == chrom_gtf

    # chrX falls back to the whole-genome annotation when the chrX file is absent
    whole_genome = tmp_path / "gencode" / "gencode.v49.annotation.gtf.gz"
    whole_genome.write_text("", encoding="utf-8")
    assert ref.get_gtf_path("X") == str(whole_genome)

    with pytest.raises(FileNotFoundError):
        ref.get_gtf_path("7")

    assert get_gtf_path_for_chrom("14", reference_dir=tmp_path) == str(chrom_gtf)


def test_gtf_missing_chromosome_x_reports_both_candidates(tmp_path):
    _make_reference_tree(tmp_path, chrom="14")
    ref = ReferenceManager(reference_dir=tmp_path, ancestry="EUR")

    with pytest.raises(FileNotFoundError) as excinfo:
        ref.get_gtf_path("X")

    assert "gencode.v49.annotation.chrX.gtf.gz" in str(excinfo.value)
    assert "gencode.v49.annotation.gtf.gz" in str(excinfo.value)


def test_recombination_path(tmp_path):
    _make_reference_tree(tmp_path)
    ref = ReferenceManager(reference_dir=tmp_path, ancestry="EUR")

    expected = tmp_path / "recombination" / "recomb1000GAvg.bw"
    assert ref.recombination_bw_path() == expected
    assert ref.get_recombination_bw_path() == str(expected)

    empty = ReferenceManager(reference_dir=tmp_path / "does_not_exist", ancestry="EUR")
    assert empty.get_recombination_bw_path() is None
    with pytest.raises(FileNotFoundError):
        empty.get_recombination_bw_path(required=True)


def test_reference_dir_from_environment(tmp_path, monkeypatch):
    _make_reference_tree(tmp_path, ancestry="AFR")
    monkeypatch.setenv("LOCUSBLEND_REFERENCE_DIR", str(tmp_path))

    ref = ReferenceManager(ancestry="AFR")
    assert ref.reference_dir == tmp_path
    assert ref.ancestry_dir == tmp_path / "1000g" / "AFR"
    assert ref.get_recombination_bw_path() == str(tmp_path / "recombination" / "recomb1000GAvg.bw")
    assert default_recombination_bw_path() == str(tmp_path / "recombination" / "recomb1000GAvg.bw")


def test_missing_reference_dir_raises_on_path_lookup(monkeypatch):
    monkeypatch.delenv("LOCUSBLEND_REFERENCE_DIR", raising=False)

    ref = ReferenceManager()
    assert ref.has_reference_dir is False
    with pytest.raises(ValueError):
        ref.bfile_prefix("14")
    assert default_recombination_bw_path() is None


def test_describe_lists_layout(tmp_path):
    _make_reference_tree(tmp_path)
    ref = ReferenceManager(reference_dir=tmp_path, ancestry="SAS")
    described = ref.describe()
    assert str(tmp_path) in described
    assert "SAS" in described

