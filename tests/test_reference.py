"""ReferenceManager path resolution against a temporary fake reference tree."""

from pathlib import Path

import pytest

from locusblend.reference import (
    INTERNAL_1000G_ANCESTRIES,
    INTERNAL_1000G_DEFAULT_ANCESTRY,
    SUPPORTED_ANCESTRIES,
    ReferenceManager,
    ReferenceValidation,
    default_recombination_bw_path,
    format_internal_1000g_ancestry_option,
    get_gtf_path_for_chrom,
    get_internal_bfile_prefix_for_chrom,
    get_internal_1000g_prefix,
    normalize_internal_1000g_ancestry,
    resolve_ancestry,
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


# ----------------------------------------------------------------------
# ancestry validation (public behavior)
# ----------------------------------------------------------------------
def test_supported_ancestries_constant():
    assert SUPPORTED_ANCESTRIES == ("AFR", "AMR", "EAS", "EUR", "SAS")


@pytest.mark.parametrize(
    "value,expected",
    [
        ("EUR", "EUR"),
        ("eur", "EUR"),
        ("EUr", "EUR"),
        (" afr ", "AFR"),
        ("sas", "SAS"),
        ("aMr", "AMR"),
    ],
)
def test_resolve_ancestry_is_case_insensitive(value, expected):
    assert resolve_ancestry(value) == expected


@pytest.mark.parametrize("bad", ["ABC", "EUR1", "european", "", None, "EU RX"])
def test_resolve_ancestry_rejects_unknown(bad):
    with pytest.raises(ValueError) as excinfo:
        resolve_ancestry(bad)
    message = str(excinfo.value)
    assert "Unsupported ancestry" in message
    assert "EUR" in message and "AFR" in message


def test_reference_manager_is_strict_by_default(tmp_path):
    with pytest.raises(ValueError):
        ReferenceManager(reference_dir=tmp_path, ancestry="ABC")

    # mixed case still normalizes cleanly
    ref = ReferenceManager(reference_dir=tmp_path, ancestry="eUr")
    assert ref.ancestry == "EUR"


def test_reference_manager_opt_out_keeps_baseline_fallback(tmp_path):
    ref = ReferenceManager(reference_dir=tmp_path, ancestry="ABC", strict_ancestry=False)
    assert ref.ancestry == "EUR"

    # the baseline-parity module wrappers keep the silent fallback
    _make_reference_tree(tmp_path, ancestry="EUR")
    assert str(get_internal_1000g_prefix("14", "ABC", reference_dir=tmp_path)).endswith("ch14")
    assert get_internal_bfile_prefix_for_chrom("14", "ABC", reference_dir=tmp_path).endswith("ch14")


def test_baseline_ancestry_helper_still_falls_back():
    assert normalize_internal_1000g_ancestry("ABC") == "EUR"
    assert normalize_internal_1000g_ancestry("sas") == "SAS"


# ----------------------------------------------------------------------
# validation / preflight
# ----------------------------------------------------------------------
def test_validate_for_locus_ok_when_everything_is_present(tmp_path):
    _make_reference_tree(tmp_path, chrom="14")
    ref = ReferenceManager(reference_dir=tmp_path, ancestry="EUR")

    validation = ref.validate_for_locus("14")

    assert isinstance(validation, ReferenceValidation)
    assert validation.ok is True
    assert validation.errors == ()
    assert validation.warnings == ()
    assert validation.chrom == "14"
    assert validation.ancestry == "EUR"
    # only the requested chromosome is checked (no full-directory scan)
    assert validation.checked_paths["bfile_prefix"].endswith("ch14")
    # describe()/to_dict() are usable for diagnostics
    assert "reference validation: ok" in validation.describe()
    assert validation.to_dict()["chromosome"] == "14"


def test_validate_unset_reference_dir_is_a_config_error(monkeypatch):
    monkeypatch.delenv("LOCUSBLEND_REFERENCE_DIR", raising=False)
    ref = ReferenceManager()

    validation = ref.validate()
    assert validation.ok is False
    assert validation.config_errors
    assert validation.file_errors == ()
    assert "LOCUSBLEND_REFERENCE_DIR" in validation.config_errors[0]

    with pytest.raises(ValueError) as excinfo:
        validation.raise_for_errors()
    assert "reference_dir" in str(excinfo.value)

    # and the per-locus variant behaves the same way
    with pytest.raises(ValueError):
        ref.validate_for_locus("14").raise_for_errors()


def test_validate_nonexistent_reference_dir(tmp_path):
    missing = tmp_path / "not_here"
    ref = ReferenceManager(reference_dir=missing, ancestry="EUR")

    validation = ref.validate()
    assert validation.ok is False
    assert any("does not exist" in error for error in validation.config_errors)
    with pytest.raises(ValueError):
        validation.raise_for_errors()


def test_validate_missing_ancestry_directory(tmp_path):
    # reference_dir exists and has gencode, but no 1000g/SAS directory
    (tmp_path / "gencode").mkdir(parents=True)
    (tmp_path / "recombination").mkdir(parents=True)
    ref = ReferenceManager(reference_dir=tmp_path, ancestry="SAS")

    validation = ref.validate_for_locus("14")
    assert validation.ok is False
    assert validation.config_errors == ()
    assert any("SAS reference directory is missing" in error for error in validation.file_errors)
    with pytest.raises(FileNotFoundError) as excinfo:
        validation.raise_for_errors()
    message = str(excinfo.value)
    assert "reference_dir/1000g/SAS" in message
    assert "reference_dir=... or set LOCUSBLEND_REFERENCE_DIR" in message


def test_validate_missing_bfile_files(tmp_path):
    _make_reference_tree(tmp_path, chrom="14")
    ref = ReferenceManager(reference_dir=tmp_path, ancestry="EUR")
    Path(str(ref.bfile_prefix("14")) + ".bed").unlink()

    validation = ref.validate_for_locus("14")
    assert validation.ok is False
    assert len(validation.file_errors) == 1
    error = validation.file_errors[0]
    assert ".bed" in error
    assert "chromosome 14" in error
    # .bim/.fam still exist, so they are not reported
    assert ".bim" not in error and ".fam" not in error

    with pytest.raises(FileNotFoundError):
        validation.raise_for_errors()


def test_validate_missing_gencode_annotation(tmp_path):
    _make_reference_tree(tmp_path, chrom="14")
    ref = ReferenceManager(reference_dir=tmp_path, ancestry="EUR")
    ref.gtf_path("14").unlink()

    validation = ref.validate_for_locus("14")
    assert validation.ok is False
    assert any("GENCODE annotation for chromosome 14 is missing" in e for e in validation.file_errors)


def test_validate_chromosome_x_gencode_fallback(tmp_path):
    _make_reference_tree(tmp_path, chrom="X")
    ref = ReferenceManager(reference_dir=tmp_path, ancestry="EUR")

    # chrX file present -> ok
    assert ref.validate_for_locus("X").ok is True

    # chrX file missing but the whole-genome annotation present -> ok
    ref.gtf_path("X").unlink()
    ref.genome_gtf_path().write_text("", encoding="utf-8")
    validation = ref.validate_for_locus("X")
    assert validation.ok is True
    assert validation.checked_paths["gencode_genome"].endswith("gencode.v49.annotation.gtf.gz")

    # both missing -> GENCODE error naming both candidates
    ref.genome_gtf_path().unlink()
    validation = ref.validate_for_locus("X")
    assert validation.ok is False
    assert any("chromosome X is missing" in e for e in validation.file_errors)


def test_validate_missing_recombination_is_only_a_warning(tmp_path):
    _make_reference_tree(tmp_path, chrom="14")
    ref = ReferenceManager(reference_dir=tmp_path, ancestry="EUR")
    ref.recombination_bw_path().unlink()

    validation = ref.validate_for_locus("14")
    assert validation.ok is True
    assert validation.file_errors == ()
    assert len(validation.warnings) == 1
    assert "Recombination BigWig not found" in validation.warnings[0]
    assert "recombination overlay is skipped" in validation.warnings[0]

    # non-fatal: raise_for_errors does nothing
    validation.raise_for_errors()

    # explicit opt-in turns it into an error for callers that truly need it
    strict = ref.validate_for_locus("14", require_recombination=True)
    assert strict.ok is False
    with pytest.raises(FileNotFoundError):
        strict.raise_for_errors()


def test_validate_without_chrom_checks_chromosome_independent_parts(tmp_path):
    _make_reference_tree(tmp_path, chrom="14")
    ref = ReferenceManager(reference_dir=tmp_path, ancestry="EUR")

    validation = ref.validate()
    assert validation.ok is True
    assert validation.chrom is None
    assert "ancestry_dir" in validation.checked_paths
    assert "gencode_dir" in validation.checked_paths
    # no chromosome-specific paths were probed
    assert "bfile_prefix" not in validation.checked_paths

    # a missing gencode directory is reported without a chromosome
    other_root = tmp_path / "no_gencode"
    (other_root / "1000g" / "EUR").mkdir(parents=True)
    other = ReferenceManager(reference_dir=other_root, ancestry="EUR")
    validation = other.validate()
    assert validation.ok is False
    assert any("GENCODE directory is missing" in e for e in validation.file_errors)


def test_validate_rejects_unrecognized_ancestry_before_any_lookup(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        ReferenceManager(reference_dir=tmp_path, ancestry="NOPE")
    assert "Unsupported ancestry 'NOPE'" in str(excinfo.value)
