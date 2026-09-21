"""Manual reference-directory tests.

LocusBlend does not bundle or download reference data: you prepare a reference
directory and point the package at it with ``reference_dir=...`` or
``LOCUSBLEND_REFERENCE_DIR``. These tests build tiny fake trees in ``tmp_path``
and never touch the network.
"""

import importlib.util
import json
import socket
import sys
import urllib.request
from pathlib import Path

import pytest

import locusblend
from locusblend import paths
from locusblend.reference import (
    ReferenceManager,
    ReferenceStatus,
    ReferenceValidation,
    reference_status,
)

import test_api as api_helpers

EUR = "EUR"
CHROM = "14"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def make_reference_tree(
    root,
    ancestry=EUR,
    chroms=(CHROM,),
    gencode=True,
    recombination=False,
    partial_chrom=None,
):
    """Create a small manual reference tree at the expected paths."""
    (root / "1000g" / ancestry).mkdir(parents=True, exist_ok=True)
    manager = ReferenceManager(reference_dir=root, ancestry=ancestry)
    for chrom in chroms:
        for suffix in (".bed", ".bim", ".fam"):
            Path(str(manager.bfile_prefix(chrom)) + suffix).write_text("", encoding="utf-8")
    if gencode:
        (root / "gencode").mkdir(parents=True, exist_ok=True)
        gtf = root / "gencode" / f"gencode.v49.annotation.chr{chroms[0]}.gtf.gz"
        gtf.write_text("", encoding="utf-8")
    if recombination:
        (root / "recombination").mkdir(parents=True, exist_ok=True)
        (root / "recombination" / "recomb1000GAvg.bw").write_text("", encoding="utf-8")
    if partial_chrom is not None:
        prefix = manager.bfile_prefix(partial_chrom)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        Path(str(prefix) + ".bed").write_text("", encoding="utf-8")
    return root


def forbid_network(monkeypatch):
    """Fail loudly if anything tries to open a network connection."""
    calls = []

    def boom(*args, **kwargs):
        calls.append(args)
        raise AssertionError("reference handling must not use the network")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    return calls


# ----------------------------------------------------------------------
# resolution: reference_dir -> LOCUSBLEND_REFERENCE_DIR -> unconfigured
# ----------------------------------------------------------------------
def test_explicit_reference_dir_wins(monkeypatch, tmp_path):
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("LOCUSBLEND_REFERENCE_DIR", str(tmp_path / "env"))

    location = paths.resolve_reference_dir(explicit)

    assert location.path == explicit
    assert location.source == "explicit"
    assert location.is_resolved is True

    manager = ReferenceManager(reference_dir=explicit, ancestry=EUR)
    assert manager.reference_dir == explicit
    assert manager.reference_source == "explicit"


def test_environment_variable_fallback(monkeypatch, tmp_path):
    env_dir = tmp_path / "env_reference"
    env_dir.mkdir()
    monkeypatch.setenv("LOCUSBLEND_REFERENCE_DIR", str(env_dir))

    location = paths.resolve_reference_dir()

    assert location.path == env_dir
    assert location.source == "env"

    manager = ReferenceManager(ancestry=EUR)
    assert manager.reference_dir == env_dir
    assert manager.reference_source == "env"
    assert manager.has_reference_dir is True


def test_no_reference_configured(monkeypatch):
    monkeypatch.delenv("LOCUSBLEND_REFERENCE_DIR", raising=False)

    location = paths.resolve_reference_dir()
    assert location.path is None
    assert location.source is None
    assert location.is_resolved is False

    manager = ReferenceManager(ancestry=EUR)
    assert manager.has_reference_dir is False
    assert manager.reference_source is None

    with pytest.raises(ValueError) as excinfo:
        manager.bfile_prefix(CHROM)
    message = str(excinfo.value)
    assert "reference_dir" in message
    assert "LOCUSBLEND_REFERENCE_DIR" in message

    validation = manager.validate()
    assert validation.ok is False
    assert validation.config_errors
    assert validation.file_errors == ()
    assert "reference_dir=..." in validation.config_errors[0]
    assert "LOCUSBLEND_REFERENCE_DIR" in validation.config_errors[0]
    with pytest.raises(ValueError):
        manager.validate_for_locus(CHROM).raise_for_errors()


def test_nonexistent_explicit_path_is_reported(tmp_path):
    missing = tmp_path / "not_here"
    manager = ReferenceManager(reference_dir=missing, ancestry=EUR)

    validation = manager.validate()
    assert validation.ok is False
    assert any("does not exist" in error for error in validation.config_errors)
    with pytest.raises(ValueError):
        validation.raise_for_errors()

    status = reference_status(missing)
    assert status.exists is False
    assert status.reference_dir == str(missing)
    assert any("does not exist yet" in note for note in status.notes)


# ----------------------------------------------------------------------
# ancestry support
# ----------------------------------------------------------------------
@pytest.mark.parametrize("code", ["AFR", "AMR", "EAS", "EUR", "SAS"])
def test_supported_ancestries_are_accepted(tmp_path, code):
    manager = ReferenceManager(reference_dir=tmp_path, ancestry=code.lower())
    assert manager.ancestry == code


@pytest.mark.parametrize("bad", ["ABC", "EUR1", "", None])
def test_invalid_ancestry_is_rejected(tmp_path, bad):
    with pytest.raises(ValueError) as excinfo:
        ReferenceManager(reference_dir=tmp_path, ancestry=bad)
    assert "Unsupported ancestry" in str(excinfo.value)

    # ``ancestry=None`` means "no ancestry filter" for reference_status
    if bad is not None:
        with pytest.raises(ValueError):
            reference_status(reference_dir=tmp_path, ancestry=bad)


# ----------------------------------------------------------------------
# reference_status inspection
# ----------------------------------------------------------------------
def test_status_reports_complete_chromosomes(tmp_path):
    root = make_reference_tree(tmp_path / "reference", chroms=("14", "X"), gencode=True)

    status = reference_status(root)

    assert isinstance(status, ReferenceStatus)
    assert status.reference_dir == str(root)
    assert status.source == "explicit"
    assert status.exists is True
    assert status.ancestry_chroms[EUR] == ("14", "X")
    assert status.available_ancestries == (EUR,)
    assert status.partial_chroms == {}
    assert status.is_chrom_available(EUR, "chr14") is True
    assert status.is_chrom_available("eur", "X") is True
    assert status.is_chrom_available(EUR, "7") is False
    assert status.is_chrom_available("ABC", CHROM) is False
    assert status.gencode_available is True
    assert status.recombination_available is False
    assert "reference status:" in status.describe()
    assert json.loads(json.dumps(status.to_dict()))["available_ancestries"] == [EUR]


def test_status_reports_partial_chromosomes(tmp_path):
    root = make_reference_tree(
        tmp_path / "reference", chroms=(), gencode=False, partial_chrom="7"
    )

    status = reference_status(root)

    assert status.ancestry_chroms == {}
    assert status.partial_chroms[EUR] == ("7",)
    assert status.available_ancestries == ()
    assert any("no complete 1000G ancestry panel" in note for note in status.notes)
    assert any("no GENCODE annotation" in note for note in status.notes)


def test_status_detects_gencode_and_recombination(tmp_path):
    root = make_reference_tree(tmp_path / "reference", gencode=True, recombination=True)

    status = reference_status(root)

    assert status.gencode_available is True
    assert status.recombination_available is True
    assert status.gencode_files == (f"gencode.v49.annotation.chr{CHROM}.gtf.gz",)
    assert status.recombination_files == ("recomb1000GAvg.bw",)
    payload = status.to_dict()
    assert payload["recombination"] == {
        "available": True,
        "optional": True,
        "files": ["recomb1000GAvg.bw"],
    }
    assert "recombination: available" in status.describe()


def test_status_missing_recombination_is_optional(tmp_path):
    root = make_reference_tree(tmp_path / "reference", recombination=False)

    status = reference_status(root)

    assert status.gencode_available is True
    assert status.recombination_available is False
    assert any("optional" in note for note in status.notes)
    assert "not present (optional)" in status.describe()


def test_status_validates_requested_ancestry_and_chrom(tmp_path):
    root = make_reference_tree(tmp_path / "reference", chroms=(CHROM,))

    ok_status = reference_status(root, ancestry=EUR, chrom=CHROM)
    assert ok_status.validation is not None
    assert isinstance(ok_status.validation, ReferenceValidation)
    assert ok_status.validation.ok is True

    missing_status = reference_status(root, ancestry=EUR, chrom="7")
    assert missing_status.validation is not None
    assert missing_status.validation.ok is False
    assert any(".bed" in error for error in missing_status.validation.file_errors)
    assert any("not complete for EUR" in note for note in missing_status.notes)


def test_status_notes_missing_ancestry_panel(tmp_path):
    root = make_reference_tree(tmp_path / "reference", ancestry="AFR", chroms=(CHROM,))

    status = reference_status(root, ancestry=EUR)

    assert status.available_ancestries == ("AFR",)
    assert any("ancestry EUR has no complete chromosome panel" in note for note in status.notes)


def test_status_unconfigured(monkeypatch):
    monkeypatch.delenv("LOCUSBLEND_REFERENCE_DIR", raising=False)

    status = reference_status()

    assert status.reference_dir is None
    assert status.source is None
    assert status.exists is False
    assert any("No reference directory configured" in note for note in status.notes)
    note = status.notes[0]
    assert "reference_dir=..." in note
    assert "LOCUSBLEND_REFERENCE_DIR" in note


def test_status_uses_environment_variable(monkeypatch, tmp_path):
    root = make_reference_tree(tmp_path / "reference")
    monkeypatch.setenv("LOCUSBLEND_REFERENCE_DIR", str(root))

    status = reference_status()

    assert status.reference_dir == str(root)
    assert status.source == "env"
    assert status.available_ancestries == (EUR,)


# ----------------------------------------------------------------------
# plot() integration: manual references, never a download
# ----------------------------------------------------------------------
def test_plot_uses_environment_reference_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCUSBLEND_REFERENCE_DIR", raising=False)
    api_helpers.patch_reference_layer(monkeypatch, tmp_path, patch_manager=False)
    monkeypatch.setenv("LOCUSBLEND_REFERENCE_DIR", str(tmp_path))

    result = locusblend.plot(
        api_helpers.make_dataset("one"),
        api_helpers.make_dataset("two"),
        ancestry=EUR,
        mode="standard",
        chrom=CHROM,
        center_bp=api_helpers.CENTER_BP,
    )

    assert result.metadata["reference_source"] == "env"
    assert result.metadata["reference_dir"] == str(tmp_path)
    assert result.metadata["reference_validation"]["ok"] is True


def test_plot_without_reference_is_actionable_and_uses_no_network(monkeypatch):
    monkeypatch.delenv("LOCUSBLEND_REFERENCE_DIR", raising=False)
    calls = forbid_network(monkeypatch)

    with pytest.raises(ValueError) as excinfo:
        locusblend.plot(
            api_helpers.make_dataset("one"),
            api_helpers.make_dataset("two"),
            ancestry=EUR,
            mode="standard",
            chrom=CHROM,
            center_bp=api_helpers.CENTER_BP,
        )

    message = str(excinfo.value)
    assert "reference_dir is not configured" in message
    assert "LOCUSBLEND_REFERENCE_DIR" in message
    assert calls == []


def test_plot_with_configured_reference_uses_no_network(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCUSBLEND_REFERENCE_DIR", raising=False)
    api_helpers.patch_reference_layer(monkeypatch, tmp_path, patch_manager=False)
    monkeypatch.setenv("LOCUSBLEND_REFERENCE_DIR", str(tmp_path))
    calls = forbid_network(monkeypatch)

    locusblend.plot(
        api_helpers.make_dataset("one"),
        api_helpers.make_dataset("two"),
        ancestry=EUR,
        mode="standard",
        chrom=CHROM,
        center_bp=api_helpers.CENTER_BP,
    )

    assert calls == []


# ----------------------------------------------------------------------
# removed distribution surface stays removed
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    [
        "install_reference",
        "get_default_reference_dir",
        "ReferenceManifest",
        "ManifestError",
        "DownloadError",
        "ChecksumError",
        "InstallResult",
        "ReferenceInstallError",
    ],
)
def test_distribution_api_is_gone(name):
    assert not hasattr(locusblend, name), name


@pytest.mark.parametrize("module", ["locusblend.fetch", "locusblend.manifest", "locusblend.install"])
def test_distribution_modules_are_gone(module):
    assert importlib.util.find_spec(module) is None, module


def test_package_root_surface_and_no_streamlit():
    for name in (
        "plot",
        "reference_status",
        "ReferenceManager",
        "ReferenceStatus",
        "ReferenceValidation",
        "LocusBlendConfig",
        "LocusBlendResult",
        "IndexVariant",
    ):
        assert hasattr(locusblend, name), name
    assert "streamlit" not in sys.modules
