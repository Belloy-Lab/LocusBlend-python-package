"""Release-prep checks: reference-preparation documentation and helper script.

These tests keep the written workflow, the shell helper and
``ReferenceManager``'s file-name convention from drifting apart. The bash-based
comparison is skipped when no usable bash interpreter is available.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from locusblend.reference import ReferenceManager

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs" / "reference_preparation.md"
SCRIPT = REPO_ROOT / "scripts" / "prepare_1000g_reference.sh"

ANCESTRIES = ("AFR", "AMR", "EAS", "EUR", "SAS")
SAMPLE_CHROMS = ("1", "14", "22", "X")

VERIFIED_FLAGS = (
    "--pfile",
    "vzs",
    "--keep",
    "--remove",
    "--chr",
    "--set-missing-var-ids '@:#$r:$a'",
    "--new-id-max-allele-len 1000",
    "--make-bed",
    "--out",
)


def _find_usable_bash():
    """Return a bash executable that actually works, or None."""
    candidates = [
        os.environ.get("LOCUSBLEND_BASH"),
        shutil.which("bash"),
        r"C:\Program Files\Git\bin\bash.exe",
        "/bin/bash",
        "/usr/bin/bash",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if not os.path.isfile(candidate) and not shutil.which(candidate):
            continue
        try:
            probe = subprocess.run(
                [candidate, "-c", "echo ok"], capture_output=True, text=True, timeout=60
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0 and "ok" in probe.stdout:
            return candidate
    return None


def test_preparation_docs_and_script_exist():
    assert DOCS.is_file(), DOCS
    assert SCRIPT.is_file(), SCRIPT


def test_docs_document_the_verified_workflow():
    text = DOCS.read_text(encoding="utf-8")

    # upstream input + citation
    assert "1000 Genomes" in text
    assert "GRCh38" in text
    assert "PLINK2" in text
    assert "3,202 samples" in text
    assert "Byrska-Bishop" in text
    assert "PMID: 36055201" in text

    # policy: no redistribution, no URLs
    assert "does **not** download" in text
    assert "http://" not in text and "https://" not in text
    assert "zenodo" not in text.lower()

    # layout
    for entry in ("1000g/", "AFR/", "AMR/", "EAS/", "EUR/", "SAS/", "gencode/", "recombination/"):
        assert entry in text, entry
    assert "1-22 and X" in text
    assert "chX" in text
    assert "ch23" in text  # documented as the historical name that must not be used

    # verified PLINK2 command and its constraints
    for flag in (
        "--pfile",
        "vzs",
        "--keep",
        "--remove",
        "--chr",
        "--set-missing-var-ids '@:#$r:$a'",
        "--new-id-max-allele-len 1000",
        "--make-bed",
        "--out",
    ):
        assert flag in text, flag
    assert "--rm-dup force-first" in text  # explicitly ruled out
    assert "optional" in text  # recombination + related-sample removal file
    assert "QC" in text and "sample-selection policy" in text


def test_script_uses_the_verified_command_and_naming():
    text = SCRIPT.read_text(encoding="utf-8")

    assert text.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in text
    assert "command -v plink2" in text
    assert "ANCESTRIES=(AFR AMR EAS EUR SAS)" in text
    assert "CHROMOSOMES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X)" in text
    assert "src/locusblend/reference.py" in text  # naming template source of truth

    for flag in VERIFIED_FLAGS:
        assert flag in text, flag

    # constraints: the historical ch23 name may only appear in the header comment,
    # never in the code that builds output names
    assert "--rm-dup" not in text
    code_lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    assert all("ch23" not in line for line in code_lines)
    assert "http://" not in text and "https://" not in text
    assert "1000g/${anc}" in text  # writes into OUTPUT_ROOT/1000g/<ANCESTRY>/
    assert "AFR.txt" in text and "SAS.txt" in text  # keep-list naming documented


def test_script_cli_behaviour():
    bash = _find_usable_bash()
    if bash is None:
        pytest.skip("no usable bash interpreter available")

    help_run = subprocess.run(
        [bash, SCRIPT.as_posix(), "--help"], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert help_run.returncode == 0
    assert "Usage:" in help_run.stdout

    no_args = subprocess.run(
        [bash, SCRIPT.as_posix()], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert no_args.returncode != 0
    assert "Usage:" in (no_args.stdout + no_args.stderr)


def test_script_output_template_matches_reference_manager():
    """The script's derived PLINK prefix must match ReferenceManager exactly."""
    bash = _find_usable_bash()
    if bash is None:
        pytest.skip("no usable bash interpreter available")

    run = subprocess.run(
        [bash, SCRIPT.as_posix(), "--print-template"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert run.returncode == 0, run.stderr

    match = re.search(r"^bfile_prefix_template: (.+)$", run.stdout, flags=re.MULTILINE)
    assert match, run.stdout
    template = match.group(1).strip()

    for code in ANCESTRIES:
        manager = ReferenceManager(reference_dir=REPO_ROOT, ancestry=code)
        for chrom in SAMPLE_CHROMS:
            expected = manager.bfile_prefix(chrom).name
            actual = template.replace("{ancestry}", code).replace("{chrom}", chrom)
            assert actual == expected, (code, chrom, actual, expected)

    # chromosome X is written as chX, never the historical ch23
    assert "chX" in run.stdout
    assert "ch23" not in run.stdout