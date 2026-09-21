"""Central reference-data path resolution for LocusBlend.

The package never bundles 1000 Genomes, GENCODE or recombination data, and the
rest of the package must not build reference paths by hand. Every reference
path goes through :class:`ReferenceManager`.

Expected layout (``reference_dir`` lives outside this Git repository)::

    reference_dir/
    ├── 1000g/
    │   ├── AFR/
    │   ├── AMR/
    │   ├── EAS/
    │   ├── EUR/
    │   └── SAS/
    ├── gencode/
    └── recombination/

The baseline ``app.2.8.12.py`` kept these files in a single ``data/`` directory
next to the script, using the file names in the templates below. The same names
are used inside the ancestry / ``gencode`` / ``recombination`` sub-directories,
so an existing reference collection only has to be moved, not renamed. The
templates (and the sub-directory names) are class attributes so the naming
convention can be adjusted centrally later.

No remote downloading is implemented in this pass: files that are missing are
reported with the same messages the baseline used.

``ReferenceManager`` also provides a cheap preflight:
``manager.validate_for_locus(chrom)`` reports missing configuration, missing
PLINK chromosome files and a missing GENCODE annotation as errors, while a
missing recombination BigWig is only a warning (it is optional). Ancestry codes
are case-insensitive, but an unrecognized code raises ``ValueError`` instead of
silently running EUR.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from .io import normalize_chrom

# Ancestry codes and labels, verbatim from app.2.8.12.py.
INTERNAL_1000G_ANCESTRIES = {
    "AFR": "African / African ancestry",
    "AMR": "Admixed American ancestry",
    "EAS": "East Asian ancestry",
    "EUR": "European ancestry",
    "SAS": "South Asian ancestry",
}
INTERNAL_1000G_DEFAULT_ANCESTRY = "EUR"
INTERNAL_1000G_ANCESTRY_OPTIONS = list(INTERNAL_1000G_ANCESTRIES.keys())

# Public API view: the five supported 1000G super-population codes.
SUPPORTED_ANCESTRIES = tuple(INTERNAL_1000G_ANCESTRY_OPTIONS)

# Environment variables used to locate external data and executables.
REFERENCE_DIR_ENV_VAR = "LOCUSBLEND_REFERENCE_DIR"
PLINK_ENV_VAR = "LOCUSBLEND_PLINK"


def normalize_internal_1000g_ancestry(value):
    """Return a supported 1000G super-population code, defaulting to EUR.

    Verbatim from ``app.2.8.12.py``.
    """
    ancestry = str(value or INTERNAL_1000G_DEFAULT_ANCESTRY).strip().upper()
    return ancestry if ancestry in INTERNAL_1000G_ANCESTRIES else INTERNAL_1000G_DEFAULT_ANCESTRY


def format_internal_1000g_ancestry_option(ancestry):
    """Return the sidebar option label, e.g. ``EUR - European ancestry``.

    Verbatim from ``app.2.8.12.py``.
    """
    ancestry = normalize_internal_1000g_ancestry(ancestry)
    label = INTERNAL_1000G_ANCESTRIES[ancestry].split(" / ", 1)[0]
    return f"{ancestry} - {label}"


def resolve_ancestry(value) -> str:
    """Return a supported ancestry code, raising for unrecognized values.

    Case is normalized (``"eur"`` and ``"EUr"`` both give ``"EUR"``), but an
    unknown code such as ``"ABC"`` raises ``ValueError`` instead of silently
    falling back to EUR. Used by :class:`ReferenceManager` for public-facing
    input; the baseline-parity helper
    :func:`normalize_internal_1000g_ancestry` keeps its silent fallback for
    internal/app compatibility.
    """
    code = str(value or "").strip().upper()
    if code not in INTERNAL_1000G_ANCESTRIES:
        raise ValueError(
            f"Unsupported ancestry {value!r}. Supported ancestries are: "
            + ", ".join(SUPPORTED_ANCESTRIES)
            + "."
        )
    return code


def default_reference_dir() -> Optional[Path]:
    """Return the reference directory from ``LOCUSBLEND_REFERENCE_DIR``, if set."""
    value = os.environ.get(REFERENCE_DIR_ENV_VAR, "").strip()
    return Path(value).expanduser() if value else None


@dataclass(frozen=True)
class ReferenceValidation:
    """Result of :meth:`ReferenceManager.validate` / ``validate_for_locus``.

    ``config_errors`` are configuration problems (unset or non-existent
    ``reference_dir``); ``file_errors`` are missing required reference files.
    Missing optional files (the recombination BigWig) only produce ``warnings``.
    """

    reference_dir: Optional[str] = None
    ancestry: Optional[str] = None
    chrom: Optional[str] = None
    config_errors: Tuple[str, ...] = ()
    file_errors: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    checked_paths: Dict[str, Optional[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when there are no configuration or required-file errors."""
        return not self.config_errors and not self.file_errors

    @property
    def errors(self) -> Tuple[str, ...]:
        """All errors (configuration first)."""
        return tuple(self.config_errors) + tuple(self.file_errors)

    def _message(self, headline: str) -> str:
        lines = [
            f"{headline} (ancestry={self.ancestry!r}, chromosome={self.chrom!r}, "
            f"reference_dir={self.reference_dir!r}):"
        ]
        lines.extend(f"  - {item}" for item in self.errors)
        lines.append(
            "Reference data live outside the package: pass reference_dir=... or set "
            f"{REFERENCE_DIR_ENV_VAR}. See the README for the expected directory layout "
            "(1000g/<ancestry>, gencode, recombination)."
        )
        return "\n".join(lines)

    def raise_for_errors(self) -> None:
        """Raise the matching error for the recorded problems, if any.

        Configuration problems raise ``ValueError``; missing required reference
        files raise ``FileNotFoundError`` (matching the individual getters).
        """
        if self.config_errors:
            raise ValueError(self._message("Incomplete LocusBlend reference configuration"))
        if self.file_errors:
            raise FileNotFoundError(self._message("Missing LocusBlend reference data"))

    def describe(self) -> str:
        """Return a short human-readable report (for logging/diagnostics)."""
        status = "ok" if self.ok else "incomplete"
        lines = [
            f"reference validation: {status}",
            f"  reference_dir: {self.reference_dir}",
            f"  ancestry: {self.ancestry}",
            f"  chromosome: {self.chrom}",
        ]
        lines.extend(f"  error: {item}" for item in self.errors)
        lines.extend(f"  warning: {item}" for item in self.warnings)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-friendly view of the validation result."""
        return {
            "ok": self.ok,
            "reference_dir": self.reference_dir,
            "ancestry": self.ancestry,
            "chromosome": self.chrom,
            "config_errors": list(self.config_errors),
            "file_errors": list(self.file_errors),
            "warnings": list(self.warnings),
            "checked_paths": dict(self.checked_paths),
        }


class ReferenceManager:
    """Resolve reference files under an external ``reference_dir``.

    Parameters
    ----------
    reference_dir:
        Root of the external reference collection. When omitted, the
        ``LOCUSBLEND_REFERENCE_DIR`` environment variable is used; if that is
        unset too, path lookups raise ``ValueError``.
    ancestry:
        One of ``AFR``, ``AMR``, ``EAS``, ``EUR`` (default), ``SAS``. Unknown
        values fall back to EUR, exactly like
        :func:`normalize_internal_1000g_ancestry` in the baseline app.

    Example
    -------
    >>> ref = ReferenceManager(reference_dir="D:/locusblend_reference", ancestry="EUR")
    >>> ref.bfile_prefix("14")           # doctest: +SKIP
    PosixPath('D:/locusblend_reference/1000g/EUR/1000g_EUR_hg38_..._ch14')
    """

    # ---- naming convention (single place to adjust later) ----
    bfile_prefix_template = (
        "1000g_{ancestry}_hg38_high_coverage_Illumina.filtered."
        "SNV_INDEL_SV_phased_panel_include_MHC_ch{chrom}"
    )
    gtf_chrom_template = "gencode.v49.annotation.chr{chrom}.gtf.gz"
    gtf_genome_template = "gencode.v49.annotation.gtf.gz"
    recombination_bw_template = "recomb1000GAvg.bw"

    # ---- layout (single place to adjust later) ----
    ancestry_subdir = "1000g"
    gencode_subdir = "gencode"
    recombination_subdir = "recombination"

    bfile_suffixes = (".bed", ".bim", ".fam")

    def __init__(
        self,
        reference_dir=None,
        ancestry=INTERNAL_1000G_DEFAULT_ANCESTRY,
        *,
        strict_ancestry=True,
    ):
        """Create a reference manager.

        ``ancestry`` is case-insensitive; an unrecognized code raises
        ``ValueError`` (``strict_ancestry=True``, the public default) so typos
        cannot silently run EUR. Pass ``strict_ancestry=False`` to keep the
        baseline helper's silent EUR fallback (used by the module-level
        compatibility wrappers below).
        """
        if reference_dir is None:
            reference_dir = default_reference_dir()
        self.reference_dir = (
            Path(reference_dir).expanduser() if reference_dir is not None else None
        )
        self.ancestry = (
            resolve_ancestry(ancestry)
            if strict_ancestry
            else normalize_internal_1000g_ancestry(ancestry)
        )

    def __repr__(self) -> str:
        return (
            f"ReferenceManager(reference_dir={str(self.reference_dir)!r}, "
            f"ancestry={self.ancestry!r})"
        )

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------
    @property
    def has_reference_dir(self) -> bool:
        """True when a reference directory is configured."""
        return self.reference_dir is not None

    def _require_reference_dir(self) -> Path:
        if self.reference_dir is None:
            raise ValueError(
                "No reference directory configured. Pass reference_dir=... or set "
                f"the {REFERENCE_DIR_ENV_VAR} environment variable."
            )
        return self.reference_dir

    def dir_for(self, *parts) -> Path:
        """Return ``reference_dir`` joined with *parts* (no existence check)."""
        return self._require_reference_dir().joinpath(*[str(p) for p in parts])

    @property
    def thousand_genomes_dir(self) -> Path:
        """``reference_dir/1000g`` (all ancestries)."""
        return self.dir_for(self.ancestry_subdir)

    @property
    def ancestry_dir(self) -> Path:
        """``reference_dir/1000g/<ancestry>`` for the active ancestry."""
        return self.dir_for(self.ancestry_subdir, self.ancestry)

    @property
    def gencode_dir(self) -> Path:
        """``reference_dir/gencode``."""
        return self.dir_for(self.gencode_subdir)

    @property
    def recombination_dir(self) -> Path:
        """``reference_dir/recombination``."""
        return self.dir_for(self.recombination_subdir)

    # ------------------------------------------------------------------
    # 1000G PLINK bfiles
    # ------------------------------------------------------------------
    def bfile_prefix(self, chrom) -> Path:
        """Chromosome-specific PLINK bfile prefix (no existence check)."""
        chrom = normalize_chrom(chrom)
        name = self.bfile_prefix_template.format(ancestry=self.ancestry, chrom=chrom)
        return self.ancestry_dir / name

    def missing_bfile_paths(self, chrom):
        """Return the missing ``.bed`` / ``.bim`` / ``.fam`` paths for *chrom*."""
        prefix = str(self.bfile_prefix(chrom))
        return [prefix + suffix for suffix in self.bfile_suffixes if not os.path.exists(prefix + suffix)]

    def get_bfile_prefix(self, chrom) -> str:
        """Return the chromosome-specific bfile prefix, verifying all three files.

        Error messages are preserved from
        ``get_internal_bfile_prefix_for_chrom`` in ``app.2.8.12.py``, including
        the chromosome X hint.
        """
        chrom = normalize_chrom(chrom)
        prefix = str(self.bfile_prefix(chrom))
        missing = self.missing_bfile_paths(chrom)
        if missing:
            if chrom == "X":
                message = (
                    f"Internal 1000G {self.ancestry} reference files for chromosome X were not found. "
                    "Select an ancestry with chrX support or upload your own LD reference."
                )
            else:
                message = (
                    f"Internal 1000G {self.ancestry} reference files for chromosome {chrom} "
                    "were not found."
                )
            raise FileNotFoundError(message + "\n" + "\n".join(f"  {m}" for m in missing))
        return prefix

    # ------------------------------------------------------------------
    # GENCODE annotation
    # ------------------------------------------------------------------
    def gtf_path(self, chrom) -> Path:
        """Chromosome-specific GENCODE GTF path (no existence check)."""
        chrom = normalize_chrom(chrom)
        return self.gencode_dir / self.gtf_chrom_template.format(chrom=chrom)

    def genome_gtf_path(self) -> Path:
        """Whole-genome GENCODE GTF path (no existence check)."""
        return self.gencode_dir / self.gtf_genome_template

    def get_gtf_path(self, chrom) -> str:
        """Return the GENCODE GTF for *chrom*, with the baseline's fallbacks.

        Chromosome X falls back to the whole-genome annotation file when the
        chrX file is absent, mirroring ``get_gtf_path_for_chrom`` in
        ``app.2.8.12.py``.
        """
        chrom = normalize_chrom(chrom)
        path = self.gtf_path(chrom)
        if chrom == "X" and not path.is_file():
            full_path = self.genome_gtf_path()
            if full_path.is_file():
                return str(full_path)
            raise FileNotFoundError(
                f"Missing GENCODE annotation for chromosome X. Add "
                f"{path} or {full_path}."
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing GENCODE annotation for chromosome {chrom}: {path}"
            )
        return str(path)

    # ------------------------------------------------------------------
    # recombination-rate track
    # ------------------------------------------------------------------
    def recombination_bw_path(self) -> Path:
        """Recombination BigWig path (no existence check)."""
        return self.recombination_dir / self.recombination_bw_template

    def get_recombination_bw_path(self, required: bool = False) -> Optional[str]:
        """Return the recombination BigWig path, or None when it is missing.

        With ``required=True`` a missing file raises ``FileNotFoundError``.
        """
        path = self.recombination_bw_path()
        if path.is_file():
            return str(path)
        if required:
            raise FileNotFoundError(f"Missing recombination BigWig: {path}")
        return None

    # ------------------------------------------------------------------
    # validation / preflight
    # ------------------------------------------------------------------
    def _reference_dir_errors(self):
        errors = []
        if self.reference_dir is None:
            errors.append(
                "reference_dir is not configured (pass reference_dir=... or set "
                f"{REFERENCE_DIR_ENV_VAR})."
            )
        elif not self.reference_dir.exists():
            errors.append(f"reference_dir does not exist: {self.reference_dir}")
        elif not self.reference_dir.is_dir():
            errors.append(f"reference_dir is not a directory: {self.reference_dir}")
        return errors

    def _ancestry_dir_errors(self, checked_paths):
        errors = []
        ancestry_dir = self.ancestry_dir
        checked_paths["ancestry_dir"] = str(ancestry_dir)
        if not ancestry_dir.is_dir():
            errors.append(
                f"1000G {self.ancestry} reference directory is missing: {ancestry_dir} "
                f"(expected reference_dir/{self.ancestry_subdir}/{self.ancestry})"
            )
        return errors

    def _bfile_errors(self, chrom, checked_paths):
        errors = []
        prefix = self.bfile_prefix(chrom)
        checked_paths["bfile_prefix"] = str(prefix)
        missing = self.missing_bfile_paths(chrom)
        if missing:
            errors.append(
                f"1000G {self.ancestry} PLINK reference files for chromosome {chrom} are "
                "missing: " + ", ".join(missing)
            )
        return errors

    def _gtf_errors(self, chrom, checked_paths):
        errors = []
        path = self.gtf_path(chrom)
        checked_paths["gencode"] = str(path)
        if path.is_file():
            return errors
        if chrom == "X":
            full_path = self.genome_gtf_path()
            checked_paths["gencode_genome"] = str(full_path)
            if full_path.is_file():
                return errors
            errors.append(
                f"GENCODE annotation for chromosome X is missing: {path} or {full_path}"
            )
        else:
            errors.append(f"GENCODE annotation for chromosome {chrom} is missing: {path}")
        return errors

    def _recombination_warning(self, checked_paths):
        path = self.recombination_bw_path()
        checked_paths["recombination"] = str(path)
        if path.is_file():
            return None
        return (
            f"Recombination BigWig not found at {path}; "
            "the recombination overlay is skipped."
        )

    def validate_for_locus(self, chrom, *, require_recombination=False) -> ReferenceValidation:
        """Validate everything needed to plot one locus on *chrom*.

        Only cheap path checks are performed (no file reads, no scanning of
        other chromosomes):

        * ``reference_dir`` is configured and exists
        * the selected ancestry directory exists
        * the chromosome PLINK .bed/.bim/.fam files exist (required)
        * the chromosome GENCODE annotation exists (required; chromosome X may
          fall back to the whole-genome annotation file)
        * the recombination BigWig exists (OPTIONAL - a missing file only adds
          a warning unless ``require_recombination=True``)
        """
        chrom = normalize_chrom(chrom)
        config_errors = []
        file_errors = []
        warnings = []
        checked_paths = {}

        config_errors.extend(self._reference_dir_errors())
        if not config_errors:
            file_errors.extend(self._ancestry_dir_errors(checked_paths))
            file_errors.extend(self._bfile_errors(chrom, checked_paths))
            file_errors.extend(self._gtf_errors(chrom, checked_paths))
            warning = self._recombination_warning(checked_paths)
            if warning:
                (file_errors if require_recombination else warnings).append(warning)

        return ReferenceValidation(
            reference_dir=None if self.reference_dir is None else str(self.reference_dir),
            ancestry=self.ancestry,
            chrom=chrom,
            config_errors=tuple(config_errors),
            file_errors=tuple(file_errors),
            warnings=tuple(warnings),
            checked_paths=checked_paths,
        )

    def validate(self, chrom=None, *, require_recombination=False) -> ReferenceValidation:
        """Validate the reference layout (see :meth:`validate_for_locus`).

        Without *chrom* only the chromosome-independent parts are checked
        (``reference_dir``, the ancestry directory and the ``gencode``
        directory), which is useful as an early preflight before the locus is
        known. The recombination BigWig is always optional.
        """
        if chrom is not None and str(chrom).strip() != "":
            return self.validate_for_locus(chrom, require_recombination=require_recombination)

        config_errors = []
        file_errors = []
        warnings = []
        checked_paths = {}

        config_errors.extend(self._reference_dir_errors())
        if not config_errors:
            file_errors.extend(self._ancestry_dir_errors(checked_paths))
            gencode_dir = self.gencode_dir
            checked_paths["gencode_dir"] = str(gencode_dir)
            if not gencode_dir.is_dir():
                file_errors.append(f"GENCODE directory is missing: {gencode_dir}")
            warning = self._recombination_warning(checked_paths)
            if warning:
                (file_errors if require_recombination else warnings).append(warning)

        return ReferenceValidation(
            reference_dir=None if self.reference_dir is None else str(self.reference_dir),
            ancestry=self.ancestry,
            chrom=None,
            config_errors=tuple(config_errors),
            file_errors=tuple(file_errors),
            warnings=tuple(warnings),
            checked_paths=checked_paths,
        )

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def describe(self) -> str:
        """Return a short human-readable summary of the resolved layout."""
        root = str(self.reference_dir) if self.reference_dir is not None else "<unset>"
        lines = [
            f"reference_dir: {root}",
            f"ancestry: {self.ancestry}",
            f"1000G: {self.ancestry_subdir}/{self.ancestry}",
            f"GENCODE: {self.gencode_subdir}",
            f"recombination: {self.recombination_subdir}",
        ]
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Module-level helpers preserving the baseline function names/signatures.
# ----------------------------------------------------------------------
def get_internal_1000g_prefix(
    chrom,
    ancestry=INTERNAL_1000G_DEFAULT_ANCESTRY,
    reference_dir=None,
):
    """Return the chromosome-specific 1000G bfile prefix (no existence check).

    Baseline name/signature preserved; ``reference_dir`` is new and falls back
    to ``LOCUSBLEND_REFERENCE_DIR``. Keeps the baseline's silent EUR fallback
    for unrecognized ancestries (``strict_ancestry=False``).
    """
    return ReferenceManager(
        reference_dir=reference_dir, ancestry=ancestry, strict_ancestry=False
    ).bfile_prefix(chrom)


def get_internal_bfile_prefix_for_chrom(
    chrom,
    ancestry=INTERNAL_1000G_DEFAULT_ANCESTRY,
    reference_dir=None,
):
    """Return the 1000G bfile prefix for *chrom*, verifying .bed/.bim/.fam.

    Keeps the baseline's silent EUR fallback for unrecognized ancestries.
    """
    return ReferenceManager(
        reference_dir=reference_dir, ancestry=ancestry, strict_ancestry=False
    ).get_bfile_prefix(chrom)


def get_gtf_path_for_chrom(chrom, reference_dir=None):
    """Return the GENCODE GTF for *chrom*, verifying it exists."""
    return ReferenceManager(reference_dir=reference_dir, strict_ancestry=False).get_gtf_path(chrom)


def default_recombination_bw_path() -> Optional[str]:
    """Best-effort default recombination BigWig path (None when unset/missing).

    Used by ``plotting.get_plotly_locus_py`` when no explicit ``bw_path`` is
    supplied.
    """
    try:
        return ReferenceManager().get_recombination_bw_path()
    except ValueError:
        return None
