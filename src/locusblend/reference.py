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
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

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


def default_reference_dir() -> Optional[Path]:
    """Return the reference directory from ``LOCUSBLEND_REFERENCE_DIR``, if set."""
    value = os.environ.get(REFERENCE_DIR_ENV_VAR, "").strip()
    return Path(value).expanduser() if value else None


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

    def __init__(self, reference_dir=None, ancestry=INTERNAL_1000G_DEFAULT_ANCESTRY):
        if reference_dir is None:
            reference_dir = default_reference_dir()
        self.reference_dir = (
            Path(reference_dir).expanduser() if reference_dir is not None else None
        )
        self.ancestry = normalize_internal_1000g_ancestry(ancestry)

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
    to ``LOCUSBLEND_REFERENCE_DIR``.
    """
    return ReferenceManager(reference_dir=reference_dir, ancestry=ancestry).bfile_prefix(chrom)


def get_internal_bfile_prefix_for_chrom(
    chrom,
    ancestry=INTERNAL_1000G_DEFAULT_ANCESTRY,
    reference_dir=None,
):
    """Return the 1000G bfile prefix for *chrom*, verifying .bed/.bim/.fam."""
    return ReferenceManager(reference_dir=reference_dir, ancestry=ancestry).get_bfile_prefix(chrom)


def get_gtf_path_for_chrom(chrom, reference_dir=None):
    """Return the GENCODE GTF for *chrom*, verifying it exists."""
    return ReferenceManager(reference_dir=reference_dir).get_gtf_path(chrom)


def default_recombination_bw_path() -> Optional[str]:
    """Best-effort default recombination BigWig path (None when unset/missing).

    Used by ``plotting.get_plotly_locus_py`` when no explicit ``bw_path`` is
    supplied.
    """
    try:
        return ReferenceManager().get_recombination_bw_path()
    except ValueError:
        return None

