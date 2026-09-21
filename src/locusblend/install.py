"""Explicit reference-data installation and inspection.

``install_reference()`` is the only code path in LocusBlend that downloads
reference data, and it only runs when the user calls it: :func:`locusblend.plot`
never downloads anything and never calls the installer.

The installed layout is unchanged from the manual layout::

    reference_dir/
    ├── 1000g/<ANCESTRY>/            per-ancestry PLINK panels
    ├── gencode/                     shared GENCODE annotation
    ├── recombination/               shared recombination BigWig (optional)
    └── .locusblend-reference.json   lightweight install marker
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .fetch import (
    DEFAULT_TIMEOUT,
    STATUS_DOWNLOADED,
    STATUS_REPLACED,
    STATUS_SKIPPED,
    ChecksumError,
    DownloadError,
    download_file,
)
from .io import get_supported_chromosomes, normalize_chrom
from .manifest import (
    ManifestError,
    ReferenceManifest,
    destination_within,
    load_manifest,
)
from .paths import (
    REFERENCE_DIR_ENV_VAR,
    SOURCE_MANAGED,
    resolve_reference_dir,
)
from .reference import ReferenceManager, ReferenceValidation, resolve_ancestry

#: Marker file written inside a managed (or manual) reference directory.
REFERENCE_MARKER_FILENAME = ".locusblend-reference.json"
MARKER_SCHEMA_VERSION = 1

#: Chromosomes installed by default: the full ancestry (1-22 + X).
DEFAULT_INSTALL_CHROMS = tuple(get_supported_chromosomes())

STATUS_FAILED = "failed"


class ReferenceInstallError(RuntimeError):
    """Raised when required reference files could not be installed."""


@dataclass(frozen=True)
class FileInstallResult:
    """Outcome of installing one manifest file."""

    resource: str
    path: str
    status: str
    optional: bool = False
    bytes_written: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status != STATUS_FAILED

    def describe(self) -> str:
        suffix = " (optional)" if self.optional else ""
        detail = f" - {self.error}" if self.error else ""
        return f"{self.status}: {self.path}{suffix}{detail}"


@dataclass(frozen=True)
class InstallResult:
    """Result of :func:`install_reference`."""

    reference_dir: str
    ancestry: str
    bundle_version: Optional[str] = None
    files: Tuple[FileInstallResult, ...] = ()
    warnings: Tuple[str, ...] = ()
    marker_path: Optional[str] = None

    @property
    def downloaded(self) -> Tuple[FileInstallResult, ...]:
        return tuple(f for f in self.files if f.status == STATUS_DOWNLOADED)

    @property
    def skipped(self) -> Tuple[FileInstallResult, ...]:
        return tuple(f for f in self.files if f.status == STATUS_SKIPPED)

    @property
    def replaced(self) -> Tuple[FileInstallResult, ...]:
        return tuple(f for f in self.files if f.status == STATUS_REPLACED)

    @property
    def failed(self) -> Tuple[FileInstallResult, ...]:
        return tuple(f for f in self.files if f.status == STATUS_FAILED)

    @property
    def ok(self) -> bool:
        """True when no *required* file failed (optional files only warn)."""
        return not any(not f.optional for f in self.failed)

    def describe(self) -> str:
        lines = [
            f"reference install: ancestry={self.ancestry}, "
            f"bundle_version={self.bundle_version!r}",
            f"  reference_dir: {self.reference_dir}",
            f"  files: {len(self.files)} "
            f"(downloaded {len(self.downloaded)}, skipped {len(self.skipped)}, "
            f"replaced {len(self.replaced)}, failed {len(self.failed)})",
        ]
        if self.marker_path:
            lines.append(f"  marker: {self.marker_path}")
        lines.extend(f"  warning: {w}" for w in self.warnings)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference_dir": self.reference_dir,
            "ancestry": self.ancestry,
            "bundle_version": self.bundle_version,
            "marker_path": self.marker_path,
            "files": [
                {
                    "resource": f.resource,
                    "path": f.path,
                    "status": f.status,
                    "optional": f.optional,
                    "bytes_written": f.bytes_written,
                    "error": f.error,
                }
                for f in self.files
            ],
            "warnings": list(self.warnings),
        }


# ----------------------------------------------------------------------
# install marker
# ----------------------------------------------------------------------
def read_reference_marker(reference_dir) -> Optional[Dict[str, Any]]:
    """Return the install marker of *reference_dir*, or None when unavailable."""
    path = Path(reference_dir) / REFERENCE_MARKER_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _chrom_sort_key(chrom: str) -> int:
    order = {c: i for i, c in enumerate(get_supported_chromosomes())}
    return order.get(normalize_chrom(chrom), 999)


def _present_units(root: Path, files: Iterable) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Return (ancestry -> complete chroms, resource -> present relative paths).

    A chromosome counts as installed only when *every* selected file for that
    ancestry/chromosome pair is present on disk.
    """
    expected: Dict[Tuple[str, str], int] = {}
    present: Dict[Tuple[str, str], int] = {}
    shared: Dict[str, List[str]] = {}

    for entry in files:
        try:
            exists = destination_within(root, entry.path).is_file()
        except ManifestError:
            exists = False
        if entry.ancestry is None:
            if exists:
                shared.setdefault(entry.resource, []).append(entry.path)
            continue
        key = (entry.ancestry, normalize_chrom(entry.chrom))
        expected[key] = expected.get(key, 0) + 1
        if exists:
            present[key] = present.get(key, 0) + 1

    complete: Dict[str, List[str]] = {}
    for code in sorted({key[0] for key in expected}):
        chroms = [
            chrom
            for (candidate, chrom), count in expected.items()
            if candidate == code and count > 0 and present.get((candidate, chrom), 0) == count
        ]
        if chroms:
            complete[code] = sorted(chroms, key=_chrom_sort_key)

    resources = {name: sorted(set(paths)) for name, paths in shared.items()}
    return complete, resources


def _update_reference_marker(root: Path, manifest: ReferenceManifest, selected_files) -> Path:
    """Merge installed units into the marker file and write it atomically.

    Only relative destination paths and resource/chromosome coverage are
    recorded - never machine-specific absolute paths.
    """
    marker_path = Path(root) / REFERENCE_MARKER_FILENAME
    existing = read_reference_marker(root) or {}

    ancestries: Dict[str, set] = {
        code: set(chroms) for code, chroms in (existing.get("ancestries") or {}).items()
    }
    resources: Dict[str, set] = {
        name: set(paths) for name, paths in (existing.get("resources") or {}).items()
    }

    installed_chroms, installed_resources = _present_units(root, selected_files)
    for code, chroms in installed_chroms.items():
        ancestries.setdefault(code, set()).update(chroms)
    for name, paths in installed_resources.items():
        resources.setdefault(name, set()).update(paths)

    marker = {
        "schema_version": MARKER_SCHEMA_VERSION,
        "bundle_version": manifest.bundle_version,
        "genome_build": manifest.genome_build,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ancestries": {
            code: sorted(chroms, key=_chrom_sort_key) for code, chroms in sorted(ancestries.items())
        },
        "resources": {name: sorted(paths) for name, paths in sorted(resources.items())},
    }

    tmp_path = marker_path.with_name(marker_path.name + ".tmp")
    tmp_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, marker_path)
    return marker_path


# ----------------------------------------------------------------------
# installation
# ----------------------------------------------------------------------
def install_reference(
    ancestry="EUR",
    reference_dir=None,
    manifest=None,
    force: bool = False,
    *,
    chroms: Optional[Iterable[str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> InstallResult:
    """Explicitly install reference data for one ancestry.

    Parameters
    ----------
    ancestry:
        One of AFR, AMR, EAS, EUR (default), SAS (case-insensitive). Only this
        ancestry is installed; shared GENCODE/recombination resources are
        installed once per reference directory.
    reference_dir:
        Target reference directory. ``None`` resolves like normal use: the
        ``LOCUSBLEND_REFERENCE_DIR`` environment variable when set, otherwise the
        managed default directory (``locusblend.get_default_reference_dir()``),
        which is created when missing.
    manifest:
        The bundle to install from: a local JSON path, a ``file://``/``https://``
        URL, a mapping, or a :class:`~locusblend.manifest.ReferenceManifest`.
        There is **no default public manifest yet**, so this must be supplied.
    force:
        Re-download even when an existing file already matches its checksum.
    chroms:
        Optional chromosome subset (default: every chromosome in the manifest,
        i.e. chr1-22 + X for published bundles).
    timeout:
        Per-file network timeout in seconds.

    Returns
    -------
    InstallResult

    Raises
    ------
    ManifestError
        Missing/unsafe/malformed manifest, or an ancestry without resources.
    ReferenceInstallError
        A required file could not be downloaded or failed verification.
    """
    code = resolve_ancestry(ancestry)

    if manifest is None:
        raise ManifestError(
            "No public reference manifest is configured yet (this is a "
            "development-stage package). Provide manifest=... with a local JSON "
            "path, a file:// or https:// URL, or a manifest mapping."
        )

    manifest_obj = load_manifest(manifest)

    selected_chroms = None
    if chroms is not None:
        selected_chroms = tuple(normalize_chrom(c) for c in chroms)

    ancestry_files = manifest_obj.files_for_ancestry(
        code, chroms=selected_chroms, include_shared=False
    )
    if not ancestry_files:
        available = ", ".join(manifest_obj.ancestries()) or "none"
        raise ManifestError(
            f"manifest bundle {manifest_obj.bundle_version!r} contains no files for "
            f"ancestry {code}. Ancestries in this manifest: {available}."
        )
    selected = manifest_obj.files_for_ancestry(code, chroms=selected_chroms)

    location = resolve_reference_dir(
        reference_dir, use_managed_default=True, require_existing_managed=False
    )
    root = Path(location.path).expanduser()

    warnings: List[str] = []
    if reference_dir is None and location.source == SOURCE_MANAGED:
        warnings.append(f"Installing into the managed reference directory: {root}")

    results: List[FileInstallResult] = []
    required_failures: List[str] = []

    for entry in selected:
        try:
            destination = destination_within(root, entry.path)
        except ManifestError as exc:
            results.append(
                FileInstallResult(entry.resource, entry.path, STATUS_FAILED, entry.optional, 0, str(exc))
            )
            required_failures.append(str(exc))
            continue

        try:
            outcome = download_file(
                entry.url,
                destination,
                sha256=entry.sha256,
                size=entry.size,
                force=force,
                timeout=timeout,
            )
        except (DownloadError, ChecksumError) as exc:
            message = f"{entry.path}: {exc}"
            results.append(
                FileInstallResult(entry.resource, entry.path, STATUS_FAILED, entry.optional, 0, str(exc))
            )
            if entry.optional:
                warnings.append(f"optional file not installed - {message}")
            else:
                required_failures.append(message)
            continue

        results.append(
            FileInstallResult(
                entry.resource,
                entry.path,
                outcome.status,
                entry.optional,
                outcome.bytes_written,
            )
        )

    marker_path = None
    try:
        marker_path = _update_reference_marker(root, manifest_obj, selected)
    except OSError as exc:
        warnings.append(f"could not update the reference marker: {exc}")

    result = InstallResult(
        reference_dir=str(root),
        ancestry=code,
        bundle_version=manifest_obj.bundle_version,
        files=tuple(results),
        warnings=tuple(warnings),
        marker_path=str(marker_path) if marker_path is not None else None,
    )

    if required_failures:
        raise ReferenceInstallError(
            "Reference installation failed for "
            f"{len(required_failures)} required file(s) in {root}:\n  - "
            + "\n  - ".join(required_failures)
            + "\nNo invalid file was left in place; run install_reference() again "
            "to retry."
        )
    return result


# ----------------------------------------------------------------------
# inspection
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ReferenceStatus:
    """Structured inspection result for a reference directory."""

    reference_dir: Optional[str] = None
    source: Optional[str] = None
    exists: bool = False
    ancestry_chroms: Dict[str, Tuple[str, ...]] = None  # type: ignore[assignment]
    partial_chroms: Dict[str, Tuple[str, ...]] = None  # type: ignore[assignment]
    gencode_files: Tuple[str, ...] = ()
    recombination_files: Tuple[str, ...] = ()
    marker: Optional[Dict[str, Any]] = None
    notes: Tuple[str, ...] = ()
    validation: Optional[ReferenceValidation] = None

    def __post_init__(self):
        if self.ancestry_chroms is None:
            object.__setattr__(self, "ancestry_chroms", {})
        if self.partial_chroms is None:
            object.__setattr__(self, "partial_chroms", {})

    @property
    def installed_ancestries(self) -> Tuple[str, ...]:
        """Ancestries with at least one complete chromosome."""
        return tuple(sorted(self.ancestry_chroms))

    @property
    def gencode_available(self) -> bool:
        return bool(self.gencode_files)

    @property
    def recombination_available(self) -> bool:
        return bool(self.recombination_files)

    def is_chrom_available(self, ancestry, chrom) -> bool:
        """True when all PLINK files for *ancestry*/*chrom* are present."""
        try:
            code = resolve_ancestry(ancestry)
        except ValueError:
            return False
        return normalize_chrom(chrom) in self.ancestry_chroms.get(code, ())

    def describe(self) -> str:
        lines = [
            f"reference status: {self.reference_dir} (source: {self.source})",
            f"  exists: {self.exists}",
            f"  ancestries: {', '.join(self.installed_ancestries) or 'none'}",
            f"  GENCODE: {'available' if self.gencode_available else 'missing'}",
            f"  recombination: "
            f"{'available' if self.recombination_available else 'not installed (optional)'}",
        ]
        if self.marker:
            lines.append(
                f"  marker: bundle_version={self.marker.get('bundle_version')!r}, "
                f"genome_build={self.marker.get('genome_build')!r}"
            )
        else:
            lines.append("  marker: none")
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference_dir": self.reference_dir,
            "source": self.source,
            "exists": self.exists,
            "installed_ancestries": list(self.installed_ancestries),
            "ancestry_chroms": {code: list(chroms) for code, chroms in self.ancestry_chroms.items()},
            "partial_chroms": {code: list(chroms) for code, chroms in self.partial_chroms.items()},
            "gencode": {
                "available": self.gencode_available,
                "files": list(self.gencode_files),
            },
            "recombination": {
                "available": self.recombination_available,
                "optional": True,
                "files": list(self.recombination_files),
            },
            "marker": self.marker,
            "notes": list(self.notes),
            "validation": self.validation.to_dict() if self.validation is not None else None,
        }


def reference_status(reference_dir=None, ancestry=None, chrom=None) -> ReferenceStatus:
    """Inspect a reference directory without reading reference file contents.

    Parameters
    ----------
    reference_dir:
        Directory to inspect; ``None`` resolves like normal use (explicit ->
        ``LOCUSBLEND_REFERENCE_DIR`` -> managed default, which is reported even
        when it does not exist yet).
    ancestry:
        Optional ancestry to validate (raises for unknown codes).
    chrom:
        Optional chromosome to validate together with *ancestry* (1-22 or X).

    Returns
    -------
    ReferenceStatus
        Structured data (``to_dict()``) plus ``describe()`` for humans.
    """
    code = resolve_ancestry(ancestry) if ancestry is not None else None
    requested_chrom = normalize_chrom(chrom) if chrom is not None else None

    location = resolve_reference_dir(
        reference_dir, use_managed_default=True, require_existing_managed=False
    )
    if not location.is_resolved:
        return ReferenceStatus(
            reference_dir=None,
            source=None,
            exists=False,
            notes=(
                "No reference directory configured. Provide reference_dir=..., set "
                f"{REFERENCE_DIR_ENV_VAR}, or install managed reference data with "
                "locusblend.install_reference(ancestry=...).",
            ),
        )

    root = Path(location.path)
    exists = root.is_dir()
    notes: List[str] = []
    ancestry_chroms: Dict[str, Tuple[str, ...]] = {}
    partial_chroms: Dict[str, Tuple[str, ...]] = {}
    gencode_files: Tuple[str, ...] = ()
    recombination_files: Tuple[str, ...] = ()
    marker = read_reference_marker(root) if exists else None
    validation: Optional[ReferenceValidation] = None

    if not exists:
        notes.append(
            f"reference_dir does not exist yet: {root}. Install reference data with "
            "locusblend.install_reference(ancestry=...) or point reference_dir=... at "
            "an existing collection."
        )

    thousand_dir = root / "1000g"
    if exists and thousand_dir.is_dir():
        for child in sorted(thousand_dir.iterdir()):
            if not child.is_dir():
                continue
            try:
                child_code = resolve_ancestry(child.name)
            except ValueError:
                continue
            manager = ReferenceManager(reference_dir=root, ancestry=child_code)
            complete: List[str] = []
            partial: List[str] = []
            for candidate in DEFAULT_INSTALL_CHROMS:
                missing = manager.missing_bfile_paths(candidate)
                if not missing:
                    complete.append(candidate)
                elif len(missing) < len(manager.bfile_suffixes):
                    partial.append(candidate)
            if complete:
                ancestry_chroms[child_code] = tuple(complete)
            if partial:
                partial_chroms[child_code] = tuple(partial)

    gencode_dir = root / "gencode"
    if exists and gencode_dir.is_dir():
        gencode_files = tuple(sorted(p.name for p in gencode_dir.glob("*.gtf.gz")))

    recombination_dir = root / "recombination"
    if exists and recombination_dir.is_dir():
        recombination_files = tuple(sorted(p.name for p in recombination_dir.glob("*.bw")))

    if exists and not ancestry_chroms:
        notes.append(
            f"no complete 1000G ancestry panel found under {thousand_dir}; install "
            "data with locusblend.install_reference(ancestry=...)."
        )
    if exists and gencode_files == ():
        notes.append("no GENCODE annotation (*.gtf.gz) found in the gencode directory.")
    if exists and marker is None:
        notes.append(
            "install marker not found; run locusblend.install_reference(...) to record "
            "installed resources."
        )

    if code is not None:
        if requested_chrom is not None:
            validation = ReferenceManager(reference_dir=root, ancestry=code).validate_for_locus(
                requested_chrom
            )
            if not validation.ok:
                notes.append(
                    f"chromosome {requested_chrom} is not complete for {code}; see validation."
                )
        elif code not in ancestry_chroms:
            notes.append(f"ancestry {code} has no complete chromosome in {root}.")

    return ReferenceStatus(
        reference_dir=str(root),
        source=location.source,
        exists=exists,
        ancestry_chroms=ancestry_chroms,
        partial_chroms=partial_chroms,
        gencode_files=gencode_files,
        recombination_files=recombination_files,
        marker=marker,
        notes=tuple(notes),
        validation=validation,
    )