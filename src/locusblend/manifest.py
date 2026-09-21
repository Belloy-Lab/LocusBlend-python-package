"""Reference bundle manifest: schema, parsing, validation and selection.

A manifest describes the *published* files of a reference bundle so the
installer knows what to download and how to verify it. This module only models,
parses and validates manifests - it contains no URLs, checksums or bundle
versions of its own, and nothing here is downloaded implicitly.

Schema (``schema_version`` 1)::

    {
      "schema_version": 1,
      "bundle_version": "<publisher-defined version string>",
      "genome_build": "GRCh38",
      "files": [
        {
          "resource": "1000g",              # 1000g | gencode | recombination
          "path": "1000g/EUR/<name>.bed",   # relative to the reference directory
          "url": "<https or file URL>",
          "sha256": "<64 hex characters>",
          "size": 123456,                   # optional
          "ancestry": "EUR",                # required for 1000g, absent for shared
          "chrom": "14",                    # required for 1000g (1-22 or X)
          "component": "bed",               # optional label (bed/bim/fam/gtf/bw/...)
          "optional": false                 # optional files only warn on failure
        }
      ]
    }

Shared resources (GENCODE annotation, recombination BigWig) omit ``ancestry``
and are installed once per reference directory, not once per ancestry.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .io import get_supported_chromosomes, normalize_chrom
from .reference import resolve_ancestry

#: Manifest schema versions this package can read.
MANIFEST_SCHEMA_VERSION = 1
SUPPORTED_MANIFEST_SCHEMA_VERSIONS = frozenset({MANIFEST_SCHEMA_VERSION})

#: Genome build spellings accepted in a manifest (normalized to GRCh38).
GENOME_BUILD_ALIASES = MappingProxyType(
    {"grch38": "GRCh38", "hg38": "GRCh38", "grch38/hg38": "GRCh38"}
)
DEFAULT_GENOME_BUILD = "GRCh38"

#: Reference resources LocusBlend knows about.
RESOURCE_1000G = "1000g"
RESOURCE_GENCODE = "gencode"
RESOURCE_RECOMBINATION = "recombination"
KNOWN_RESOURCES = (RESOURCE_1000G, RESOURCE_GENCODE, RESOURCE_RECOMBINATION)
SHARED_RESOURCES = (RESOURCE_GENCODE, RESOURCE_RECOMBINATION)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """Raised when a manifest is missing, malformed or unsafe."""


def _require_mapping(value, what):
    if not isinstance(value, Mapping):
        raise ManifestError(f"{what} must be a JSON object, got {type(value).__name__}.")
    return value


def _require_str(entry, key, what):
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{what} is missing a non-empty {key!r} string.")
    return value.strip()


def validate_relative_path(path) -> str:
    """Validate a manifest destination path and return it in POSIX form.

    Only relative paths inside the reference directory are accepted. Absolute
    paths, Windows drive-qualified paths, UNC paths, ``~`` and any ``..``
    traversal (including backslash variants) raise :class:`ManifestError`.
    """
    if not isinstance(path, str) or not path.strip():
        raise ManifestError("manifest file entry has an empty 'path'.")
    raw = path.strip()
    if "\0" in raw:
        raise ManifestError(f"manifest path contains a NUL byte: {raw!r}")

    posix = raw.replace("\\", "/")

    if raw.startswith("~") or posix.startswith("~"):
        raise ManifestError(
            f"manifest path must be relative to the reference directory (no '~'): {raw!r}"
        )
    if posix.startswith("/"):
        raise ManifestError(f"manifest path must be relative, got absolute path: {raw!r}")
    if posix.startswith("//"):
        raise ManifestError(f"manifest path must not be a UNC/network path: {raw!r}")
    windows = PureWindowsPath(raw)
    if windows.drive or windows.root:
        raise ManifestError(
            f"manifest path must not be drive-qualified: {raw!r}"
        )
    if PurePosixPath(posix).is_absolute():
        raise ManifestError(f"manifest path must be relative: {raw!r}")

    parts = []
    for part in posix.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise ManifestError(f"manifest path must not contain '..': {raw!r}")
        if ":" in part:
            raise ManifestError(f"manifest path contains an invalid ':' segment: {raw!r}")
        parts.append(part)
    if not parts:
        raise ManifestError(f"manifest path resolves to nothing: {raw!r}")
    return "/".join(parts)


def destination_within(reference_root, relative_path) -> Path:
    """Return ``reference_root/relative_path``, verifying it stays inside.

    Both sides are resolved before the containment check so symlinked escapes
    are rejected as well.
    """
    relative = validate_relative_path(relative_path)
    root = Path(reference_root).expanduser().resolve()
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ManifestError(
            f"manifest path escapes the reference directory: {relative_path!r} -> {target}"
        ) from exc
    return target


def normalize_genome_build(value) -> str:
    """Validate and normalize a manifest genome build (GRCh38/hg38)."""
    if value is None:
        return DEFAULT_GENOME_BUILD
    key = str(value).strip().lower()
    if key not in GENOME_BUILD_ALIASES:
        raise ManifestError(
            f"Unsupported genome_build {value!r}. LocusBlend reference bundles are "
            f"{DEFAULT_GENOME_BUILD} only (no liftover is performed)."
        )
    return GENOME_BUILD_ALIASES[key]


def normalize_sha256(value) -> str:
    """Validate and normalize a SHA256 hex digest."""
    if not isinstance(value, str) or not value.strip():
        raise ManifestError("manifest file entry is missing a 'sha256' checksum.")
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest.split(":", 1)[1]
    if not _SHA256_RE.match(digest):
        raise ManifestError(
            f"invalid sha256 checksum {value!r}: expected 64 hexadecimal characters."
        )
    return digest


@dataclass(frozen=True)
class ManifestFile:
    """One file described by a manifest."""

    resource: str
    path: str
    url: str
    sha256: str
    size: Optional[int] = None
    ancestry: Optional[str] = None
    chrom: Optional[str] = None
    component: Optional[str] = None
    optional: bool = False

    @property
    def is_shared(self) -> bool:
        """True for resources shared across ancestries (GENCODE/recombination)."""
        return self.ancestry is None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "resource": self.resource,
            "path": self.path,
            "url": self.url,
            "sha256": self.sha256,
        }
        if self.size is not None:
            data["size"] = self.size
        if self.ancestry is not None:
            data["ancestry"] = self.ancestry
        if self.chrom is not None:
            data["chrom"] = self.chrom
        if self.component is not None:
            data["component"] = self.component
        if self.optional:
            data["optional"] = True
        return data

    def describe(self) -> str:
        scope = "shared" if self.is_shared else f"ancestry {self.ancestry}"
        chrom = f", chr{self.chrom}" if self.chrom else ""
        optional = ", optional" if self.optional else ""
        return f"{self.resource} [{scope}{chrom}{optional}] -> {self.path}"


@dataclass(frozen=True)
class ReferenceManifest:
    """A parsed, validated reference bundle manifest."""

    schema_version: int
    bundle_version: str
    genome_build: str = DEFAULT_GENOME_BUILD
    files: Tuple[ManifestFile, ...] = ()
    source: Optional[str] = None

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def ancestries(self) -> Tuple[str, ...]:
        """Return the ancestries that have per-ancestry (1000g) resources."""
        found = {f.ancestry for f in self.files if f.ancestry is not None}
        return tuple(sorted(found))

    def chromosomes_for_ancestry(self, ancestry) -> Tuple[str, ...]:
        """Return the chromosomes covered for *ancestry* (sorted 1-22 then X)."""
        code = resolve_ancestry(ancestry)
        chroms = {
            normalize_chrom(f.chrom)
            for f in self.files
            if f.ancestry == code and f.chrom is not None
        }
        order = {c: i for i, c in enumerate(get_supported_chromosomes())}
        return tuple(sorted(chroms, key=lambda c: order.get(c, 999)))

    def shared_files(self) -> Tuple[ManifestFile, ...]:
        """Return the files shared across ancestries (GENCODE, recombination)."""
        return tuple(f for f in self.files if f.is_shared)

    def files_for_ancestry(
        self,
        ancestry,
        *,
        chroms: Optional[Iterable[str]] = None,
        include_shared: bool = True,
    ) -> Tuple[ManifestFile, ...]:
        """Select the files to install for one ancestry.

        ``chroms=None`` selects every chromosome present in the manifest for
        that ancestry (the installer requests chr1-22 + X by default).
        ``include_shared`` adds the shared GENCODE/recombination resources.
        """
        code = resolve_ancestry(ancestry)
        wanted = None
        if chroms is not None:
            wanted = {normalize_chrom(chrom) for chrom in chroms}

        selected = []
        for entry in self.files:
            if entry.is_shared:
                if include_shared:
                    selected.append(entry)
                continue
            if entry.ancestry != code:
                continue
            if wanted is not None and entry.chrom is not None and entry.chrom not in wanted:
                continue
            selected.append(entry)
        return tuple(selected)

    def resources(self) -> Tuple[str, ...]:
        """Return the resource names present in the manifest (sorted)."""
        return tuple(sorted({f.resource for f in self.files}))

    def describe(self) -> str:
        lines = [
            f"reference manifest: bundle_version={self.bundle_version!r}, "
            f"genome_build={self.genome_build}, schema_version={self.schema_version}",
            f"  files: {len(self.files)}",
            f"  ancestries: {', '.join(self.ancestries()) or 'none'}",
            f"  resources: {', '.join(self.resources()) or 'none'}",
        ]
        if self.source:
            lines.append(f"  source: {self.source}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "bundle_version": self.bundle_version,
            "genome_build": self.genome_build,
            "files": [f.to_dict() for f in self.files],
        }
        return data


def parse_manifest_file_entry(entry, *, index=0) -> ManifestFile:
    """Validate one ``files`` entry of a manifest."""
    what = f"manifest files[{index}]"
    entry = _require_mapping(entry, what)

    resource = _require_str(entry, "resource", what)
    if resource not in KNOWN_RESOURCES:
        raise ManifestError(
            f"{what} has unsupported resource {resource!r}. Known resources: "
            + ", ".join(KNOWN_RESOURCES)
            + "."
        )

    path = validate_relative_path(_require_str(entry, "path", what))
    url = _require_str(entry, "url", what)
    sha256 = normalize_sha256(entry.get("sha256"))

    size = entry.get("size")
    if size is not None:
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ManifestError(f"{what} has an invalid 'size' (expected positive integer).")

    ancestry = entry.get("ancestry")
    if ancestry is not None:
        if not isinstance(ancestry, str) or not ancestry.strip():
            raise ManifestError(f"{what} has an invalid 'ancestry' value.")
        try:
            ancestry = resolve_ancestry(ancestry)
        except ValueError as exc:
            raise ManifestError(f"{what} has an unsupported 'ancestry': {exc}") from exc

    chrom = entry.get("chrom")
    if chrom is not None:
        if isinstance(chrom, bool) or not isinstance(chrom, (str, int)):
            raise ManifestError(f"{what} has an invalid 'chrom' value.")
        chrom = normalize_chrom(chrom)
        if chrom not in get_supported_chromosomes():
            raise ManifestError(
                f"{what} has unsupported chromosome {entry.get('chrom')!r} "
                "(supported: 1-22 and X)."
            )

    component = entry.get("component")
    if component is not None and not isinstance(component, str):
        raise ManifestError(f"{what} has an invalid 'component' value.")

    optional = entry.get("optional", False)
    if not isinstance(optional, bool):
        raise ManifestError(f"{what} has an invalid 'optional' flag (expected boolean).")

    if resource == RESOURCE_1000G:
        if ancestry is None:
            raise ManifestError(f"{what} (1000g) must declare an 'ancestry'.")
        if chrom is None:
            raise ManifestError(f"{what} (1000g) must declare a 'chrom'.")
    elif ancestry is not None:
        raise ManifestError(
            f"{what} ({resource}) is a shared resource and must not declare an 'ancestry'."
        )

    return ManifestFile(
        resource=resource,
        path=path,
        url=url,
        sha256=sha256,
        size=size,
        ancestry=ancestry,
        chrom=chrom,
        component=component.strip() if isinstance(component, str) else None,
        optional=optional,
    )


def parse_manifest(data, *, source: Optional[str] = None) -> ReferenceManifest:
    """Parse and validate a manifest mapping (or JSON string).

    Raises :class:`ManifestError` with a clear message for malformed input.
    """
    if isinstance(data, (str, bytes, bytearray)):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"manifest is not valid JSON: {exc}") from exc

    data = _require_mapping(data, "manifest")

    schema_version = data.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ManifestError("manifest is missing an integer 'schema_version'.")
    if schema_version not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
        raise ManifestError(
            f"unsupported manifest schema_version {schema_version}; this package "
            "supports: "
            + ", ".join(str(v) for v in sorted(SUPPORTED_MANIFEST_SCHEMA_VERSIONS))
            + "."
        )

    bundle_version = data.get("bundle_version")
    if not isinstance(bundle_version, str) or not bundle_version.strip():
        raise ManifestError("manifest is missing a non-empty 'bundle_version' string.")

    genome_build = normalize_genome_build(data.get("genome_build"))

    raw_files = data.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ManifestError("manifest 'files' must be a non-empty list.")

    files = []
    seen_paths = {}
    for index, entry in enumerate(raw_files):
        parsed = parse_manifest_file_entry(entry, index=index)
        if parsed.path in seen_paths:
            raise ManifestError(
                f"manifest lists the destination path {parsed.path!r} more than once "
                f"(files[{seen_paths[parsed.path]}] and files[{index}])."
            )
        seen_paths[parsed.path] = index
        files.append(parsed)

    return ReferenceManifest(
        schema_version=schema_version,
        bundle_version=bundle_version.strip(),
        genome_build=genome_build,
        files=tuple(files),
        source=source,
    )


def load_manifest(source) -> ReferenceManifest:
    """Load a manifest from an object, mapping, local path or URL.

    Accepted inputs: :class:`ReferenceManifest`, a mapping, a path to a JSON
    file, a ``file://`` URL, or an ``https://`` URL (fetched with the standard
    library). No default/public manifest exists yet - callers must supply one.
    """
    if isinstance(source, ReferenceManifest):
        return source
    if isinstance(source, Mapping):
        return parse_manifest(source)
    if isinstance(source, (str, Path)):
        text, location = _read_manifest_text(source)
        return parse_manifest(text, source=location)
    raise ManifestError(
        f"Unsupported manifest source {type(source).__name__}; pass a path, a URL, "
        "a mapping, or a ReferenceManifest."
    )


def _read_manifest_text(source):
    """Return ``(text, description)`` for a path, file:// URL or https:// URL."""
    raw = str(source)
    lowered = raw.lower()
    if lowered.startswith(("https://", "file://", "http://")):
        from .fetch import fetch_bytes

        if lowered.startswith("http://"):
            raise ManifestError(
                "manifest URLs must use https:// (or file:// for local testing); "
                "plain http:// is not accepted."
            )
        if lowered.startswith("file://"):
            path = _file_url_to_path(raw)
            return _read_local_manifest(path), raw
        data = fetch_bytes(raw, limit=MANIFEST_MAX_BYTES)
        try:
            return data.decode("utf-8"), raw
        except UnicodeDecodeError as exc:
            raise ManifestError(f"manifest at {raw} is not valid UTF-8: {exc}") from exc

    path = Path(raw).expanduser()
    return _read_local_manifest(path), str(path)


def _file_url_to_path(url: str) -> Path:
    from urllib.request import url2pathname
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.netloc not in ("", "localhost"):
        raise ManifestError(f"unsupported file:// manifest host: {parsed.netloc!r}")
    return Path(url2pathname(parsed.path))


def _read_local_manifest(path: Path) -> str:
    if not path.is_file():
        raise ManifestError(f"manifest file not found: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"could not read manifest {path}: {exc}") from exc


#: Upper bound for remotely fetched manifests (they are JSON metadata only).
MANIFEST_MAX_BYTES = 8 * 1024 * 1024

