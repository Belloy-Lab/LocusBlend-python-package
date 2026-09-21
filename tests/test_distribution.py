"""Pass 4 tests: reference distribution (managed dirs, manifests, downloads).

Everything here is local: tiny files in ``tmp_path`` served through ``file://``
URLs (or monkeypatched helpers). No test requires internet access, and no test
touches the developer's real managed cache directory.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

import locusblend
from locusblend import fetch, install, manifest as manifest_module, paths
from locusblend.install import (
    REFERENCE_MARKER_FILENAME,
    ReferenceInstallError,
    install_reference,
    reference_status,
)
from locusblend.manifest import (
    ManifestError,
    destination_within,
    load_manifest,
    normalize_genome_build,
    normalize_sha256,
    parse_manifest,
    validate_relative_path,
)
from locusblend.reference import ReferenceManager

import test_api as api_helpers

EUR = "EUR"
CHROM = "14"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def write_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_entry(resource, path, url, data, *, ancestry=None, chrom=None, component=None, optional=False, sha256=None, size=None):
    entry = {
        "resource": resource,
        "path": path,
        "url": url,
        "sha256": sha256 if sha256 is not None else sha256_of(data),
    }
    if size is None:
        entry["size"] = len(data)
    else:
        entry["size"] = size
    if ancestry is not None:
        entry["ancestry"] = ancestry
    if chrom is not None:
        entry["chrom"] = chrom
    if component is not None:
        entry["component"] = component
    if optional:
        entry["optional"] = True
    return entry


def published_bundle(
    tmp_path,
    *,
    ancestries=("EUR",),
    chroms=(CHROM,),
    shared=("gencode", "recombination"),
    optional_recombination=True,
    bundle_version="test-bundle-1",
    corrupt_sha=False,
    break_recombination_url=False,
):
    """Create a tiny published bundle plus a matching manifest dictionary."""
    published = tmp_path / "published"
    root = tmp_path / "reference"
    files = []

    for code in ancestries:
        manager = ReferenceManager(reference_dir=root, ancestry=code)
        for chrom in chroms:
            for suffix, component in ((".bed", "bed"), (".bim", "bim"), (".fam", "fam")):
                target = Path(str(manager.bfile_prefix(chrom)) + suffix)
                relative = target.relative_to(root).as_posix()
                data = f"{code}-chr{chrom}-{component}".encode()
                source = write_bytes(published / relative, data)
                digest = sha256_of(data)
                if corrupt_sha and code == "EUR" and component == "bed":
                    digest = "0" * 64
                files.append(
                    file_entry(
                        "1000g",
                        relative,
                        source.as_uri(),
                        data,
                        ancestry=code,
                        chrom=chrom,
                        component=component,
                        sha256=digest,
                    )
                )

    if "gencode" in shared:
        relative = f"gencode/gencode.v49.annotation.chr{chroms[0]}.gtf.gz"
        data = b"gencode-annotation"
        source = write_bytes(published / relative, data)
        files.append(file_entry("gencode", relative, source.as_uri(), data, component="gtf"))

    if "recombination" in shared:
        relative = "recombination/recomb1000GAvg.bw"
        data = b"recombination-track"
        source = write_bytes(published / relative, data)
        url = "https://example.invalid/not-downloaded.bw" if break_recombination_url else source.as_uri()
        files.append(
            file_entry(
                "recombination",
                relative,
                url,
                data,
                component="bw",
                optional=optional_recombination,
            )
        )

    manifest = {
        "schema_version": 1,
        "bundle_version": bundle_version,
        "genome_build": "GRCh38",
        "files": files,
    }
    manifest_path = tmp_path / "reference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest, manifest_path, root, published


def patch_managed_dir(monkeypatch, path):
    monkeypatch.setattr(paths, "get_default_reference_dir", lambda: Path(path))


def forbid_network(monkeypatch):
    """Replace every download entry point with a recording failure."""
    calls = []

    def boom(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("network/download must not be triggered here")

    monkeypatch.setattr(fetch, "download_file", boom)
    monkeypatch.setattr(fetch, "open_url", boom)
    monkeypatch.setattr(fetch, "fetch_bytes", boom)
    monkeypatch.setattr(install, "download_file", boom)
    return calls


# ----------------------------------------------------------------------
# 1-5. managed directory + resolution priority
# ----------------------------------------------------------------------
def test_managed_dir_uses_platformdirs_cache_root(monkeypatch, tmp_path):
    monkeypatch.setattr(paths.platformdirs, "user_cache_dir", lambda *a, **k: str(tmp_path))
    managed = paths.get_default_reference_dir()
    assert managed == tmp_path / "references"
    assert managed.is_absolute()


def test_managed_dir_default_is_absolute():
    managed = locusblend.get_default_reference_dir()
    assert managed.is_absolute()
    assert managed.name == "references"
    assert "locusblend" in str(managed).lower()


def test_explicit_reference_dir_wins(monkeypatch, tmp_path):
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("LOCUSBLEND_REFERENCE_DIR", str(tmp_path / "env"))
    patch_managed_dir(monkeypatch, tmp_path / "managed")

    location = paths.resolve_reference_dir(explicit)
    assert location.path == explicit
    assert location.source == "explicit"


def test_env_overrides_managed_default(monkeypatch, tmp_path):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("LOCUSBLEND_REFERENCE_DIR", str(env_dir))
    patch_managed_dir(monkeypatch, managed)

    location = paths.resolve_reference_dir(None)
    assert location.path == env_dir
    assert location.source == "env"


def test_managed_default_used_only_when_present(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCUSBLEND_REFERENCE_DIR", raising=False)
    managed = tmp_path / "managed"
    patch_managed_dir(monkeypatch, managed)

    missing = paths.resolve_reference_dir(None)
    assert missing.path is None and missing.source is None

    managed.mkdir()
    present = paths.resolve_reference_dir(None)
    assert present.path == managed and present.source == "managed"

    # ReferenceManager applies the same priority and records the source
    assert ReferenceManager().reference_source == "managed"
    assert ReferenceManager(reference_dir=tmp_path).reference_source == "explicit"


def test_plot_uses_installed_managed_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCUSBLEND_REFERENCE_DIR", raising=False)
    api_helpers.patch_reference_layer(monkeypatch, tmp_path, patch_manager=False)
    patch_managed_dir(monkeypatch, tmp_path)

    result = locusblend.plot(
        api_helpers.make_dataset("one"),
        api_helpers.make_dataset("two"),
        ancestry=EUR,
        mode="standard",
        chrom=CHROM,
        center_bp=api_helpers.CENTER_BP,
    )

    assert result.metadata["reference_source"] == "managed"
    assert result.metadata["reference_dir"] == str(tmp_path)
    assert result.metadata["reference_validation"]["ok"] is True


def test_plot_without_usable_reference_is_actionable_and_never_downloads(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCUSBLEND_REFERENCE_DIR", raising=False)
    patch_managed_dir(monkeypatch, tmp_path / "missing_managed")
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
    assert "install_reference" in message
    assert calls == []


def test_plot_never_downloads_when_managed_data_is_installed(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCUSBLEND_REFERENCE_DIR", raising=False)
    api_helpers.patch_reference_layer(monkeypatch, tmp_path, patch_manager=False)
    patch_managed_dir(monkeypatch, tmp_path)
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
# 6-9. manifest parsing / validation
# ----------------------------------------------------------------------
def test_install_reference_rejects_invalid_ancestry(tmp_path):
    _, manifest_path, root, _ = published_bundle(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        install_reference(ancestry="ABC", reference_dir=root, manifest=manifest_path)
    assert "Unsupported ancestry 'ABC'" in str(excinfo.value)


def test_install_reference_accepts_mixed_case_ancestry(tmp_path):
    _, manifest_path, root, _ = published_bundle(tmp_path)
    result = install_reference(ancestry="eUr", reference_dir=root, manifest=manifest_path)
    assert result.ancestry == "EUR"


def test_install_reference_without_manifest_is_clear(tmp_path):
    with pytest.raises(ManifestError) as excinfo:
        install_reference(ancestry=EUR, reference_dir=tmp_path)
    message = str(excinfo.value)
    assert "No public reference manifest is configured yet" in message
    assert "manifest=" in message


def test_load_manifest_from_mapping_path_and_file_url(tmp_path):
    bundle, manifest_path, _, _ = published_bundle(tmp_path)

    from_mapping = load_manifest(bundle)
    from_path = load_manifest(manifest_path)
    from_url = load_manifest(manifest_path.as_uri())

    assert from_mapping.bundle_version == from_path.bundle_version == from_url.bundle_version
    assert len(from_mapping.files) == len(from_path.files) == len(from_url.files)
    assert from_mapping.genome_build == "GRCh38"
    assert from_path.ancestries() == ("EUR",)
    assert from_path.resources() == ("1000g", "gencode", "recombination")
    assert "reference manifest:" in from_path.describe()
    assert from_path.to_dict()["files"][0]["resource"] == "1000g"


def test_manifest_requires_an_existing_local_file(tmp_path):
    with pytest.raises(ManifestError) as excinfo:
        load_manifest(tmp_path / "nope.json")
    assert "manifest file not found" in str(excinfo.value)


def _minimal_file(**overrides):
    entry = {
        "resource": "1000g",
        "path": "1000g/EUR/x_ch14.bed",
        "url": "file:///tmp/x.bed",
        "sha256": "a" * 64,
        "ancestry": "EUR",
        "chrom": "14",
    }
    entry.update(overrides)
    return entry


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda m: m.update(schema_version="1"), "schema_version"),
        (lambda m: m.update(schema_version=2), "unsupported manifest schema_version"),
        (lambda m: m.pop("schema_version"), "schema_version"),
        (lambda m: m.pop("bundle_version"), "bundle_version"),
        (lambda m: m.update(bundle_version="  "), "bundle_version"),
        (lambda m: m.update(files=[]), "non-empty list"),
        (lambda m: m.update(files="nope"), "non-empty list"),
        (lambda m: m.update(genome_build="GRCh37"), "Unsupported genome_build"),
        (lambda m: m.update(files=[_minimal_file(resource="unknown")]), "unsupported resource"),
        (lambda m: m.update(files=[_minimal_file(sha256="abc")]), "invalid sha256"),
        (lambda m: m.update(files=[_minimal_file(sha256=None)]), "sha256"),
        (lambda m: m.update(files=[_minimal_file(size=-5)]), "invalid 'size'"),
        (lambda m: m.update(files=[_minimal_file(optional="yes")]), "'optional' flag"),
        (lambda m: m.update(files=[_minimal_file(ancestry="ABC")]), "Unsupported ancestry"),
        (lambda m: m.update(files=[_minimal_file(ancestry=None)]), "must declare an 'ancestry'"),
        (lambda m: m.update(files=[_minimal_file(chrom=None)]), "must declare a 'chrom'"),
        (lambda m: m.update(files=[_minimal_file(chrom="Y")]), "unsupported chromosome"),
        (lambda m: m.update(files=[_minimal_file(url="")]), "url"),
        (lambda m: m.update(files=[_minimal_file(path="")]), "empty 'path'"),
        (
            lambda m: m.update(
                files=[
                    _minimal_file(resource="gencode", path="gencode/a.gtf.gz", ancestry="EUR", chrom=None),
                ]
            ),
            "must not declare an 'ancestry'",
        ),
        (
            lambda m: m.update(files=[_minimal_file(), _minimal_file()]),
            "more than once",
        ),
    ],
)
def test_malformed_manifests_fail_clearly(mutate, expected):
    document = {
        "schema_version": 1,
        "bundle_version": "bundle-x",
        "genome_build": "GRCh38",
        "files": [_minimal_file()],
    }
    mutate(document)
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(document)
    assert expected in str(excinfo.value)


def test_manifest_rejects_non_json_text():
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest("{not json")
    assert "not valid JSON" in str(excinfo.value)


def test_manifest_rejects_plain_http_urls(tmp_path):
    bundle, manifest_path, _, _ = published_bundle(tmp_path)
    bundle["files"][0]["url"] = "http://example.invalid/x.bed"
    with pytest.raises(ManifestError) as excinfo:
        load_manifest(bundle["files"][0]["url"])
    assert "https" in str(excinfo.value)


def test_genome_build_and_checksum_normalization():
    assert normalize_genome_build("hg38") == "GRCh38"
    assert normalize_genome_build(None) == "GRCh38"
    assert normalize_sha256("SHA256:" + "A" * 64) == "a" * 64
    with pytest.raises(ManifestError):
        normalize_sha256("zz")


# ----------------------------------------------------------------------
# 10-11. path safety
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "path",
    [
        "../evil.bed",
        "1000g/../../etc/passwd",
        "..\\evil.bed",
        "a/../b",
        "~/evil.bed",
        "",
        "   ",
        "a/b\x00c",
    ],
)
def test_path_traversal_and_junk_rejected(path):
    with pytest.raises(ManifestError):
        validate_relative_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "C:\\evil.bed",
        "C:/evil.bed",
        "\\\\server\\share\\x.bed",
        "//server/share/x.bed",
        "D:evil.bed",
    ],
)
def test_absolute_manifest_paths_rejected(path):
    with pytest.raises(ManifestError):
        validate_relative_path(path)


def test_valid_relative_paths_are_normalized():
    assert validate_relative_path("1000g\\EUR\\x.bed") == "1000g/EUR/x.bed"
    assert validate_relative_path("./gencode//a.gtf.gz") == "gencode/a.gtf.gz"


def test_destination_within_stays_inside(tmp_path):
    inside = destination_within(tmp_path, "1000g/EUR/x.bed")
    assert inside == (tmp_path.resolve() / "1000g/EUR/x.bed")
    with pytest.raises(ManifestError):
        destination_within(tmp_path, "../outside.bed")


def test_destination_within_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "reference" / "link"
    link.parent.mkdir(parents=True)
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available in this environment")

    with pytest.raises(ManifestError):
        destination_within(tmp_path / "reference", "link/escaped.bed")


def test_install_rejects_unsafe_manifest_paths(tmp_path):
    bundle, manifest_path, root, published = published_bundle(tmp_path)
    bundle["files"][0]["path"] = "../../escaped.bed"
    manifest_path.write_text(json.dumps(bundle), encoding="utf-8")

    with pytest.raises(ManifestError):
        install_reference(ancestry=EUR, reference_dir=root, manifest=manifest_path)
    assert not (tmp_path.parent / "escaped.bed").exists()


# ----------------------------------------------------------------------
# 12. checksum helpers
# ----------------------------------------------------------------------
def test_sha256_helpers(tmp_path):
    data = b"locusblend-reference-test\n"
    path = write_bytes(tmp_path / "tiny.bin", data)
    digest = sha256_of(data)

    assert fetch.sha256_file(path) == digest
    assert fetch.file_matches_checksum(path, digest) is True
    assert fetch.file_matches_checksum(path, "b" * 64) is False
    assert fetch.file_matches_checksum(tmp_path / "missing.bin", digest) is False
    assert fetch.normalize_checksum(digest.upper()) == digest
    assert fetch.normalize_checksum("sha256:" + digest) == digest
    with pytest.raises(fetch.ChecksumError):
        fetch.normalize_checksum("nope")
    with pytest.raises(fetch.ChecksumError):
        fetch.normalize_checksum(None)


# ----------------------------------------------------------------------
# 13-15. download engine
# ----------------------------------------------------------------------
def test_download_file_success_is_atomic(tmp_path):
    data = b"reference-data"
    source = write_bytes(tmp_path / "source.bin", data)
    destination = tmp_path / "out" / "nested" / "target.bin"

    result = fetch.download_file(source.as_uri(), destination, sha256=sha256_of(data))

    assert result.status == "downloaded"
    assert result.bytes_written == len(data)
    assert destination.read_bytes() == data
    assert list(destination.parent.glob("*.part")) == []


def test_download_file_checksum_mismatch_discards_data(tmp_path):
    data = b"reference-data"
    source = write_bytes(tmp_path / "source.bin", data)
    destination = tmp_path / "out" / "target.bin"

    with pytest.raises(fetch.ChecksumError) as excinfo:
        fetch.download_file(source.as_uri(), destination, sha256="0" * 64)

    assert "Checksum mismatch" in str(excinfo.value)
    assert not destination.exists()
    assert list(destination.parent.glob("*.part")) == []


def test_download_file_size_mismatch_discards_data(tmp_path):
    data = b"reference-data"
    source = write_bytes(tmp_path / "source.bin", data)
    destination = tmp_path / "target.bin"

    with pytest.raises(fetch.ChecksumError):
        fetch.download_file(source.as_uri(), destination, sha256=sha256_of(data), size=len(data) + 5)
    assert not destination.exists()

    with pytest.raises(fetch.ChecksumError):
        fetch.download_file(source.as_uri(), destination, sha256=sha256_of(data), size=len(data) - 5)
    assert not destination.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_download_file_missing_source_is_clear(tmp_path):
    with pytest.raises(fetch.DownloadError) as excinfo:
        fetch.download_file(
            (tmp_path / "missing.bin").as_uri(), tmp_path / "out.bin", sha256="a" * 64
        )
    assert "Could not fetch" in str(excinfo.value)
    assert not (tmp_path / "out.bin").exists()


def test_download_file_rejects_non_https_schemes(tmp_path):
    for url in ("http://example.invalid/x.bin", "ftp://example.invalid/x.bin"):
        with pytest.raises(fetch.DownloadError) as excinfo:
            fetch.download_file(url, tmp_path / "out.bin", sha256="a" * 64)
        assert "Unsupported URL scheme" in str(excinfo.value)
    assert fetch.SUPPORTED_URL_SCHEMES == ("https", "file")


def test_download_file_skips_valid_existing_file(tmp_path):
    data = b"already-there"
    source = write_bytes(tmp_path / "source.bin", data)
    destination = write_bytes(tmp_path / "target.bin", data)
    before = destination.stat().st_mtime_ns

    result = fetch.download_file(source.as_uri(), destination, sha256=sha256_of(data))

    assert result.status == "skipped"
    assert result.bytes_written == 0
    assert destination.read_bytes() == data
    assert destination.stat().st_mtime_ns == before


def test_download_file_replaces_invalid_existing_file(tmp_path):
    source = write_bytes(tmp_path / "source.bin", b"correct-data")
    destination = write_bytes(tmp_path / "target.bin", b"corrupt")

    result = fetch.download_file(
        source.as_uri(), destination, sha256=sha256_of(b"correct-data")
    )

    assert result.status == "replaced"
    assert destination.read_bytes() == b"correct-data"


def test_download_file_force_redownloads_valid_file(tmp_path):
    source = write_bytes(tmp_path / "source.bin", b"fresh-data")
    destination = write_bytes(tmp_path / "target.bin", b"fresh-data")

    result = fetch.download_file(
        source.as_uri(), destination, sha256=sha256_of(b"fresh-data"), force=True
    )

    assert result.status == "replaced"
    assert destination.read_bytes() == b"fresh-data"
    assert list(tmp_path.glob("*.part")) == []


# ----------------------------------------------------------------------
# 16-18. install_reference selection semantics
# ----------------------------------------------------------------------
def test_install_installs_only_requested_ancestry(tmp_path):
    bundle, manifest_path, root, _ = published_bundle(tmp_path, ancestries=("AFR", "EUR"))
    result = install_reference(ancestry=EUR, reference_dir=root, manifest=manifest_path)

    assert result.ancestry == EUR
    assert result.ok is True
    assert all(f.status == "downloaded" for f in result.files)

    manager_eur = ReferenceManager(reference_dir=root, ancestry=EUR)
    assert manager_eur.get_bfile_prefix(CHROM)
    assert not (root / "1000g" / "AFR").exists()
    assert (root / "gencode").is_dir()

    paths_installed = {f.path for f in result.files}
    assert all("1000g/EUR/" in p or p.startswith(("gencode/", "recombination/")) for p in paths_installed)


def test_install_reuses_shared_resources(tmp_path):
    bundle, manifest_path, root, _ = published_bundle(tmp_path, ancestries=("EUR", "SAS"))

    first = install_reference(ancestry=EUR, reference_dir=root, manifest=manifest_path)
    second = install_reference(ancestry="SAS", reference_dir=root, manifest=manifest_path)

    shared_first = {f.path for f in first.files if not f.path.startswith("1000g/")}
    shared_second = {f.path: f.status for f in second.files if not f.path.startswith("1000g/")}
    assert shared_first
    assert all(shared_second[path] == "skipped" for path in shared_first)

    # shared resources exist exactly once, outside the ancestry directories
    assert (root / "gencode").is_dir()
    assert not list((root / "1000g").glob("*/gencode*"))


def test_install_optional_recombination_failure_is_nonfatal(tmp_path):
    bundle, manifest_path, root, _ = published_bundle(
        tmp_path, shared=("gencode", "recombination"), break_recombination_url=True
    )
    result = install_reference(ancestry=EUR, reference_dir=root, manifest=manifest_path)

    assert result.ok is True
    assert any("optional file not installed" in warning for warning in result.warnings)
    assert any(
        f.path.startswith("recombination/") and f.status == "failed" for f in result.files
    )
    # required resources are still installed
    assert ReferenceManager(reference_dir=root, ancestry=EUR).get_bfile_prefix(CHROM)
    assert list((root / "gencode").glob("*.gtf.gz"))


def test_install_required_failure_raises_and_leaves_no_files(tmp_path):
    bundle, manifest_path, root, _ = published_bundle(tmp_path, corrupt_sha=True)

    with pytest.raises(ReferenceInstallError) as excinfo:
        install_reference(ancestry=EUR, reference_dir=root, manifest=manifest_path)

    message = str(excinfo.value)
    assert "Checksum mismatch" in message
    assert not (root / "1000g" / "EUR").exists() or not list(
        (root / "1000g" / "EUR").glob("*.bed")
    )
    assert list(root.rglob("*.part")) == []


def test_install_rejects_manifest_without_requested_ancestry(tmp_path):
    bundle, manifest_path, root, _ = published_bundle(tmp_path, ancestries=("AFR",))
    with pytest.raises(ManifestError) as excinfo:
        install_reference(ancestry="SAS", reference_dir=root, manifest=manifest_path)
    assert "no files for ancestry SAS" in str(excinfo.value)
    assert "AFR" in str(excinfo.value)


def test_install_supports_chromosome_subset(tmp_path):
    bundle, manifest_path, root, _ = published_bundle(tmp_path, chroms=("14", "15"))

    result = install_reference(
        ancestry=EUR, reference_dir=root, manifest=manifest_path, chroms=["14"]
    )

    assert {f.path for f in result.files if "1000g/" in f.path} == {
        f"1000g/EUR/{ReferenceManager(reference_dir=root, ancestry=EUR).bfile_prefix('14').name}{suffix}"
        for suffix in (".bed", ".bim", ".fam")
    }
    manager = ReferenceManager(reference_dir=root, ancestry=EUR)
    assert manager.get_bfile_prefix("14")
    with pytest.raises(FileNotFoundError):
        manager.get_bfile_prefix("15")


# ----------------------------------------------------------------------
# 19-20. inspection + marker
# ----------------------------------------------------------------------
def test_install_writes_marker_safely_and_updates_it(tmp_path):
    bundle, manifest_path, root, _ = published_bundle(tmp_path, ancestries=("EUR", "SAS"))

    install_reference(ancestry=EUR, reference_dir=root, manifest=manifest_path)
    marker_path = root / REFERENCE_MARKER_FILENAME
    assert marker_path.is_file()
    marker = json.loads(marker_path.read_text(encoding="utf-8"))

    assert marker["schema_version"] == 1
    assert marker["bundle_version"] == "test-bundle-1"
    assert marker["genome_build"] == "GRCh38"
    assert marker["ancestries"] == {"EUR": [CHROM]}
    assert marker["resources"]["gencode"] == [f"gencode/gencode.v49.annotation.chr{CHROM}.gtf.gz"]
    # no machine-specific absolute paths are recorded
    assert str(tmp_path) not in marker_path.read_text(encoding="utf-8")
    assert not list(root.glob("*.tmp"))

    install_reference(ancestry="SAS", reference_dir=root, manifest=manifest_path)
    updated = json.loads(marker_path.read_text(encoding="utf-8"))
    assert updated["ancestries"] == {"EUR": [CHROM], "SAS": [CHROM]}
    assert updated["updated_at"] >= marker["updated_at"]


def test_reference_status_reports_structure(tmp_path):
    bundle, manifest_path, root, _ = published_bundle(tmp_path, shared=("gencode",))
    install_reference(ancestry=EUR, reference_dir=root, manifest=manifest_path)

    status = reference_status(root)

    assert status.reference_dir == str(root)
    assert status.source == "explicit"
    assert status.exists is True
    assert status.installed_ancestries == ("EUR",)
    assert status.ancestry_chroms[EUR] == (CHROM,)
    assert status.is_chrom_available(EUR, "chr14") is True
    assert status.is_chrom_available(EUR, "7") is False
    assert status.is_chrom_available("ABC", CHROM) is False
    assert status.gencode_available is True
    assert status.recombination_available is False
    assert status.marker["bundle_version"] == "test-bundle-1"
    assert "reference status:" in status.describe()
    assert json.loads(json.dumps(status.to_dict()))["installed_ancestries"] == [EUR]


def test_reference_status_validates_requested_chrom(tmp_path):
    bundle, manifest_path, root, _ = published_bundle(tmp_path)
    install_reference(ancestry=EUR, reference_dir=root, manifest=manifest_path)

    ok_status = reference_status(root, ancestry=EUR, chrom=CHROM)
    assert ok_status.validation is not None and ok_status.validation.ok is True

    missing_status = reference_status(root, ancestry=EUR, chrom="7")
    assert missing_status.validation is not None
    assert missing_status.validation.ok is False
    assert any(".bed" in error for error in missing_status.validation.file_errors)
    assert any("not complete for EUR" in note for note in missing_status.notes)


def test_reference_status_unconfigured(monkeypatch, tmp_path):
    """Defensive branch: nothing resolvable at all (no explicit/env/managed)."""
    monkeypatch.delenv("LOCUSBLEND_REFERENCE_DIR", raising=False)
    monkeypatch.setattr(
        install, "resolve_reference_dir", lambda *a, **k: paths.ReferenceLocation()
    )

    status = reference_status()

    assert status.reference_dir is None
    assert status.source is None
    assert status.exists is False
    assert any("install_reference" in note for note in status.notes)


def test_reference_status_reports_managed_path_when_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCUSBLEND_REFERENCE_DIR", raising=False)
    managed = tmp_path / "managed"
    patch_managed_dir(monkeypatch, managed)

    status = reference_status()

    assert status.reference_dir == str(managed)
    assert status.source == "managed"
    assert status.exists is False
    assert any("does not exist yet" in note for note in status.notes)


def test_reference_status_partial_chromosomes_are_reported(tmp_path):
    bundle, manifest_path, root, _ = published_bundle(tmp_path)
    install_reference(ancestry=EUR, reference_dir=root, manifest=manifest_path)
    # remove one component of the installed chromosome
    manager = ReferenceManager(reference_dir=root, ancestry=EUR)
    Path(str(manager.bfile_prefix(CHROM)) + ".fam").unlink()

    status = reference_status(root)

    assert status.ancestry_chroms == {}
    assert status.partial_chroms[EUR] == (CHROM,)
    assert any("no complete 1000G ancestry panel" in note for note in status.notes)


# ----------------------------------------------------------------------
# 21-25. package surface, hygiene, no-network guarantee
# ----------------------------------------------------------------------
def test_package_root_exposes_distribution_api():
    for name in (
        "install_reference",
        "reference_status",
        "get_default_reference_dir",
        "ReferenceManifest",
        "ManifestError",
        "DownloadError",
        "ChecksumError",
        "ReferenceInstallError",
    ):
        assert hasattr(locusblend, name), name
    assert callable(locusblend.install_reference)
    assert callable(locusblend.reference_status)
    assert "streamlit" not in sys.modules
    assert issubclass(locusblend.ManifestError, ValueError)
    assert issubclass(locusblend.ChecksumError, locusblend.DownloadError)


def test_no_reference_data_or_plink_bundled_in_package():
    package_dir = Path(locusblend.__file__).resolve().parent
    bundled = [
        path.name
        for path in package_dir.iterdir()
        if path.name.endswith((".bed", ".bim", ".fam", ".gtf.gz", ".bw"))
    ]
    assert bundled == []
    assert fetch.SUPPORTED_URL_SCHEMES == ("https", "file")
    assert manifest_module.MANIFEST_SCHEMA_VERSION == 1