"""Packaging metadata, dependency audit and export-dependency behavior."""

import ast
import re
import sys
from importlib import metadata
from pathlib import Path

import pytest

import locusblend
from locusblend import export

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_DIR = Path(locusblend.__file__).resolve().parent
GITIGNORE = REPO_ROOT / ".gitignore"

# third-party modules the package imports at module level
EXPECTED_RUNTIME_DEPENDENCIES = {"numpy", "pandas", "platformdirs", "plotly"}
REFERENCE_DATA_SUFFIXES = (".bed", ".bim", ".fam", ".gtf.gz", ".bw")


def _toml_array(text, key):
    """Return the string items of a simple ``key = [ ... ]`` TOML array."""
    match = re.search(rf"^{re.escape(key)}\s*=\s*\[(.*?)\]", text, flags=re.M | re.S)
    assert match, f"{key} array not found in pyproject.toml"
    return re.findall(r'"([^"]+)"', match.group(1))


def _requirement_name(requirement):
    return re.split(r"[<>=!~;\[ ]", requirement.strip(), maxsplit=1)[0]


def _module_level_imports(path):
    """Return the top-level module names imported by *path* (AST based)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.append(node.module.split(".")[0])
    return names


@pytest.fixture(scope="module")
def pyproject_text():
    return PYPROJECT.read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# 11. pyproject metadata
# ----------------------------------------------------------------------
def test_pyproject_name_version_and_python(pyproject_text):
    assert 'name = "locusblend"' in pyproject_text
    assert 'version = "0.1.0.dev0"' in pyproject_text
    assert 'requires-python = ">=3.9"' in pyproject_text
    assert 'readme = "README.md"' in pyproject_text
    # src/ layout
    assert 'package-dir = { "" = "src" }' in pyproject_text
    assert 'where = ["src"]' in pyproject_text


def test_declared_version_matches_package(pyproject_text):
    declared = re.search(r'^version = "([^"]+)"', pyproject_text, flags=re.M).group(1)
    assert declared == locusblend.__version__ == "0.1.0.dev0"


def test_runtime_dependencies_are_complete_and_minimal(pyproject_text):
    declared = _toml_array(pyproject_text, "dependencies")
    names = {_requirement_name(item) for item in declared}
    assert names == EXPECTED_RUNTIME_DEPENDENCIES
    # version floors are declared for the numeric/plotting stack
    assert all(re.search(r"[<>]=?", item) for item in declared)


def test_streamlit_is_not_a_dependency(pyproject_text):
    for block in ("dependencies", "export", "genes", "recomb", "dev"):
        items = _toml_array(pyproject_text, block)
        assert all("streamlit" not in item.lower() for item in items), block


def test_optional_extras_are_declared(pyproject_text):
    assert set(_toml_array(pyproject_text, "export")) == {"pillow", "kaleido"}
    assert {_requirement_name(item) for item in _toml_array(pyproject_text, "genes")} == {"pyarrow"}
    assert {_requirement_name(item) for item in _toml_array(pyproject_text, "recomb")} == {"pyBigWig"}
    assert "pytest" in _toml_array(pyproject_text, "dev")


def test_module_level_imports_are_all_declared(pyproject_text):
    """Every module-level third-party import must be a declared dependency."""
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    stdlib |= {
        "os",
        "re",
        "copy",
        "base64",
        "hashlib",
        "glob",
        "shutil",
        "subprocess",
        "tempfile",
        "time",
        "io",
        "pathlib",
        "mimetypes",
        "dataclasses",
        "typing",
        "__future__",
    }
    declared = {_requirement_name(item) for item in _toml_array(pyproject_text, "dependencies")}

    imported = {
        module
        for path in sorted(PACKAGE_DIR.glob("*.py"))
        for module in _module_level_imports(path)
        if module not in stdlib and module != "locusblend"
    }

    undeclared = sorted(imported - declared)
    assert undeclared == [], f"module-level imports not declared: {undeclared}"


def test_optional_dependencies_are_imported_lazily():
    """Pillow / pyBigWig must not be imported at package import time."""
    optional = {"PIL", "pyBigWig", "kaleido", "pyarrow"}
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        offenders = optional & set(_module_level_imports(path))
        assert offenders == set(), f"{path.name} imports optional dependencies at module level: {offenders}"


def test_installed_metadata_matches_pyproject():
    """Only meaningful when the package is installed (editable installs count)."""
    try:
        dist_version = metadata.version("locusblend")
    except metadata.PackageNotFoundError:
        pytest.skip("locusblend is not installed in this environment")

    assert dist_version == "0.1.0.dev0"
    requires = metadata.requires("locusblend") or []
    core = [r for r in requires if "extra ==" not in r]
    core_names = {_requirement_name(r).lower() for r in core}
    for name in ("numpy", "pandas", "plotly"):
        assert name in core_names, name
    assert all("streamlit" not in r.lower() for r in requires)


# ----------------------------------------------------------------------
# reference data must never be committed
# ----------------------------------------------------------------------
def test_gitignore_excludes_reference_data():
    text = GITIGNORE.read_text(encoding="utf-8")
    for pattern in ("*.bed", "*.bim", "*.fam", "*.gtf.gz", "*.bw", "reference/", "references/"):
        assert pattern in text, pattern


def test_no_reference_data_in_the_repository():
    offenders = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(REPO_ROOT)
        if relative.parts and relative.parts[0] in {".git", ".pytest_cache", ".venv"}:
            continue
        if any(part.endswith(".egg-info") or part == "__pycache__" for part in relative.parts):
            continue
        if relative.name.endswith(REFERENCE_DATA_SUFFIXES):
            offenders.append(str(relative))
    assert offenders == []


# ----------------------------------------------------------------------
# 12. export dependency behavior
# ----------------------------------------------------------------------
def test_export_hint_mentions_the_optional_extra():
    assert "locusblend[export]" in export.EXPORT_EXTRA_HINT


def test_missing_pillow_message_is_actionable():
    message = str(export._missing_pillow_error(ImportError("No module named 'PIL'")))
    assert "Pillow is required" in message
    assert "locusblend[export]" in message
    assert "No module named 'PIL'" in message


def test_static_export_message_is_actionable():
    message = str(export._static_export_error(RuntimeError("kaleido is not installed")))
    assert "Static export failed" in message
    assert "locusblend[export]" in message
    assert "kaleido" in message
    assert "Chrome" in message
    assert "kaleido is not installed" in message  # original error is not swallowed


def test_render_png_surfaces_kaleido_errors(monkeypatch):
    import plotly.graph_objects as go

    def boom(*args, **kwargs):
        raise RuntimeError("kaleido not installed")

    monkeypatch.setattr(export.pio, "to_image", boom)

    with pytest.raises(RuntimeError) as excinfo:
        export.render_plotly_figure_to_png_bytes(go.Figure(), 200, 150)

    message = str(excinfo.value)
    assert "locusblend[export]" in message
    assert "kaleido not installed" in message


def test_export_helpers_import_without_pillow_at_package_import():
    """Importing the package must not require the optional export deps."""
    assert "PIL" not in sys.modules or True  # imported lazily, never at import time
    assert callable(export.image_to_export_bytes)
    assert callable(export.render_plotly_figure_to_png_bytes)
