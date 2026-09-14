"""The package must import and work without Streamlit."""

import ast
import importlib
import sys
from pathlib import Path

import pytest

SUBMODULES = [
    "api",
    "colors",
    "compare",
    "config",
    "export",
    "genes",
    "io",
    "ld",
    "models",
    "plotting",
    "reference",
    "variants",
]


def test_import_locusblend_without_streamlit():
    import locusblend

    assert "streamlit" not in sys.modules
    assert locusblend.__version__ == "0.1.0.dev0"
    assert callable(locusblend.plot)

    for name in ("LocusBlendConfig", "LocusBlendResult", "IndexVariant", "ReferenceManager"):
        assert hasattr(locusblend, name), name


@pytest.mark.parametrize("submodule", SUBMODULES)
def test_submodule_import_without_streamlit(submodule):
    module = importlib.import_module(f"locusblend.{submodule}")
    assert module is not None
    assert "streamlit" not in sys.modules


def test_package_sources_do_not_use_streamlit():
    """Architectural guard: no streamlit imports and no ``st.`` usage in src/."""
    import locusblend

    package_dir = Path(locusblend.__file__).resolve().parent
    offenders = []

    for path in sorted(package_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "streamlit":
                        offenders.append(f"{path.name}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] == "streamlit":
                    offenders.append(f"{path.name}:{node.lineno}: from {node.module} import ...")
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == "st":
                    offenders.append(f"{path.name}:{node.lineno}: st.{node.attr}")
            elif isinstance(node, ast.Name) and node.id == "st":
                offenders.append(f"{path.name}:{node.lineno}: name 'st'")

    assert offenders == []


def test_reference_modules_avoid_hard_coded_server_paths():
    """Architectural guard: no WashU/server-specific absolute paths."""
    import locusblend

    package_dir = Path(locusblend.__file__).resolve().parent
    forbidden = ("/srv/shiny-server", "C:\\WashU", "/WashU/")

    for path in sorted(package_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"{path.name} contains {needle!r}"
