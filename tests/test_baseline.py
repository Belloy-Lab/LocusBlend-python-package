"""``app.2.8.12.py`` is the immutable behavioral baseline.

The baseline must never be modified, cleaned up, reformatted or patched, so
this test pins its exact content. If this test fails, the change is a bug in a
refactoring pass, not a test that needs updating.
"""

import ast
import hashlib
from pathlib import Path

BASELINE_PATH = Path(__file__).resolve().parent.parent / "app.2.8.12.py"
EXPECTED_SHA256 = "4df15c3a6e5b357d2b9998dc8e8e156ca557e5369a9e2091686f960655333851"
EXPECTED_SIZE_BYTES = 262968


def test_baseline_file_is_unchanged():
    assert BASELINE_PATH.is_file()
    data = BASELINE_PATH.read_bytes()
    assert len(data) == EXPECTED_SIZE_BYTES
    assert hashlib.sha256(data).hexdigest() == EXPECTED_SHA256


def test_extracted_modules_do_not_import_the_baseline():
    """The package must never import the Streamlit baseline module."""
    import locusblend

    package_dir = Path(locusblend.__file__).resolve().parent
    for path in sorted(package_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] != "app", f"{path.name}: import {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                # relative imports (level > 0) are inside the package
                if node.level == 0 and node.module:
                    assert node.module.split(".")[0] != "app", f"{path.name}: from {node.module}"
