"""Documentation coverage checks for the package source."""

import ast
from pathlib import Path


def test_all_source_definitions_have_docstrings():
    """Require docstrings for every package class and function."""

    source = Path(__file__).parents[1] / "src" / "asterodetect"
    missing = []
    for path in sorted(source.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(
                node,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ) and ast.get_docstring(node) is None:
                missing.append(f"{path.name}:{node.lineno}:{node.name}")
    assert not missing, "missing docstrings: " + ", ".join(missing)
