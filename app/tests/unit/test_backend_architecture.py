from __future__ import annotations

import ast
from pathlib import Path


SERVER_ROOT = Path(__file__).resolve().parents[2] / "server"

###############################################################################
def python_files(root: Path) -> list[Path]:
    ignored = {"__pycache__", ".venv", "pytest-cache-files"}
    return [
        path
        for path in root.rglob("*.py")
        if not any(
            part in ignored or part.startswith("pytest-cache-files")
            for part in path.parts
        )
    ]

###############################################################################
def test_schema_ownership_and_removed_tables_are_canonical() -> None:
    schema_root = SERVER_ROOT / "repositories" / "schemas"
    assert not (schema_root / "models.py").exists()
    source = "\n".join(path.read_text(encoding="utf-8") for path in python_files(SERVER_ROOT))
    assert "repositories.schemas.models" not in source
    assert "ClinicalSessionLab" not in source
    assert "ClinicalSessionDrug" not in source
    assert "clinical_session_labs" not in source
    assert "clinical_session_drugs" not in source

###############################################################################
MAX_BACKEND_FILE_LINES = 1200


def _module_name(node: ast.Import | ast.ImportFrom) -> str:
    if isinstance(node, ast.Import):
        return ", ".join(alias.name for alias in node.names)
    return node.module or "."


def _imports(path: Path) -> list[ast.Import | ast.ImportFrom]:
    return [
        node
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def test_backend_files_do_not_exceed_line_limit() -> None:
    for path in python_files(SERVER_ROOT):
        lines = len(path.read_text(encoding="utf-8").splitlines())
        assert lines <= MAX_BACKEND_FILE_LINES, f"{path.relative_to(SERVER_ROOT)}: {lines} lines"


def test_backend_imports_are_module_level() -> None:
    for path in python_files(SERVER_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_imports = {id(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            assert id(node) in module_imports, (
                f"{path.relative_to(SERVER_ROOT)}:{node.lineno}: {_module_name(node)}"
            )


def test_backend_does_not_define_nested_functions() -> None:
    for path in python_files(SERVER_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for parent in ast.walk(tree):
            if not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for child in ast.walk(parent):
                if child is not parent and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    assert False, f"{path.relative_to(SERVER_ROOT)}:{child.lineno}: nested function"


def test_backend_does_not_use_global_or_nonlocal_statements() -> None:
    for path in python_files(SERVER_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            assert not isinstance(node, (ast.Global, ast.Nonlocal)), (
                f"{path.relative_to(SERVER_ROOT)}:{node.lineno}: {type(node).__name__}"
            )


def test_backend_does_not_suppress_import_order() -> None:
    for path in python_files(SERVER_ROOT):
        text = path.read_text(encoding="utf-8")
        assert "noqa: E402" not in text, path.relative_to(SERVER_ROOT)


def test_backend_does_not_use_dynamic_application_imports() -> None:
    for path in python_files(SERVER_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "import_module", (
                    f"{path.relative_to(SERVER_ROOT)}:{node.lineno}: import_module"
                )


def test_backend_layer_dependencies() -> None:
    forbidden = {
        "api": ("repositories", "sqlalchemy"),
        "services": ("api",),
        "repositories": ("api", "services"),
        "domain": ("api", "services", "repositories"),
        "common": ("api", "services", "repositories"),
        "configurations": ("api",),
    }
    for path in python_files(SERVER_ROOT):
        relative = path.relative_to(SERVER_ROOT)
        layer = relative.parts[0]
        if layer not in forbidden:
            continue
        for node in _imports(path):
            imported = _module_name(node).lower()
            for blocked in forbidden[layer]:
                if blocked in imported and not (path.name == "app.py" and layer == "repositories"):
                    assert False, f"{relative}:{node.lineno}: {imported}"

###############################################################################
def test_domain_owns_extractor_and_pattern_contracts() -> None:
    contracts = (SERVER_ROOT / "domain" / "clinical" / "extractor_contracts.py").read_text(
        encoding="utf-8"
    )
    assert "class LocalDiseaseContextEntry" in contracts
    assert "class LocalLabExtractionPayload" in contracts
    assert "class HepaticPatternResolutionResult" in contracts
    assert "class LocalDiseaseContextEntry" not in (
        SERVER_ROOT / "services" / "clinical" / "disease.py"
    ).read_text(encoding="utf-8")
    assert "class LocalLabExtractionPayload" not in (
        SERVER_ROOT / "services" / "clinical" / "labs.py"
    ).read_text(encoding="utf-8")
