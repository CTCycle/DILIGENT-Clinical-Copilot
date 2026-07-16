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
def test_backend_layer_imports_and_file_size_guard() -> None:
    for path in python_files(SERVER_ROOT):
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 1100, path
    repositories = SERVER_ROOT / "repositories"
    for path in python_files(repositories):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert not any(isinstance(child, (ast.Import, ast.ImportFrom)) for child in ast.walk(node)), path
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                assert not any(name.startswith("services.") for name in names), path


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
