from __future__ import annotations

import ast
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
PROMPT_ROOT = APP_ROOT / "server" / "common" / "prompts"
SERVICE_ROOT = APP_ROOT / "server" / "services"


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def test_prompt_package_contains_python_modules_only() -> None:
    unexpected = [
        path
        for path in PROMPT_ROOT.iterdir()
        if path.is_file() and path.suffix != ".py"
    ]
    assert unexpected == []


def test_prompt_definitions_do_not_call_strip() -> None:
    violations: list[str] = []
    for path in _python_files(PROMPT_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            value = None
            if isinstance(node, ast.Assign):
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                value = node.value
            if not isinstance(value, ast.Call):
                continue
            if isinstance(value.func, ast.Attribute) and value.func.attr == "strip":
                violations.append(f"{path.relative_to(APP_ROOT)}:{node.lineno}")
    assert violations == []


def test_services_do_not_define_prompt_constants() -> None:
    violations: list[str] = []
    for path in _python_files(SERVICE_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                name = target.id
                if (
                    name.isupper()
                    and "PROMPT" in name
                    and not name.endswith("PROMPT_VERSION")
                ):
                    violations.append(
                        f"{path.relative_to(APP_ROOT)}:{node.lineno}:{name}"
                    )
    assert violations == []


def test_services_do_not_embed_substantial_llm_call_prompts() -> None:
    violations: list[str] = []
    for path in _python_files(SERVICE_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg not in {"system_prompt", "user_prompt"}:
                    continue
                if (
                    isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                    and len(keyword.value.value) >= 80
                ):
                    violations.append(
                        f"{path.relative_to(APP_ROOT)}:{node.lineno}:{keyword.arg}"
                    )
    assert violations == []


def test_legacy_extraction_prompt_module_is_removed() -> None:
    assert not (PROMPT_ROOT / "extraction.py").exists()
    for path in _python_files(SERVICE_ROOT):
        assert "common.prompts.extraction" not in path.read_text(encoding="utf-8")
