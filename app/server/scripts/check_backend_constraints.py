from __future__ import annotations

import ast
from pathlib import Path

EXCLUDED_DIRS = {".venv", "__pycache__", ".pytest_cache"}
MAX_PHYSICAL_LINES = 1000
LEGACY_OVERSIZED_FILES = {
    Path("app/server/repositories/serialization/data.py"),
    Path("app/server/services/clinical/hepatox_core.py"),
    Path("app/server/services/clinical/matches_core.py"),
    Path("app/server/services/clinical/parser.py"),
    Path("app/server/services/inspection/service.py"),
    Path("app/server/services/llm/ollama_client.py"),
    Path("app/server/services/session/session_service.py"),
    Path("app/server/services/updater/livertox_core.py"),
}


class ConstraintVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.violations: list[str] = []
        self.scope_stack: list[ast.AST] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Import(self, node: ast.Import) -> None:
        self._check_module_import(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._check_module_import(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self.scope_stack:
            self.violations.append(
                f"{self.path}:{node.lineno}: nested function '{node.name}'"
            )
        self.scope_stack.append(node)
        self.generic_visit(node)
        self.scope_stack.pop()

    def _check_module_import(self, node: ast.Import | ast.ImportFrom) -> None:
        if self.scope_stack:
            self.violations.append(
                f"{self.path}:{node.lineno}: import is not at module scope"
            )


def iter_python_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        paths.append(path)
    return paths


def check_file(path: Path, repo_root: Path) -> list[str]:
    violations: list[str] = []
    text = path.read_text(encoding="utf-8")
    line_count = text.count("\n") + (0 if text.endswith("\n") else 1)
    relative_path = path.relative_to(repo_root)
    if line_count > MAX_PHYSICAL_LINES and relative_path not in LEGACY_OVERSIZED_FILES:
        violations.append(
            f"{path}:1: file has {line_count} physical lines; max is {MAX_PHYSICAL_LINES}"
        )
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        violations.append(f"{path}:{exc.lineno or 1}: syntax error: {exc.msg}")
        return violations
    visitor = ConstraintVisitor(path)
    visitor.visit(tree)
    violations.extend(visitor.violations)
    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    roots = (repo_root / "app" / "server",)
    violations: list[str] = []
    for root in roots:
        for path in iter_python_files(root):
            violations.extend(check_file(path, repo_root))
    if violations:
        for violation in violations:
            print(violation)
        return 1
    print("Backend constraints passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
