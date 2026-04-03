from __future__ import annotations

import ast
from pathlib import Path


SERVER_ROOT = Path("DILIGENT/server")
DOMAIN_ROOT = SERVER_ROOT / "domain"


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _is_dataclass_decorator(decorator: ast.expr) -> bool:
    if isinstance(decorator, ast.Name):
        return decorator.id == "dataclass"
    if isinstance(decorator, ast.Attribute):
        return decorator.attr == "dataclass"
    if isinstance(decorator, ast.Call):
        return _is_dataclass_decorator(decorator.func)
    return False


def _is_pydantic_model(class_node: ast.ClassDef) -> bool:
    for base in class_node.bases:
        if isinstance(base, ast.Name) and base.id in {"BaseModel", "BaseSettings"}:
            return True
        if isinstance(base, ast.Attribute) and base.attr in {"BaseModel", "BaseSettings"}:
            return True
    return False


def _format_violation(path: Path, node: ast.AST, label: str) -> str:
    line = getattr(node, "lineno", 1)
    return f"{path.as_posix()}:{line} ({label})"


def test_no_nested_local_python_functions() -> None:
    violations: list[str] = []
    for path in _iter_python_files(SERVER_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.function_depth = 0

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                if self.function_depth > 0:
                    violations.append(_format_violation(path, node, "nested function"))
                self.function_depth += 1
                self.generic_visit(node)
                self.function_depth -= 1

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                if self.function_depth > 0:
                    violations.append(_format_violation(path, node, "nested function"))
                self.function_depth += 1
                self.generic_visit(node)
                self.function_depth -= 1

        Visitor().visit(tree)
    assert not violations, "Nested local functions are forbidden:\n" + "\n".join(violations)


def test_no_conditional_python_imports() -> None:
    violations: list[str] = []
    for path in _iter_python_files(SERVER_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.if_depth = 0

            def visit_If(self, node: ast.If) -> None:
                self.if_depth += 1
                self.generic_visit(node)
                self.if_depth -= 1

            def visit_Import(self, node: ast.Import) -> None:
                if self.if_depth > 0:
                    violations.append(_format_violation(path, node, "conditional import"))
                self.generic_visit(node)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if self.if_depth > 0:
                    violations.append(_format_violation(path, node, "conditional import"))
                self.generic_visit(node)

        Visitor().visit(tree)
    assert not violations, "Conditional imports are forbidden:\n" + "\n".join(violations)


def test_models_live_under_domain() -> None:
    violations: list[str] = []
    for path in _iter_python_files(SERVER_ROOT):
        if DOMAIN_ROOT in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if any(_is_dataclass_decorator(decorator) for decorator in node.decorator_list):
                violations.append(_format_violation(path, node, "dataclass outside domain"))
                continue
            if _is_pydantic_model(node):
                violations.append(_format_violation(path, node, "pydantic model outside domain"))
    assert not violations, (
        "Dataclasses and Pydantic models must be defined under DILIGENT/server/domain:\n"
        + "\n".join(violations)
    )
