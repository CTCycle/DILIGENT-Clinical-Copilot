from __future__ import annotations

import ast
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = APP_ROOT / "server"
DOMAIN_ROOT = SERVER_ROOT / "domain"
ALLOWED_DATACLASS_OUTSIDE_DOMAIN = {
    (SERVER_ROOT / "services" / "clinical" / "drug_blocks.py").resolve(),
    (SERVER_ROOT / "services" / "clinical" / "match_resolution.py").resolve(),
    (SERVER_ROOT / "services" / "session" / "clinical_section_parsers.py").resolve(),
    (SERVER_ROOT / "services" / "session" / "preflight.py").resolve(),
    (SERVER_ROOT / "services" / "runtime" / "state.py").resolve(),
}
ALLOWED_FASTAPI_IMPORTS_IN_SERVICES = {
    (SERVER_ROOT / "services" / "session" / "request_validation.py").resolve(),
}
EXCLUDED_DIRS = {
    "__pycache__",
    ".venv",
    ".uv-cache",
    ".pytest_cache",
    "node_modules",
    "dist",
}


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file() and EXCLUDED_DIRS.isdisjoint(path.parts)
    )


def test_backend_structure_scan_scope_is_not_empty() -> None:
    assert _iter_python_files(SERVER_ROOT), (
        f"Backend structure tests scanned no files under {SERVER_ROOT}"
    )


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
        if isinstance(base, ast.Name) and base.id in {
            "BaseModel",
            "BaseSettings",
            "RootModel",
        }:
            return True
        if isinstance(base, ast.Attribute) and base.attr in {
            "BaseModel",
            "BaseSettings",
            "RootModel",
        }:
            return True
    return False


def _format_violation(path: Path, node: ast.AST, label: str) -> str:
    line = getattr(node, "lineno", 1)
    return f"{path.as_posix()}:{line} ({label})"


def _is_top_level_import(
    tree: ast.Module, import_node: ast.Import | ast.ImportFrom
) -> bool:
    return import_node in tree.body


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
    assert not violations, "Nested local functions are forbidden:\n" + "\n".join(
        violations
    )


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
                    violations.append(
                        _format_violation(path, node, "conditional import")
                    )
                self.generic_visit(node)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if self.if_depth > 0:
                    violations.append(
                        _format_violation(path, node, "conditional import")
                    )
                self.generic_visit(node)

        Visitor().visit(tree)
    assert not violations, "Conditional imports are forbidden:\n" + "\n".join(
        violations
    )


def test_models_live_under_domain() -> None:
    violations: list[str] = []
    for path in _iter_python_files(SERVER_ROOT):
        if DOMAIN_ROOT in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if path.resolve() in ALLOWED_DATACLASS_OUTSIDE_DOMAIN:
                continue
            if any(
                _is_dataclass_decorator(decorator) for decorator in node.decorator_list
            ):
                violations.append(
                    _format_violation(path, node, "dataclass outside domain")
                )
                continue
            if _is_pydantic_model(node):
                violations.append(
                    _format_violation(path, node, "pydantic model outside domain")
                )
    assert not violations, (
        "Request/response models must be defined under app/server/domain; internal dataclasses outside domain require explicit allowlisting:\n"
        + "\n".join(violations)
    )


def test_livertox_updater_has_no_module_forwarding_wrappers() -> None:
    path = SERVER_ROOT / "services" / "updater" / "livertox_core.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "LiverToxUpdater":
            continue
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if len(item.body) != 1:
                continue
            statement = item.body[0]
            if not isinstance(statement, ast.Return):
                continue
            value = statement.value
            if isinstance(value, ast.Await):
                value = value.value
            if not isinstance(value, ast.Call):
                continue
            function = value.func
            if (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id
                in {
                    "livertox_download",
                    "livertox_index",
                    "livertox_parse",
                }
            ):
                violations.append(_format_violation(path, item, "facade wrapper"))
    assert not violations, (
        "LiverToxUpdater facade wrappers are forbidden:\n" + "\n".join(violations)
    )


def test_services_do_not_import_fastapi() -> None:
    services_root = SERVER_ROOT / "services"
    violations: list[str] = []
    for path in _iter_python_files(services_root):
        if path.resolve() in ALLOWED_FASTAPI_IMPORTS_IN_SERVICES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name == "fastapi" or name.startswith("fastapi."):
                        violations.append(
                            _format_violation(path, node, "fastapi import in services")
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "fastapi" or module.startswith("fastapi."):
                    violations.append(
                        _format_violation(path, node, "fastapi import in services")
                    )
    assert not violations, (
        "FastAPI imports are forbidden under app/server/services:\n"
        + "\n".join(violations)
    )


def test_backend_imports_are_top_level_only() -> None:
    root = SERVER_ROOT
    violations: list[str] = []

    for path in root.rglob("*.py"):
        if not EXCLUDED_DIRS.isdisjoint(path.parts):
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if not _is_top_level_import(tree, node):
                violations.append(f"{path.as_posix()}:{node.lineno}")

    assert not violations, (
        "Imports inside functions/classes are forbidden:\n" + "\n".join(violations)
    )


def test_services_do_not_import_sqlalchemy_orm_persistence() -> None:
    service_roots = [
        SERVER_ROOT / "services" / "inspection",
        SERVER_ROOT / "services" / "text",
    ]
    forbidden_modules = {
        "sqlalchemy",
        "repositories.schemas.models",
    }
    violations: list[str] = []

    for service_root in service_roots:
        for path in _iter_python_files(service_root):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in forbidden_modules or alias.name.startswith(
                            "sqlalchemy."
                        ):
                            violations.append(
                                _format_violation(
                                    path,
                                    node,
                                    "persistence import in services",
                                )
                            )
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module in forbidden_modules or module.startswith("sqlalchemy."):
                        violations.append(
                            _format_violation(
                                path,
                                node,
                                "persistence import in services",
                            )
                        )

    assert not violations, (
        "SQLAlchemy ORM persistence imports are forbidden under app/server/services; "
        "use repository-layer abstractions instead:\n" + "\n".join(violations)
    )
