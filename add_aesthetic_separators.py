#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import difflib
import io
import re
import sys
import tokenize
from pathlib import Path


TOP_LEVEL_SEPARATOR = "###############################################################################"
METHOD_SEPARATOR = "# -------------------------------------------------------------------------"

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".tox",
    ".nox",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
}

# Matches separator-looking comment lines, for example:
# ###############################################################################
# # -------------------------------------------------------------------------
# # =============================================================================
SEPARATOR_RE = re.compile(r"^\s*#\s*([#=\-_*])(?:\s*\1){9,}\s*$")


def is_blank(line: str) -> bool:
    return line.strip() == ""


def is_existing_separator(line: str) -> bool:
    return bool(SEPARATOR_RE.match(line.rstrip("\r\n")))


def leading_whitespace(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def detect_encoding(raw: bytes) -> str:
    readline = io.BytesIO(raw).readline
    encoding, _ = tokenize.detect_encoding(readline)
    return encoding


def detect_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def first_target_line(node: ast.AST) -> int:
    """
    Return the line where the aesthetic separator should attach.

    If real Python decorators are present, the separator goes above the
    first decorator, not between the decorator and its class/function.
    """
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        return min(decorator.lineno for decorator in decorators)
    return node.lineno


def collect_targets(tree: ast.Module) -> list[tuple[int, str]]:
    """
    Return tuples of:
      zero_based_line_index, separator_text_without_indent

    Classes and module-level functions get TOP_LEVEL_SEPARATOR.
    Methods directly inside a class get METHOD_SEPARATOR.
    Nested functions are intentionally ignored.
    """
    parents: dict[ast.AST, ast.AST] = {}

    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    targets: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            targets.append((first_target_line(node) - 1, TOP_LEVEL_SEPARATOR))
            continue

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parent = parents.get(node)

            if isinstance(parent, ast.Module):
                targets.append((first_target_line(node) - 1, TOP_LEVEL_SEPARATOR))
            elif isinstance(parent, ast.ClassDef):
                targets.append((first_target_line(node) - 1, METHOD_SEPARATOR))

    return sorted(set(targets), reverse=True)


def apply_separators(text: str, filename: str) -> str:
    try:
        tree = ast.parse(text, filename=filename, type_comments=True)
    except SyntaxError:
        raise

    targets = collect_targets(tree)
    if not targets:
        return text

    newline = detect_newline(text)
    lines = text.splitlines(keepends=True)

    for target_idx, separator in targets:
        if target_idx < 0 or target_idx >= len(lines):
            continue

        indent = leading_whitespace(lines[target_idx])

        # Remove existing aesthetic separator blocks immediately above the target.
        # This scans through blank lines and separator-looking comment lines only.
        scan_idx = target_idx - 1
        while scan_idx >= 0 and (
            is_blank(lines[scan_idx]) or is_existing_separator(lines[scan_idx])
        ):
            scan_idx -= 1

        block_start = scan_idx + 1
        block = lines[block_start:target_idx]

        if any(is_existing_separator(line) for line in block):
            del lines[block_start:target_idx]
            target_idx = block_start

        insert_lines: list[str] = []

        # Require one empty row before the aesthetic separator unless the target
        # is at the very beginning of the file.
        if target_idx > 0 and not is_blank(lines[target_idx - 1]):
            insert_lines.append(newline)

        # Separator must be directly above the target, including real decorators.
        insert_lines.append(f"{indent}{separator}{newline}")

        lines[target_idx:target_idx] = insert_lines

    new_text = "".join(lines)

    # Safety check: comments and blank lines should not break syntax, but verify.
    ast.parse(new_text, filename=filename, type_comments=True)

    return new_text


def iter_python_files(path: Path) -> list[Path]:
    """
    If path is a file, process only that file.

    If path is a directory, process .py files only inside child folders,
    not .py files directly inside path itself. This lets the script live
    at the repo root without modifying itself.
    """
    if path.is_file():
        return [path] if path.suffix == ".py" else []

    files: list[Path] = []

    for child in sorted(path.iterdir()):
        if not child.is_dir():
            continue

        if child.name in SKIP_DIRS:
            continue

        for candidate in child.rglob("*.py"):
            if any(part in SKIP_DIRS for part in candidate.parts):
                continue
            files.append(candidate)

    return sorted(files)


def read_source(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    encoding = detect_encoding(raw)
    return raw.decode(encoding), encoding


def process_file(path: Path, write: bool, show_diff: bool) -> bool:
    original, encoding = read_source(path)

    try:
        updated = apply_separators(original, str(path))
    except SyntaxError as exc:
        print(f"SKIP syntax error: {path}: {exc}", file=sys.stderr)
        return False

    if updated == original:
        return False

    if show_diff:
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
        )
        sys.stdout.writelines(diff)

    if write:
        path.write_bytes(updated.encode(encoding))

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Add deterministic aesthetic separators above classes, "
            "module-level functions, and class methods."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("."),
        help=(
            "Python file or directory to process recursively. "
            "Defaults to current directory. If a directory is used, only child "
            "folders are scanned, not root-level .py files."
        ),
    )
    parser.add_argument("--write", action="store_true", help="Modify files in place.")
    parser.add_argument("--diff", action="store_true", help="Print unified diffs.")
    args = parser.parse_args()

    files = iter_python_files(args.path)

    changed = 0
    for file_path in files:
        if process_file(file_path, write=args.write, show_diff=args.diff):
            changed += 1

    mode = "modified" if args.write else "would modify"
    print(f"{mode}: {changed} file(s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
