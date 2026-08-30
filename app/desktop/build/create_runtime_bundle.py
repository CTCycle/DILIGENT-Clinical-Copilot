from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

ARCHIVE_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _safe_relative(path: str) -> PurePosixPath:
    candidate = PurePosixPath(path.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise RuntimeError(f"Unsafe runtime path: {path}")
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        raise RuntimeError("Runtime payload must contain an entries array")
    return payload


def _collect_files(staging: Path, payload: dict[str, Any]) -> dict[str, Path]:
    collected: dict[str, Path] = {}
    forbidden_directories = set(payload.get("forbidden_directories", []))
    forbidden_extensions = set(payload.get("forbidden_extensions", []))
    for entry in payload["entries"]:
        source = staging / _safe_relative(str(entry["source"]))
        destination = _safe_relative(str(entry["destination"]))
        if not source.exists():
            if entry.get("required", True):
                raise FileNotFoundError(
                    f"Required runtime payload is missing: {source}"
                )
            continue
        paths = [source] if source.is_file() else sorted(source.rglob("*"))
        for item in paths:
            if item.is_symlink():
                raise RuntimeError(
                    f"Symlinks are not allowed in runtime payload: {item}"
                )
            if item.is_dir():
                continue
            relative = item.relative_to(source) if source.is_dir() else Path()
            archive_path = _safe_relative(str(destination / relative))
            if any(part in forbidden_directories for part in archive_path.parts):
                raise RuntimeError(f"Forbidden runtime path: {archive_path}")
            if archive_path.suffix in forbidden_extensions:
                raise RuntimeError(f"Forbidden runtime file: {archive_path}")
            if archive_path.as_posix().endswith("/.env") or archive_path.name == ".env":
                raise RuntimeError(
                    f"User environment file cannot be packaged: {archive_path}"
                )
            if archive_path.as_posix() in collected:
                raise RuntimeError(f"Duplicate runtime path: {archive_path}")
            collected[archive_path.as_posix()] = item
    if not collected:
        raise RuntimeError("Runtime payload is empty")
    return collected


def create_bundle(args: argparse.Namespace) -> None:
    staging = Path(args.staging).resolve()
    collected = _collect_files(staging, _load_payload(Path(args.payload)))
    largest = sorted(
        collected.items(), key=lambda item: item[1].stat().st_size, reverse=True
    )
    files = [
        {"path": name, "size": source.stat().st_size, "sha256": _sha256(source)}
        for name, source in sorted(collected.items())
    ]
    manifest: dict[str, Any] = {
        "release_version": args.version,
        "architecture": args.architecture,
        "python_version": args.python_version,
        "pyinstaller_version": args.pyinstaller_version,
        "tauri_version": args.tauri_version,
        "frontend_package_version": args.frontend_package_version,
        "source_commit_sha": args.source_commit_sha,
        "dirty_tree": args.dirty_tree.lower() == "true",
        "files": files,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name, source in sorted(collected.items()):
                info = zipfile.ZipInfo(name, ARCHIVE_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, source.read_bytes())
            info = zipfile.ZipInfo("runtime-manifest.json", ARCHIVE_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, manifest_bytes)
        temporary.replace(output)
        Path(args.manifest).write_bytes(manifest_bytes)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {"archive": str(output), "sha256": _sha256(output), "files": len(files)}
        )
    )
    print("20 largest runtime files:")
    for name, source in largest[:20]:
        print(f"{source.stat().st_size:>12} {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", required=True)
    parser.add_argument(
        "--payload", default=str(Path(__file__).with_name("runtime_payload.json"))
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--architecture", default="windows-x64")
    parser.add_argument("--python-version", default="unknown")
    parser.add_argument("--pyinstaller-version", default="6.21.0")
    parser.add_argument("--tauri-version", default="2.11.5")
    parser.add_argument("--frontend-package-version", default="unknown")
    parser.add_argument("--source-commit-sha", default="unknown")
    parser.add_argument("--dirty-tree", default="false")
    return parser


if __name__ == "__main__":
    create_bundle(build_parser().parse_args())
