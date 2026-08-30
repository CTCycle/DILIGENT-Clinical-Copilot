from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import PurePosixPath
from typing import Any

FORBIDDEN_SUFFIXES = {
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".lib",
    ".map",
    ".pxd",
    ".pyi",
    ".pyx",
}
FORBIDDEN_PARTS = {
    ".angular",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "cache",
    "caches",
    "stubs",
    "test",
    "tests",
    "testing",
}
REQUIRED_FILES = {
    "app/client/dist/browser/index.html",
    "app/resources/catalogs/llm_model_capabilities.json",
    "backend/DILIGENTBackend.exe",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise RuntimeError(f"unsafe archive member: {name}")
    return path


def _load_manifest(archive: zipfile.ZipFile, version: str) -> dict[str, Any]:
    try:
        manifest = json.loads(archive.read("runtime-manifest.json"))
    except (KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError("runtime-manifest.json is missing or invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("release_version") != version:
        raise RuntimeError("runtime manifest release version mismatch")
    if not isinstance(manifest.get("files"), list):
        raise RuntimeError("runtime manifest files must be an array")
    return manifest


def validate_archive(archive_path: str, version: str) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise RuntimeError("runtime archive contains duplicate members")
        for info in infos:
            path = _safe_path(info.filename)
            if info.is_dir():
                raise RuntimeError(
                    f"runtime archive contains a directory member: {info.filename}"
                )
            if path.suffix.casefold() in FORBIDDEN_SUFFIXES or any(
                part.casefold() in FORBIDDEN_PARTS for part in path.parts
            ):
                raise RuntimeError(f"forbidden runtime archive member: {info.filename}")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise RuntimeError(
                    f"runtime archive contains a symlink: {info.filename}"
                )

        manifest = _load_manifest(archive, version)
        manifest_files = manifest["files"]
        manifest_names: set[str] = set()
        total_size = 0
        for entry in manifest_files:
            if not isinstance(entry, dict):
                raise RuntimeError("runtime manifest contains an invalid file entry")
            name = str(entry.get("path", ""))
            _safe_path(name)
            if name in manifest_names:
                raise RuntimeError(f"runtime manifest contains duplicate path: {name}")
            manifest_names.add(name)
            payload = archive.read(name)
            expected_size = int(entry.get("size", -1))
            expected_hash = str(entry.get("sha256", ""))
            if len(payload) != expected_size or _sha256(payload) != expected_hash:
                raise RuntimeError(f"runtime manifest content mismatch: {name}")
            total_size += len(payload)

        archive_names = set(names)
        expected_names = manifest_names | {"runtime-manifest.json"}
        unexpected = sorted(archive_names - expected_names)
        missing = sorted(expected_names - archive_names)
        if unexpected or missing:
            raise RuntimeError(
                f"runtime archive membership mismatch; unexpected={unexpected}, missing={missing}"
            )
        missing_required = sorted(REQUIRED_FILES - manifest_names)
        if missing_required:
            raise RuntimeError(
                f"required runtime files are missing: {missing_required}"
            )
        if manifest.get("architecture") != "windows-x64":
            raise RuntimeError("runtime archive architecture is not windows-x64")
        return {
            "archive": archive_path,
            "files": len(manifest_files),
            "total_size": total_size,
            "required_files": sorted(REQUIRED_FILES),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--version", required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            validate_archive(arguments.archive, arguments.version), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
