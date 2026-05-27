from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from common.constants import RESOURCES_PATH
from domain.catalogs import CatalogEntry, CatalogManifest, normalize_catalog_value

CATALOG_MANIFEST_DIR = Path(RESOURCES_PATH) / "catalogs"


def iter_catalog_manifest_paths() -> list[Path]:
    if not CATALOG_MANIFEST_DIR.exists():
        return []
    return sorted(
        path
        for path in CATALOG_MANIFEST_DIR.glob("*.json")
        if path.name != "llm_models.json" and path.name != "local_models.json"
    )


def compute_manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_catalog_manifest(path: Path) -> CatalogManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid catalog manifest type for {path}")
    validate_manifest(payload)

    manifest_name = str(payload["manifest"])
    version = int(payload["version"])
    description = str(payload.get("description") or "")
    parsed_entries: list[CatalogEntry] = []
    for row in payload["entries"]:
        locale = str(row.get("locale") or "und")
        priority = int(row.get("priority") or 100)
        metadata = row.get("metadata")
        metadata_map = metadata if isinstance(metadata, dict) else {}
        case_sensitive = bool(metadata_map.get("case_sensitive", False))
        match_mode = str(metadata_map.get("match_mode") or "token")
        for value in row["values"]:
            parsed_entries.append(
                CatalogEntry(
                    manifest=manifest_name,
                    manifest_version=version,
                    domain=str(row["domain"]),
                    category=str(row["category"]),
                    key=str(row["key"]),
                    locale=locale,
                    value=str(value),
                    normalized_value=normalize_catalog_value(str(value)),
                    priority=priority,
                    match_mode=match_mode,
                    case_sensitive=case_sensitive,
                    metadata=MappingProxyType(metadata_map.copy()),
                )
            )
    return CatalogManifest(
        manifest=manifest_name,
        version=version,
        description=description,
        entries=tuple(parsed_entries),
    )


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    manifest_name = str(manifest.get("manifest") or "").strip()
    if not manifest_name:
        raise ValueError("manifest must be non-empty")
    version = manifest.get("version")
    if not isinstance(version, int) or version <= 0:
        raise ValueError("version must be a positive integer")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("entries must be a non-empty list")

    seen: set[tuple[str, str, str, str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("entry must be an object")
        for field in ("domain", "category", "key", "values"):
            if field not in entry:
                raise ValueError(f"entry missing required field: {field}")
        values = entry.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError("entry values must be a non-empty list")
        locale = str(entry.get("locale") or "und")
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("entry values must contain non-empty strings")
            dedupe_key = (
                manifest_name,
                str(entry["domain"]),
                str(entry["category"]),
                str(entry["key"]),
                locale,
                normalize_catalog_value(value),
            )
            if dedupe_key in seen:
                raise ValueError(
                    "duplicate normalized value for manifest/domain/category/key/locale"
                )
            seen.add(dedupe_key)

