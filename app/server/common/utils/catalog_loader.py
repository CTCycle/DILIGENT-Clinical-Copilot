from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


###############################################################################
class CatalogLoader:
    """Load static JSON catalogs from resources/catalogs at runtime."""

    # -------------------------------------------------------------------------
    @staticmethod
    def _catalogs_dir() -> Path:
        return Path(__file__).resolve().parents[3] / "resources" / "catalogs"

    # -------------------------------------------------------------------------
    @classmethod
    @lru_cache(maxsize=32)
    def load_catalog(cls, filename: str) -> dict[str, Any]:
        path = cls._catalogs_dir() / filename
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Catalog '{filename}' must contain a JSON object at root."
            )
        return payload

    # -------------------------------------------------------------------------
    @classmethod
    def get_string_list(cls, filename: str, key: str) -> list[str]:
        payload = cls.load_catalog(filename)
        values = payload.get(key)
        if not isinstance(values, list):
            return []
        return [str(item).strip() for item in values if str(item).strip()]

    # -------------------------------------------------------------------------
    @classmethod
    def get_string_set(cls, filename: str, key: str) -> set[str]:
        return set(cls.get_string_list(filename, key))

    # -------------------------------------------------------------------------
    @classmethod
    def get_catalog_records(
        cls,
        filename: str,
        key: str,
        fields: tuple[str, ...],
    ) -> tuple[tuple[str, ...], ...]:
        payload = cls.load_catalog(filename)
        rows = payload.get(key)
        if not isinstance(rows, list):
            return ()

        records: list[tuple[str, ...]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            values: list[str] = []
            missing = False
            for field in fields:
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    missing = True
                    break
                values.append(value.strip())
            if not missing:
                records.append(tuple(values))
        return tuple(records)


__all__ = ["CatalogLoader"]
