from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


def normalize_catalog_value(value: str) -> str:
    normalized = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    )
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


@dataclass(frozen=True)
class CatalogEntry:
    manifest: str
    manifest_version: int
    domain: str
    category: str
    key: str
    locale: str
    value: str
    normalized_value: str
    priority: int
    match_mode: str
    case_sensitive: bool
    metadata: MappingProxyType[str, Any]
    active: bool = True


@dataclass(frozen=True)
class CatalogManifest:
    manifest: str
    version: int
    description: str
    entries: tuple[CatalogEntry, ...]


@dataclass(frozen=True)
class CatalogSeedResult:
    manifests_seen: int
    manifests_seeded: int
    entries_written: int


@dataclass(frozen=True)
class ReferenceCatalogSnapshot:
    entries_by_scope: MappingProxyType[
        tuple[str, str, str, str], tuple[CatalogEntry, ...]
    ]

    def entries(
        self,
        domain: str,
        category: str,
        key: str | None = None,
        locale: str = "und",
    ) -> tuple[CatalogEntry, ...]:
        scope_key = key or "*"
        return self.entries_by_scope.get((domain, category, scope_key, locale), ())

    def values(
        self,
        domain: str,
        category: str,
        key: str | None = None,
        locale: str = "und",
    ) -> tuple[str, ...]:
        return tuple(
            entry.value
            for entry in self.entries(domain, category, key=key, locale=locale)
        )

    def metadata(
        self,
        domain: str,
        category: str,
        key: str,
        locale: str = "und",
    ) -> MappingProxyType[str, Any]:
        entries = self.entries(domain, category, key=key, locale=locale)
        if not entries:
            return MappingProxyType({})
        return entries[0].metadata
