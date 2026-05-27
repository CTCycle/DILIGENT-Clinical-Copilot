from __future__ import annotations

from functools import lru_cache
from types import MappingProxyType

from domain.catalogs import CatalogEntry, ReferenceCatalogSnapshot
from repositories.database.session import get_default_repository
from repositories.serialization.catalogs import ReferenceCatalogSerializer


def _build_snapshot(entries: list[CatalogEntry]) -> ReferenceCatalogSnapshot:
    grouped: dict[tuple[str, str, str, str], list[CatalogEntry]] = {}
    for entry in entries:
        scoped_key = (entry.domain, entry.category, entry.key, entry.locale)
        wildcard_key = (entry.domain, entry.category, "*", entry.locale)
        grouped.setdefault(scoped_key, []).append(entry)
        grouped.setdefault(wildcard_key, []).append(entry)
    packed = {
        key: tuple(sorted(values, key=lambda item: (-item.priority, item.value)))
        for key, values in grouped.items()
    }
    return ReferenceCatalogSnapshot(entries_by_scope=MappingProxyType(packed))


@lru_cache(maxsize=1)
def get_reference_catalog_snapshot() -> ReferenceCatalogSnapshot:
    repository = get_default_repository()
    serializer = ReferenceCatalogSerializer(session_factory=repository.session_factory)
    return _build_snapshot(serializer.list_active_entries())


def reload_reference_catalog_snapshot() -> ReferenceCatalogSnapshot:
    get_reference_catalog_snapshot.cache_clear()
    return get_reference_catalog_snapshot()


def reset_reference_catalog_snapshot_for_tests() -> None:
    get_reference_catalog_snapshot.cache_clear()

