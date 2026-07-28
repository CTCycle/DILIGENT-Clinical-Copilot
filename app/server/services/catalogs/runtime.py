from __future__ import annotations

from functools import lru_cache
from types import MappingProxyType

from common.catalogs.provider import get_catalog_provider
from domain.catalogs import CatalogEntry, ReferenceCatalogSnapshot
from repositories.database.session import get_default_repository
from repositories.serialization.catalogs import ReferenceCatalogSerializer
from services.catalogs.seeder import ReferenceCatalogSeeder

###############################################################################
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

###############################################################################
def _build_reference_catalog_snapshot() -> ReferenceCatalogSnapshot:
    repository = get_default_repository()
    serializer = ReferenceCatalogSerializer(session_factory=repository.session_factory)
    ReferenceCatalogSeeder(serializer).seed_missing_or_changed_manifests()
    return _build_snapshot(serializer.list_active_entries())

###############################################################################
@lru_cache(maxsize=1)
def _cached_reference_catalog_snapshot() -> ReferenceCatalogSnapshot:
    return _build_reference_catalog_snapshot()

###############################################################################
def get_reference_catalog_snapshot(
    repository=None,
) -> ReferenceCatalogSnapshot:
    if repository is None:
        return _cached_reference_catalog_snapshot()
    serializer = ReferenceCatalogSerializer(session_factory=repository.session_factory)
    ReferenceCatalogSeeder(serializer).seed_missing_or_changed_manifests()
    return _build_snapshot(serializer.list_active_entries())

###############################################################################
def reload_reference_catalog_snapshot(repository=None) -> ReferenceCatalogSnapshot:
    if repository is None:
        _cached_reference_catalog_snapshot.cache_clear()
        return _cached_reference_catalog_snapshot()
    _cached_reference_catalog_snapshot.cache_clear()
    serializer = ReferenceCatalogSerializer(session_factory=repository.session_factory)
    ReferenceCatalogSeeder(serializer).seed_missing_or_changed_manifests()
    return _build_snapshot(serializer.list_active_entries())

###############################################################################
def reset_reference_catalog_snapshot_for_tests() -> None:
    _cached_reference_catalog_snapshot.cache_clear()


###############################################################################
def initialize_reference_catalog_provider() -> None:
    get_catalog_provider().register(
        get_reference_catalog_snapshot,
        invalidate=_cached_reference_catalog_snapshot.cache_clear,
    )
