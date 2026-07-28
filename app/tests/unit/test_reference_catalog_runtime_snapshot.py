from __future__ import annotations

from types import MappingProxyType

import pytest

from common.catalogs.provider import get_catalog_provider
from domain.catalogs import CatalogEntry, ReferenceCatalogSnapshot
from services.catalogs.runtime import (
    _build_snapshot,
    initialize_reference_catalog_provider,
)

###############################################################################
def test_runtime_snapshot_values_and_metadata() -> None:
    snapshot = _build_snapshot(
        [
            CatalogEntry(
                manifest="m",
                manifest_version=1,
                domain="text_normalization",
                category="matching_stopwords",
                key="default",
                locale="und",
                value="mg",
                normalized_value="mg",
                priority=100,
                match_mode="token",
                case_sensitive=False,
                metadata=MappingProxyType({"match_mode": "token"}),
            )
        ]
    )
    assert snapshot.values(
        "text_normalization", "matching_stopwords", key="default"
    ) == ("mg",)
    assert (
        snapshot.metadata("text_normalization", "matching_stopwords", "default")[
            "match_mode"
        ]
        == "token"
    )


###############################################################################
def test_catalog_provider_registration_is_explicit_and_isolated() -> None:
    get_catalog_provider.cache_clear()
    provider = get_catalog_provider()
    snapshot = _build_snapshot([])
    invalidated: list[bool] = []

    with pytest.raises(RuntimeError, match="not registered"):
        provider.get_snapshot()

    def load_snapshot() -> ReferenceCatalogSnapshot:
        return snapshot

    def invalidate() -> None:
        invalidated.append(True)

    provider.register(load_snapshot, invalidate=invalidate)
    assert provider.get_snapshot() is snapshot
    provider.invalidate()
    assert invalidated == [True]
    provider.register(load_snapshot, invalidate=invalidate)

    get_catalog_provider.cache_clear()
    assert get_catalog_provider() is not provider
    initialize_reference_catalog_provider()
