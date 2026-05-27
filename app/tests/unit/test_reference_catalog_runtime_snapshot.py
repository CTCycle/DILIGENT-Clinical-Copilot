from __future__ import annotations

from types import MappingProxyType

from domain.catalogs import CatalogEntry
from services.catalogs.runtime import _build_snapshot


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
    assert snapshot.values("text_normalization", "matching_stopwords", key="default") == (
        "mg",
    )
    assert snapshot.metadata("text_normalization", "matching_stopwords", "default")[
        "match_mode"
    ] == "token"

