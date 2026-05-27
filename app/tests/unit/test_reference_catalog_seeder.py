from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

from domain.catalogs import CatalogEntry, CatalogManifest
from services.catalogs import seeder as seeder_module
from services.catalogs.seeder import ReferenceCatalogSeeder


class _FakeSerializer:
    def __init__(self) -> None:
        self.success: set[tuple[str, str]] = set()
        self.replaced: list[str] = []

    def has_successful_seed(self, manifest: str, manifest_hash: str) -> bool:
        return (manifest, manifest_hash) in self.success

    def replace_manifest_entries(self, manifest: CatalogManifest, manifest_hash: str, source_path: str) -> int:
        self.replaced.append(manifest.manifest)
        return len(manifest.entries)

    def record_seed_success(self, manifest: str, version: int, hash: str, source_path: str, entry_count: int) -> None:
        self.success.add((manifest, hash))

    def record_seed_failure(self, manifest: str, version: int, hash: str, source_path: str, error: str) -> None:
        raise AssertionError(error)


def _manifest(name: str) -> CatalogManifest:
    return CatalogManifest(
        manifest=name,
        version=1,
        description="x",
        entries=(
            CatalogEntry(
                manifest=name,
                manifest_version=1,
                domain="d",
                category="c",
                key="k",
                locale="und",
                value="v",
                normalized_value="v",
                priority=100,
                match_mode="token",
                case_sensitive=False,
                metadata=MappingProxyType({}),
            ),
        ),
    )


def test_seeder_skips_existing_hash(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    serializer = _FakeSerializer()
    serializer.success.add(("a", "h1"))
    monkeypatch.setattr(seeder_module, "iter_catalog_manifest_paths", lambda: [Path("a.json")])
    monkeypatch.setattr(seeder_module, "compute_manifest_hash", lambda _path: "h1")
    monkeypatch.setattr(seeder_module, "load_catalog_manifest", lambda _path: _manifest("a"))
    result = ReferenceCatalogSeeder(serializer).seed_missing_or_changed_manifests(force=False)
    assert result.manifests_seen == 1
    assert result.manifests_seeded == 0
    assert serializer.replaced == []

