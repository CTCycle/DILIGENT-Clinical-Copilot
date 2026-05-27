from __future__ import annotations

from pathlib import Path

from domain.catalogs import CatalogSeedResult
from repositories.serialization.catalogs import ReferenceCatalogSerializer
from services.catalogs.manifest_loader import (
    compute_manifest_hash,
    iter_catalog_manifest_paths,
    load_catalog_manifest,
)


class ReferenceCatalogSeeder:
    def __init__(self, serializer: ReferenceCatalogSerializer) -> None:
        self.serializer = serializer

    def seed_missing_or_changed_manifests(
        self,
        force: bool = False,
    ) -> CatalogSeedResult:
        paths = iter_catalog_manifest_paths()
        manifests_seeded = 0
        entries_written = 0
        for path in paths:
            written = self.seed_manifest(path, force=force)
            if written > 0:
                manifests_seeded += 1
                entries_written += written
        return CatalogSeedResult(
            manifests_seen=len(paths),
            manifests_seeded=manifests_seeded,
            entries_written=entries_written,
        )

    def seed_manifest(self, path: Path, force: bool = False) -> int:
        manifest_hash = compute_manifest_hash(path)
        manifest = load_catalog_manifest(path)
        if (not force) and self.serializer.has_successful_seed(
            manifest.manifest, manifest_hash
        ):
            return 0
        try:
            written = self.serializer.replace_manifest_entries(
                manifest=manifest,
                manifest_hash=manifest_hash,
                source_path=str(path),
            )
            self.serializer.record_seed_success(
                manifest=manifest.manifest,
                version=manifest.version,
                hash=manifest_hash,
                source_path=str(path),
                entry_count=written,
            )
            return written
        except Exception as exc:
            self.serializer.record_seed_failure(
                manifest=manifest.manifest,
                version=manifest.version,
                hash=manifest_hash,
                source_path=str(path),
                error=str(exc),
            )
            raise
