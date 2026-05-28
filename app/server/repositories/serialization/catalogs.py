from __future__ import annotations

import json
from types import MappingProxyType

from sqlalchemy import delete, select
from sqlalchemy.orm import sessionmaker

from domain.catalogs import (
    CatalogEntry,
    CatalogManifest,
    normalize_catalog_value,
)
from repositories.schemas.models import ReferenceCatalogEntry, ReferenceCatalogSeedRun


class ReferenceCatalogSerializer:
    def __init__(self, session_factory: sessionmaker) -> None:
        self.session_factory = session_factory

    def list_active_entries(self) -> list[CatalogEntry]:
        session = self.session_factory()
        try:
            rows = (
                session.execute(
                    select(ReferenceCatalogEntry).where(
                        ReferenceCatalogEntry.active.is_(True)
                    )
                )
                .scalars()
                .all()
            )
            return [self._to_domain_entry(row) for row in rows]
        finally:
            session.close()

    def replace_manifest_entries(
        self,
        manifest: CatalogManifest,
        manifest_hash: str,
        source_path: str,
    ) -> int:
        session = self.session_factory()
        try:
            session.execute(
                delete(ReferenceCatalogEntry).where(
                    ReferenceCatalogEntry.manifest == manifest.manifest
                )
            )
            written = 0
            for entry in manifest.entries:
                session.add(
                    ReferenceCatalogEntry(
                        manifest=manifest.manifest,
                        manifest_version=manifest.version,
                        domain=entry.domain,
                        category=entry.category,
                        key=entry.key,
                        locale=entry.locale,
                        value=entry.value,
                        normalized_value=entry.normalized_value
                        or normalize_catalog_value(entry.value),
                        priority=entry.priority,
                        match_mode=entry.match_mode,
                        case_sensitive=entry.case_sensitive,
                        metadata_json=json.dumps(
                            dict(entry.metadata), ensure_ascii=False
                        ),
                        active=entry.active,
                    )
                )
                written += 1
            session.commit()
            return written
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def has_successful_seed(self, manifest: str, manifest_hash: str) -> bool:
        session = self.session_factory()
        try:
            row = session.execute(
                select(ReferenceCatalogSeedRun.id).where(
                    ReferenceCatalogSeedRun.manifest == manifest,
                    ReferenceCatalogSeedRun.manifest_hash == manifest_hash,
                    ReferenceCatalogSeedRun.status == "success",
                )
            ).first()
            return row is not None
        finally:
            session.close()

    def record_seed_success(
        self,
        manifest: str,
        version: int,
        hash: str,
        source_path: str,
        entry_count: int,
    ) -> None:
        self._record_seed_run(
            manifest=manifest,
            version=version,
            hash=hash,
            source_path=source_path,
            status="success",
            entry_count=entry_count,
            error_message=None,
        )

    def record_seed_failure(
        self,
        manifest: str,
        version: int,
        hash: str,
        source_path: str,
        error: str,
    ) -> None:
        self._record_seed_run(
            manifest=manifest,
            version=version,
            hash=hash,
            source_path=source_path,
            status="failure",
            entry_count=0,
            error_message=error,
        )

    def clear_all_catalog_entries(self) -> None:
        session = self.session_factory()
        try:
            session.execute(delete(ReferenceCatalogEntry))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def upsert_runtime_observation(
        self,
        *,
        term: str,
        category: str,
        replacement: str | None = None,
        source: str = "session",
        is_active: bool = True,
        db_session=None,
    ) -> int | None:
        normalized_value = normalize_catalog_value(term)
        if not normalized_value:
            return None
        owns_session = db_session is None
        session = db_session or self.session_factory()
        try:
            row = (
                session.execute(
                    select(ReferenceCatalogEntry).where(
                        ReferenceCatalogEntry.manifest == "runtime_observations",
                        ReferenceCatalogEntry.domain == "text_normalization",
                        ReferenceCatalogEntry.category == category,
                        ReferenceCatalogEntry.key == "default",
                        ReferenceCatalogEntry.locale == "und",
                        ReferenceCatalogEntry.normalized_value == normalized_value,
                    )
                )
                .scalars()
                .first()
            )
            metadata = {"source": source, "replacement": replacement}
            if row is None:
                row = ReferenceCatalogEntry(
                    manifest="runtime_observations",
                    manifest_version=1,
                    domain="text_normalization",
                    category=category,
                    key="default",
                    locale="und",
                    value=term,
                    normalized_value=normalized_value,
                    priority=100,
                    match_mode="token",
                    case_sensitive=False,
                    metadata_json=json.dumps(metadata, ensure_ascii=False),
                    active=bool(is_active),
                )
                session.add(row)
            else:
                row.value = term
                row.active = bool(is_active)
                row.metadata_json = json.dumps(metadata, ensure_ascii=False)
            if owns_session:
                session.commit()
            else:
                session.flush()
            return int(row.id) if row.id is not None else None
        except Exception:
            if owns_session:
                session.rollback()
            raise
        finally:
            if owns_session:
                session.close()

    def list_runtime_observations(
        self,
        *,
        category: str | None = None,
    ) -> list[ReferenceCatalogEntry]:
        session = self.session_factory()
        try:
            query = select(ReferenceCatalogEntry).where(
                ReferenceCatalogEntry.manifest == "runtime_observations",
                ReferenceCatalogEntry.domain == "text_normalization",
            )
            if category:
                query = query.where(ReferenceCatalogEntry.category == category)
            return (
                session.execute(query.order_by(ReferenceCatalogEntry.id.asc()))
                .scalars()
                .all()
            )
        finally:
            session.close()

    def deactivate_runtime_observation(self, *, category: str, term: str) -> bool:
        normalized_value = normalize_catalog_value(term)
        if not normalized_value:
            return False
        session = self.session_factory()
        try:
            row = (
                session.execute(
                    select(ReferenceCatalogEntry).where(
                        ReferenceCatalogEntry.manifest == "runtime_observations",
                        ReferenceCatalogEntry.domain == "text_normalization",
                        ReferenceCatalogEntry.category == category,
                        ReferenceCatalogEntry.key == "default",
                        ReferenceCatalogEntry.locale == "und",
                        ReferenceCatalogEntry.normalized_value == normalized_value,
                    )
                )
                .scalars()
                .first()
            )
            if row is None:
                return False
            row.active = False
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _record_seed_run(
        self,
        *,
        manifest: str,
        version: int,
        hash: str,
        source_path: str,
        status: str,
        entry_count: int,
        error_message: str | None,
    ) -> None:
        session = self.session_factory()
        try:
            existing = (
                session.execute(
                    select(ReferenceCatalogSeedRun).where(
                        ReferenceCatalogSeedRun.manifest == manifest,
                        ReferenceCatalogSeedRun.manifest_hash == hash,
                        ReferenceCatalogSeedRun.status == status,
                    )
                )
                .scalars()
                .first()
            )
            if existing is None:
                session.add(
                    ReferenceCatalogSeedRun(
                        manifest=manifest,
                        manifest_version=version,
                        manifest_hash=hash,
                        source_path=source_path,
                        status=status,
                        entry_count=entry_count,
                        error_message=error_message,
                    )
                )
            else:
                existing.manifest_version = version
                existing.source_path = source_path
                existing.entry_count = entry_count
                existing.error_message = error_message
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _to_domain_entry(row: ReferenceCatalogEntry) -> CatalogEntry:
        metadata = {}
        if row.metadata_json:
            try:
                parsed = json.loads(row.metadata_json)
                if isinstance(parsed, dict):
                    metadata = parsed
            except json.JSONDecodeError:
                metadata = {}
        return CatalogEntry(
            manifest=row.manifest,
            manifest_version=int(row.manifest_version),
            domain=row.domain,
            category=row.category,
            key=row.key,
            locale=row.locale,
            value=row.value,
            normalized_value=row.normalized_value,
            priority=int(row.priority),
            match_mode=row.match_mode,
            case_sensitive=bool(row.case_sensitive),
            metadata=MappingProxyType(metadata),
            active=bool(row.active),
        )
