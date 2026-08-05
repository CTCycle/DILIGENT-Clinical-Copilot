from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from repositories.database.session import resolve_engine, resolve_session_factory
from repositories.schemas.configuration import ProviderModelCatalogCache

###############################################################################
@dataclass(frozen=True)
class ProviderModelCatalogCacheRecord:
    provider_id: str
    configuration_fingerprint: str
    models: list[dict[str, Any]]
    last_success_at: datetime | None
    last_attempt_at: datetime
    last_attempt_status: str
    last_error: str | None

###############################################################################
class ProviderModelCatalogCacheSerializer:
    """Persist provider model catalogs independently from runtime settings."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        engine: Engine | None = None,
        session_factory: sessionmaker | None = None,
    ) -> None:
        self.engine = resolve_engine(engine)
        self.session_factory = resolve_session_factory(
            engine=self.engine,
            session_factory=session_factory,
            expire_on_commit=False,
        )

    # -------------------------------------------------------------------------
    def get(
        self, provider_id: str, configuration_fingerprint: str
    ) -> ProviderModelCatalogCacheRecord | None:
        with self.session_factory() as db_session:
            row = db_session.get(ProviderModelCatalogCache, provider_id)
            if row is None:
                return None
            if row.configuration_fingerprint != configuration_fingerprint:
                db_session.delete(row)
                db_session.commit()
                return None
            return self._to_record(row)

    # -------------------------------------------------------------------------
    def save_success(
        self,
        *,
        provider_id: str,
        configuration_fingerprint: str,
        models: list[dict[str, Any]],
    ) -> ProviderModelCatalogCacheRecord:
        now = datetime.now(UTC)
        with self.session_factory() as db_session:
            row = db_session.get(ProviderModelCatalogCache, provider_id)
            if row is None:
                row = ProviderModelCatalogCache(provider_id=provider_id)
                db_session.add(row)
            row.configuration_fingerprint = configuration_fingerprint
            row.models = list(models)
            row.last_success_at = now
            row.last_attempt_at = now
            row.last_attempt_status = "success"
            row.last_error = None
            row.updated_at = now
            db_session.commit()
            db_session.refresh(row)
            return self._to_record(row)

    # -------------------------------------------------------------------------
    def save_failure(
        self,
        *,
        provider_id: str,
        configuration_fingerprint: str,
        status: str,
        error: str,
    ) -> ProviderModelCatalogCacheRecord:
        now = datetime.now(UTC)
        with self.session_factory() as db_session:
            row = db_session.get(ProviderModelCatalogCache, provider_id)
            if row is None or row.configuration_fingerprint != configuration_fingerprint:
                if row is not None:
                    db_session.delete(row)
                row = ProviderModelCatalogCache(
                    provider_id=provider_id,
                    configuration_fingerprint=configuration_fingerprint,
                    models=[],
                    last_success_at=None,
                    last_attempt_at=now,
                    last_attempt_status=status,
                    last_error=error,
                    updated_at=now,
                )
                db_session.add(row)
            else:
                row.last_attempt_at = now
                row.last_attempt_status = status
                row.last_error = error
                row.updated_at = now
            db_session.commit()
            db_session.refresh(row)
            return self._to_record(row)

    # -------------------------------------------------------------------------
    def merge_model(
        self,
        *,
        provider_id: str,
        configuration_fingerprint: str,
        model: dict[str, Any],
    ) -> ProviderModelCatalogCacheRecord:
        existing = self.get(provider_id, configuration_fingerprint)
        models = list(existing.models if existing else [])
        model_id = str(model.get("id") or "").strip()
        if model_id and not any(str(item.get("id")) == model_id for item in models):
            models.append(dict(model))
        return self.save_success(
            provider_id=provider_id,
            configuration_fingerprint=configuration_fingerprint,
            models=models,
        )

    # -------------------------------------------------------------------------
    def clear_provider(self, provider_id: str) -> None:
        with self.session_factory() as db_session:
            db_session.execute(
                delete(ProviderModelCatalogCache).where(
                    ProviderModelCatalogCache.provider_id == provider_id
                )
            )
            db_session.commit()

    # -------------------------------------------------------------------------
    @staticmethod
    def _to_record(row: ProviderModelCatalogCache) -> ProviderModelCatalogCacheRecord:
        return ProviderModelCatalogCacheRecord(
            provider_id=row.provider_id,
            configuration_fingerprint=row.configuration_fingerprint,
            models=[dict(item) for item in (row.models or [])],
            last_success_at=row.last_success_at,
            last_attempt_at=row.last_attempt_at,
            last_attempt_status=row.last_attempt_status,
            last_error=row.last_error,
        )
