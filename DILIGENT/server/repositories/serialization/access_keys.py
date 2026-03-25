from __future__ import annotations

from datetime import datetime
from typing import Literal

from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from DILIGENT.server.repositories.queries.access_keys import AccessKeyRepositoryQueries
from DILIGENT.server.repositories.queries.data import DataRepositoryQueries
from DILIGENT.server.repositories.schemas.models import AccessKey, ResearchAccessKey
from DILIGENT.server.services.cryptography import (
    encrypt as encrypt_access_key,
    fingerprint as build_fingerprint,
)

ProviderName = Literal["openai", "gemini", "tavily"]
SUPPORTED_PROVIDERS = {"openai", "gemini", "tavily"}
RESEARCH_PROVIDER = "tavily"


###############################################################################
class AccessKeySerializer:
    def __init__(
        self,
        queries: DataRepositoryQueries | None = None,
        *,
        engine: Engine | None = None,
        session_factory: sessionmaker | None = None,
    ) -> None:
        self.queries = queries or DataRepositoryQueries()
        resolved_engine = engine or self.queries.database.backend.engine  # type: ignore[attr-defined]
        self.engine = resolved_engine
        self.session_factory = session_factory or sessionmaker(
            bind=resolved_engine,
            future=True,
            expire_on_commit=False,
        )

    # -------------------------------------------------------------------------
    def ensure_research_table(self) -> None:
        ResearchAccessKey.__table__.create(bind=self.engine, checkfirst=True)

    # -------------------------------------------------------------------------
    def resolve_table(self, provider: ProviderName):
        if provider == RESEARCH_PROVIDER:
            self.ensure_research_table()
            return ResearchAccessKey
        return AccessKey

    # -------------------------------------------------------------------------
    def resolve_table_from_row(
        self,
        row: AccessKey | ResearchAccessKey,
    ) -> type[AccessKey] | type[ResearchAccessKey]:
        if str(row.provider).strip().lower() == RESEARCH_PROVIDER:
            self.ensure_research_table()
            return ResearchAccessKey
        return AccessKey

    # -------------------------------------------------------------------------
    def get_key_by_id(
        self,
        db_session,
        key_id: int,
        *,
        provider: str | None = None,
    ) -> AccessKey | ResearchAccessKey | None:
        if provider is not None:
            normalized_provider = self.normalize_provider(provider)
            table = self.resolve_table(normalized_provider)
            return db_session.get(table, key_id)

        target = db_session.get(AccessKey, key_id)
        if target is not None:
            return target

        self.ensure_research_table()
        return db_session.get(ResearchAccessKey, key_id)

    # -------------------------------------------------------------------------
    def list_keys(self, provider: str) -> list[AccessKey | ResearchAccessKey]:
        normalized_provider = self.normalize_provider(provider)
        table = self.resolve_table(normalized_provider)
        db_session = self.session_factory()
        try:
            stmt = AccessKeyRepositoryQueries.list_for_provider(table, normalized_provider)
            return db_session.execute(stmt).scalars().all()
        finally:
            db_session.close()

    # -------------------------------------------------------------------------
    def create_key(self, provider: str, plaintext_key: str) -> AccessKey | ResearchAccessKey:
        normalized_provider = self.normalize_provider(provider)
        table = self.resolve_table(normalized_provider)
        ciphertext = encrypt_access_key(plaintext_key)
        row = table(
            provider=normalized_provider,
            encrypted_value=ciphertext,
            fingerprint=build_fingerprint(ciphertext),
            is_active=False,
        )
        db_session = self.session_factory()
        try:
            db_session.add(row)
            db_session.commit()
            db_session.refresh(row)
            return row
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    # -------------------------------------------------------------------------
    def activate_key(
        self,
        key_id: int,
        *,
        provider: str | None = None,
    ) -> AccessKey | ResearchAccessKey:
        db_session = self.session_factory()
        now = datetime.now()
        try:
            target = self.get_key_by_id(
                db_session,
                key_id,
                provider=provider,
            )
            if target is None:
                raise ValueError("Access key not found")
            table = self.resolve_table_from_row(target)

            db_session.execute(
                AccessKeyRepositoryQueries.deactivate_provider_keys(
                    table, target.provider, now=now
                )
            )
            target.is_active = True
            target.updated_at = now
            db_session.commit()
            db_session.refresh(target)
            return target
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    # -------------------------------------------------------------------------
    def delete_key(self, key_id: int, *, provider: str | None = None) -> bool:
        db_session = self.session_factory()
        try:
            target = self.get_key_by_id(
                db_session,
                key_id,
                provider=provider,
            )
            if target is None:
                return False
            db_session.delete(target)
            db_session.commit()
            return True
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    # -------------------------------------------------------------------------
    def get_active_key(
        self,
        provider: str,
        *,
        mark_used: bool = False,
    ) -> AccessKey | ResearchAccessKey | None:
        normalized_provider = self.normalize_provider(provider)
        table = self.resolve_table(normalized_provider)
        db_session = self.session_factory()
        try:
            stmt = AccessKeyRepositoryQueries.active_for_provider(table, normalized_provider)
            row = db_session.execute(stmt).scalars().first()
            if row is None:
                return None
            if mark_used:
                row.last_used_at = datetime.now()
                row.updated_at = datetime.now()
                db_session.commit()
                db_session.refresh(row)
            return row
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_provider(provider: str) -> ProviderName:
        normalized = str(provider or "").strip().lower()
        if normalized not in SUPPORTED_PROVIDERS:
            raise ValueError("Unsupported provider")
        return normalized  # type: ignore[return-value]
