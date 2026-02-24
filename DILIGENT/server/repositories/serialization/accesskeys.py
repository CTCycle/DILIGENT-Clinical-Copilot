from __future__ import annotations

from datetime import datetime
from typing import Literal

from sqlalchemy import select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from DILIGENT.server.repositories.queries.data import DataRepositoryQueries
from DILIGENT.server.repositories.schemas.models import AccessKey
from DILIGENT.server.services.keys.cryptography import (
    encrypt as encrypt_access_key,
    fingerprint as build_fingerprint,
)

ProviderName = Literal["openai", "gemini"]
SUPPORTED_PROVIDERS = {"openai", "gemini"}


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
    def ensure_table(self) -> None:
        AccessKey.__table__.create(bind=self.engine, checkfirst=True)

    # -------------------------------------------------------------------------
    def list_keys(self, provider: str) -> list[AccessKey]:
        normalized_provider = self.normalize_provider(provider)
        self.ensure_table()
        db_session = self.session_factory()
        try:
            stmt = (
                select(AccessKey)
                .where(AccessKey.provider == normalized_provider)
                .order_by(AccessKey.is_active.desc(), AccessKey.created_at.desc(), AccessKey.id.desc())
            )
            return db_session.execute(stmt).scalars().all()
        finally:
            db_session.close()

    # -------------------------------------------------------------------------
    def create_key(self, provider: str, plaintext_key: str) -> AccessKey:
        normalized_provider = self.normalize_provider(provider)
        ciphertext = encrypt_access_key(plaintext_key)
        row = AccessKey(
            provider=normalized_provider,
            encrypted_value=ciphertext,
            fingerprint=build_fingerprint(ciphertext),
            is_active=False,
        )
        self.ensure_table()
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
    def activate_key(self, key_id: int) -> AccessKey:
        self.ensure_table()
        db_session = self.session_factory()
        now = datetime.now()
        try:
            target = db_session.get(AccessKey, key_id)
            if target is None:
                raise ValueError("Access key not found")

            db_session.execute(
                update(AccessKey)
                .where(AccessKey.provider == target.provider)
                .values(is_active=False, updated_at=now)
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
    def delete_key(self, key_id: int) -> bool:
        self.ensure_table()
        db_session = self.session_factory()
        try:
            target = db_session.get(AccessKey, key_id)
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
    def get_active_key(self, provider: str, *, mark_used: bool = False) -> AccessKey | None:
        normalized_provider = self.normalize_provider(provider)
        self.ensure_table()
        db_session = self.session_factory()
        try:
            stmt = (
                select(AccessKey)
                .where(
                    AccessKey.provider == normalized_provider,
                    AccessKey.is_active.is_(True),
                )
                .order_by(AccessKey.updated_at.desc(), AccessKey.id.desc())
            )
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
