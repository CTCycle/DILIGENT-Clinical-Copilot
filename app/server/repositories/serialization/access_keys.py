from __future__ import annotations

from datetime import datetime
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from repositories.database.session import (
    resolve_engine,
    resolve_session_factory,
)
from repositories.queries.access_keys import AccessKeyRepositoryQueries
from repositories.serialization.access_key_encryption import (
    AccessKeyEncryptionMaterialSerializer,
)
from repositories.schemas.models import AccessKey
from common.security.cryptography import (
    decrypt_with_key_material,
    encrypt_with_key_material,
    fingerprint_plaintext,
)
from domain.keys import ProviderName, normalize_access_key, normalize_provider_name


###############################################################################
class AccessKeySerializer:
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
        self.encryption_material_serializer = AccessKeyEncryptionMaterialSerializer(
            engine=self.engine,
            session_factory=self.session_factory,
        )

    # -------------------------------------------------------------------------
    def resolve_table(self, provider: ProviderName) -> type[AccessKey]:
        return AccessKey

    # -------------------------------------------------------------------------
    def resolve_table_from_row(
        self,
        row: AccessKey,
    ) -> type[AccessKey]:
        return AccessKey

    # -------------------------------------------------------------------------
    def get_key_by_id(
        self,
        db_session: Session,
        key_id: int,
        *,
        provider: str,
    ) -> AccessKey | None:
        normalized_provider = self.normalize_provider(provider)
        table = self.resolve_table(normalized_provider)
        return db_session.get(table, key_id)

    # -------------------------------------------------------------------------
    def list_keys(self, provider: str) -> list[AccessKey]:
        normalized_provider = self.normalize_provider(provider)
        table = self.resolve_table(normalized_provider)
        db_session = self.session_factory()
        try:
            stmt = AccessKeyRepositoryQueries.list_for_provider(
                table, normalized_provider
            )
            return db_session.execute(stmt).scalars().all()
        finally:
            db_session.close()

    # -------------------------------------------------------------------------
    def create_key(
        self, provider: str, plaintext_key: str
    ) -> AccessKey:
        normalized_provider = self.normalize_provider(provider)
        normalized_key = normalize_access_key(plaintext_key)
        table = self.resolve_table(normalized_provider)
        active_material = self.encryption_material_serializer.get_active_material()
        ciphertext = encrypt_with_key_material(
            normalized_key, active_material.key_material
        )
        row = table(
            provider=normalized_provider,
            encrypted_value=ciphertext,
            encryption_key_version=int(active_material.key_version),
            fingerprint=fingerprint_plaintext(normalized_key),
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
    def decrypt_key_row(self, row: AccessKey) -> str:
        version = getattr(row, "encryption_key_version", None)
        if version is None:
            raise RuntimeError(
                "Missing encryption key version metadata for stored provider key"
            )
        material = self.encryption_material_serializer.get_material_by_version(
            int(version)
        )
        if material is None:
            raise RuntimeError(
                f"Encryption material version {version} is not available"
            )
        return decrypt_with_key_material(row.encrypted_value, material.key_material)

    # -------------------------------------------------------------------------
    def activate_key(
        self,
        key_id: int,
        *,
        provider: str,
    ) -> AccessKey:
        db_session = self.session_factory()
        now = datetime.now()
        try:
            normalized_provider = self.normalize_provider(provider)
            target = self.get_key_by_id(
                db_session,
                key_id,
                provider=normalized_provider,
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
    def delete_key(self, key_id: int, *, provider: str) -> bool:
        db_session = self.session_factory()
        try:
            normalized_provider = self.normalize_provider(provider)
            target = self.get_key_by_id(
                db_session,
                key_id,
                provider=normalized_provider,
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
    ) -> AccessKey | None:
        normalized_provider = self.normalize_provider(provider)
        table = self.resolve_table(normalized_provider)
        db_session = self.session_factory()
        try:
            stmt = AccessKeyRepositoryQueries.active_for_provider(
                table, normalized_provider
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
        return normalize_provider_name(provider)

