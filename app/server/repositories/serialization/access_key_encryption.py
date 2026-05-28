from __future__ import annotations

from datetime import UTC, datetime

from cryptography.fernet import Fernet
from sqlalchemy import select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from repositories.schemas.models import AccessKeyEncryptionMaterial

DEFAULT_KEY_PURPOSE = "provider_access_keys"


###############################################################################
class AccessKeyEncryptionMaterialSerializer:
    def __init__(
        self,
        *,
        engine: Engine | None = None,
        session_factory: sessionmaker | None = None,
    ) -> None:
        if engine is None and session_factory is None:
            raise ValueError("engine or session_factory is required")
        self.engine = engine
        self.session_factory = session_factory or sessionmaker(
            bind=engine,
            future=True,
            expire_on_commit=False,
        )

    # -------------------------------------------------------------------------
    def ensure_seeded(
        self, purpose: str = DEFAULT_KEY_PURPOSE
    ) -> AccessKeyEncryptionMaterial:
        db_session = self.session_factory()
        try:
            existing = (
                db_session.execute(
                    select(AccessKeyEncryptionMaterial)
                    .where(AccessKeyEncryptionMaterial.key_purpose == purpose)
                    .order_by(
                        AccessKeyEncryptionMaterial.key_version.desc(),
                        AccessKeyEncryptionMaterial.id.desc(),
                    )
                )
                .scalars()
                .first()
            )
            if existing is not None:
                return existing

            now = datetime.now(UTC).replace(tzinfo=None)
            created = AccessKeyEncryptionMaterial(
                key_purpose=purpose,
                key_version=1,
                key_material=Fernet.generate_key().decode("utf-8"),
                is_active=True,
                seeded_at=now,
                activated_at=now,
            )
            db_session.add(created)
            db_session.commit()
            db_session.refresh(created)
            return created
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    # -------------------------------------------------------------------------
    def get_active_material(
        self, purpose: str = DEFAULT_KEY_PURPOSE
    ) -> AccessKeyEncryptionMaterial:
        db_session = self.session_factory()
        try:
            row = (
                db_session.execute(
                    select(AccessKeyEncryptionMaterial).where(
                        AccessKeyEncryptionMaterial.key_purpose == purpose,
                        AccessKeyEncryptionMaterial.is_active.is_(True),
                    )
                )
                .scalars()
                .first()
            )
            if row is None:
                raise RuntimeError(
                    f"No active encryption material configured for {purpose}"
                )
            return row
        finally:
            db_session.close()

    # -------------------------------------------------------------------------
    def get_material_by_version(
        self,
        version: int,
        purpose: str = DEFAULT_KEY_PURPOSE,
    ) -> AccessKeyEncryptionMaterial | None:
        db_session = self.session_factory()
        try:
            return (
                db_session.execute(
                    select(AccessKeyEncryptionMaterial).where(
                        AccessKeyEncryptionMaterial.key_purpose == purpose,
                        AccessKeyEncryptionMaterial.key_version == version,
                    )
                )
                .scalars()
                .first()
            )
        finally:
            db_session.close()

    # -------------------------------------------------------------------------
    def rotate_material(
        self, purpose: str = DEFAULT_KEY_PURPOSE
    ) -> AccessKeyEncryptionMaterial:
        db_session = self.session_factory()
        try:
            active = (
                db_session.execute(
                    select(AccessKeyEncryptionMaterial).where(
                        AccessKeyEncryptionMaterial.key_purpose == purpose,
                        AccessKeyEncryptionMaterial.is_active.is_(True),
                    )
                )
                .scalars()
                .first()
            )
            if active is None:
                raise RuntimeError(
                    f"No active encryption material configured for {purpose}"
                )

            now = datetime.now(UTC).replace(tzinfo=None)
            next_version = int(active.key_version) + 1
            db_session.execute(
                update(AccessKeyEncryptionMaterial)
                .where(
                    AccessKeyEncryptionMaterial.key_purpose == purpose,
                    AccessKeyEncryptionMaterial.is_active.is_(True),
                )
                .values(
                    is_active=False,
                    deactivated_at=now,
                    updated_at=now,
                )
            )
            created = AccessKeyEncryptionMaterial(
                key_purpose=purpose,
                key_version=next_version,
                key_material=Fernet.generate_key().decode("utf-8"),
                is_active=True,
                seeded_at=now,
                activated_at=now,
            )
            db_session.add(created)
            db_session.commit()
            db_session.refresh(created)
            return created
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()
