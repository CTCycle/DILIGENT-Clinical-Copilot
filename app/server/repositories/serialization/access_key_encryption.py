from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cryptography.fernet import Fernet
from sqlalchemy import select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from repositories.schemas.models import AccessKeyEncryptionMaterial

DEFAULT_KEY_PURPOSE = "provider_access_keys"
EXTERNAL_KEY_FILE_ENV = "DILIGENT_ACCESS_KEY_MATERIAL_FILE"

###############################################################################
@dataclass(frozen=True)
class ExternalEncryptionMaterial:
    key_purpose: str
    key_version: int
    key_material: str
    is_active: bool
    seeded_at: datetime
    activated_at: datetime
    deactivated_at: datetime | None = None
    id: int | None = None

###############################################################################
class AccessKeyEncryptionMaterialSerializer:

    # -------------------------------------------------------------------------
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
    @staticmethod
    def external_path() -> Path | None:
        configured = os.getenv(EXTERNAL_KEY_FILE_ENV, "").strip()
        return Path(configured).expanduser() if configured else None

    # -------------------------------------------------------------------------
    @classmethod
    def _read_external_store(cls, path: Path) -> dict[str, object]:
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("External access-key material file is invalid") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("External access-key material file must contain an object")
        return payload

    # -------------------------------------------------------------------------
    @staticmethod
    def _write_external_store(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(path)

    # -------------------------------------------------------------------------
    @classmethod
    def _external_material(
        cls, path: Path, purpose: str, version: int, record: dict[str, object]
    ) -> ExternalEncryptionMaterial:
        seeded_at = datetime.fromisoformat(str(record["seeded_at"]))
        activated_at = datetime.fromisoformat(str(record["activated_at"]))
        deactivated = record.get("deactivated_at")
        return ExternalEncryptionMaterial(
            key_purpose=purpose,
            key_version=version,
            key_material=str(record["key_material"]),
            is_active=bool(record.get("is_active", False)),
            seeded_at=seeded_at,
            activated_at=activated_at,
            deactivated_at=(
                datetime.fromisoformat(str(deactivated)) if deactivated else None
            ),
            id=None,
        )

    # -------------------------------------------------------------------------
    def ensure_seeded(
        self, purpose: str = DEFAULT_KEY_PURPOSE
    ) -> AccessKeyEncryptionMaterial:
        external_path = self.external_path()
        if external_path is not None:
            payload = self._read_external_store(external_path)
            purpose_store = payload.get(purpose)
            if isinstance(purpose_store, dict):
                active_version = int(purpose_store.get("active_version", 0))
                versions = purpose_store.get("versions")
                if active_version and isinstance(versions, dict):
                    record = versions.get(str(active_version))
                    if isinstance(record, dict):
                        return self._external_material(
                            external_path, purpose, active_version, record
                        )
            now = datetime.now(UTC).replace(tzinfo=None)
            key_version = 1
            record = {
                "key_material": Fernet.generate_key().decode("utf-8"),
                "is_active": True,
                "seeded_at": now.isoformat(),
                "activated_at": now.isoformat(),
            }
            payload[purpose] = {
                "active_version": key_version,
                "versions": {str(key_version): record},
            }
            self._write_external_store(external_path, payload)
            return self._external_material(external_path, purpose, key_version, record)
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
        external_path = self.external_path()
        if external_path is not None:
            seeded = self.ensure_seeded(purpose)
            return seeded
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
        external_path = self.external_path()
        if external_path is not None:
            payload = self._read_external_store(external_path)
            purpose_store = payload.get(purpose)
            if not isinstance(purpose_store, dict):
                return None
            versions = purpose_store.get("versions")
            if not isinstance(versions, dict):
                return None
            record = versions.get(str(int(version)))
            if not isinstance(record, dict):
                return None
            return self._external_material(external_path, purpose, int(version), record)
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
        external_path = self.external_path()
        if external_path is not None:
            payload = self._read_external_store(external_path)
            purpose_store = payload.get(purpose)
            if not isinstance(purpose_store, dict):
                raise RuntimeError(f"No active encryption material configured for {purpose}")
            active_version = int(purpose_store.get("active_version", 0))
            versions = purpose_store.get("versions")
            if not active_version or not isinstance(versions, dict):
                raise RuntimeError(f"No active encryption material configured for {purpose}")
            active = versions.get(str(active_version))
            if not isinstance(active, dict):
                raise RuntimeError(f"No active encryption material configured for {purpose}")
            now = datetime.now(UTC).replace(tzinfo=None)
            active["is_active"] = False
            active["deactivated_at"] = now.isoformat()
            next_version = active_version + 1
            created = {
                "key_material": Fernet.generate_key().decode("utf-8"),
                "is_active": True,
                "seeded_at": now.isoformat(),
                "activated_at": now.isoformat(),
            }
            versions[str(next_version)] = created
            purpose_store["active_version"] = next_version
            self._write_external_store(external_path, payload)
            return self._external_material(external_path, purpose, next_version, created)
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
