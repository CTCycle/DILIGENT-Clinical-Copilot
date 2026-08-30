from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cryptography.fernet import Fernet

from common.paths import RESOURCES_PATH, ROOT_DIR

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
    """Load versioned Fernet material from a protected external file.

    Encryption material is deliberately not represented by an ORM model. The
    database stores only ciphertext and the referenced version number.
    """

    # -------------------------------------------------------------------------
    def __init__(self, *, engine=None, session_factory=None) -> None:
        # Keep the constructor shape used by the repository serializers while
        # making it explicit that database handles are no longer consumed.
        _ = engine, session_factory

    # -------------------------------------------------------------------------
    @staticmethod
    def external_path() -> Path:
        configured = os.getenv(EXTERNAL_KEY_FILE_ENV, "").strip()
        if not configured:
            return RESOURCES_PATH / "access-key-material.json"
        path = Path(configured).expanduser()
        return path if path.is_absolute() else ROOT_DIR / path

    # -------------------------------------------------------------------------
    @classmethod
    def _read_store(cls, path: Path) -> dict[str, object]:
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("External access-key material file is invalid") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(
                "External access-key material file must contain an object"
            )
        return payload

    # -------------------------------------------------------------------------
    @staticmethod
    def _write_store(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(path)

    # -------------------------------------------------------------------------
    @staticmethod
    def _material(
        purpose: str, version: int, record: dict[str, object]
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
        )

    # -------------------------------------------------------------------------
    @classmethod
    def _get_version_record(
        cls, payload: dict[str, object], purpose: str, version: int
    ) -> dict[str, object] | None:
        purpose_store = payload.get(purpose)
        if not isinstance(purpose_store, dict):
            return None
        versions = purpose_store.get("versions")
        if not isinstance(versions, dict):
            return None
        record = versions.get(str(int(version)))
        return record if isinstance(record, dict) else None

    # -------------------------------------------------------------------------
    def ensure_seeded(
        self, purpose: str = DEFAULT_KEY_PURPOSE
    ) -> ExternalEncryptionMaterial:
        path = self.external_path()
        payload = self._read_store(path)
        purpose_store = payload.get(purpose)
        if isinstance(purpose_store, dict):
            active_version = int(purpose_store.get("active_version", 0))
            existing_record = self._get_version_record(payload, purpose, active_version)
            if active_version and existing_record is not None:
                return self._material(purpose, active_version, existing_record)

        now = datetime.now(UTC).replace(tzinfo=None)
        version = 1
        record: dict[str, object] = {
            "key_material": Fernet.generate_key().decode("utf-8"),
            "is_active": True,
            "seeded_at": now.isoformat(),
            "activated_at": now.isoformat(),
        }
        payload[purpose] = {
            "active_version": version,
            "versions": {str(version): record},
        }
        self._write_store(path, payload)
        return self._material(purpose, version, record)

    # -------------------------------------------------------------------------
    def get_active_material(
        self, purpose: str = DEFAULT_KEY_PURPOSE
    ) -> ExternalEncryptionMaterial:
        return self.ensure_seeded(purpose)

    # -------------------------------------------------------------------------
    def get_material_by_version(
        self, version: int, purpose: str = DEFAULT_KEY_PURPOSE
    ) -> ExternalEncryptionMaterial | None:
        payload = self._read_store(self.external_path())
        record = self._get_version_record(payload, purpose, int(version))
        return (
            self._material(purpose, int(version), record)
            if record is not None
            else None
        )

    # -------------------------------------------------------------------------
    def rotate_material(
        self, purpose: str = DEFAULT_KEY_PURPOSE
    ) -> ExternalEncryptionMaterial:
        path = self.external_path()
        payload = self._read_store(path)
        purpose_store = payload.get(purpose)
        if not isinstance(purpose_store, dict):
            raise RuntimeError(
                f"No active encryption material configured for {purpose}"
            )
        active_version = int(purpose_store.get("active_version", 0))
        active = self._get_version_record(payload, purpose, active_version)
        versions = purpose_store.get("versions")
        if not active_version or active is None or not isinstance(versions, dict):
            raise RuntimeError(
                f"No active encryption material configured for {purpose}"
            )

        now = datetime.now(UTC).replace(tzinfo=None)
        active["is_active"] = False
        active["deactivated_at"] = now.isoformat()
        next_version = active_version + 1
        created: dict[str, object] = {
            "key_material": Fernet.generate_key().decode("utf-8"),
            "is_active": True,
            "seeded_at": now.isoformat(),
            "activated_at": now.isoformat(),
        }
        versions[str(next_version)] = created
        purpose_store["active_version"] = next_version
        self._write_store(path, payload)
        return self._material(purpose, next_version, created)
