from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from repositories.schemas.models import (
    AccessKey,
    Base,
)
from repositories.serialization.access_key_encryption import (
    AccessKeyEncryptionMaterialSerializer,
)
from repositories.serialization.access_keys import AccessKeySerializer


# -----------------------------------------------------------------------------
def build_serializer() -> tuple[AccessKeySerializer, sessionmaker]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    AccessKeyEncryptionMaterialSerializer(
        engine=engine, session_factory=factory
    ).ensure_seeded()
    serializer = AccessKeySerializer(engine=engine, session_factory=factory)
    return serializer, factory


# -----------------------------------------------------------------------------
def test_decryption_fails_when_encryption_key_version_is_missing() -> None:
    serializer, factory = build_serializer()
    row = serializer.create_key("openai", "openai-secret")
    with factory() as db_session:
        loaded = db_session.get(AccessKey, row.id)
        assert loaded is not None
        stale_row = AccessKey(
            provider=loaded.provider,
            encrypted_value=loaded.encrypted_value,
            encryption_key_version=None,  # type: ignore[arg-type]
            fingerprint=loaded.fingerprint,
            is_active=loaded.is_active,
        )

    try:
        serializer.decrypt_key_row(stale_row)
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "Missing encryption key version metadata" in str(exc)


# -----------------------------------------------------------------------------
def test_decryption_fails_when_referenced_version_does_not_exist() -> None:
    serializer, factory = build_serializer()
    row = serializer.create_key("openai", "openai-secret")

    with factory() as db_session:
        loaded = db_session.get(AccessKey, row.id)
        assert loaded is not None
        loaded.encryption_key_version = 9999
        db_session.commit()
        db_session.refresh(loaded)

    try:
        serializer.decrypt_key_row(loaded)  # type: ignore[arg-type]
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "is not available" in str(exc)


# -----------------------------------------------------------------------------
def test_code_never_reads_access_key_encryption_key_env_var() -> None:
    app_dir = Path(__file__).resolve().parents[2]
    source = (app_dir / "server/common/security/cryptography.py").read_text(
        encoding="utf-8"
    )
    assert "ACCESS_KEY_ENCRYPTION_KEY" not in source


# -----------------------------------------------------------------------------
def test_unavailable_key_material_version_fails_loudly() -> None:
    serializer, _ = build_serializer()
    serializer.create_key("openai", "openai-secret")

    stale_row = AccessKey(
        provider="openai",
        encrypted_value="stale-ciphertext",
        encryption_key_version=1234,
        fingerprint="stale-fingerprint",
        is_active=True,
    )
    try:
        serializer.decrypt_key_row(stale_row)
        assert False, "Expected RuntimeError for unavailable material version"
    except RuntimeError as exc:
        assert "is not available" in str(exc)


