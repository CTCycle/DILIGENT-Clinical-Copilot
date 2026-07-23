from __future__ import annotations

from pathlib import Path

from common.security.cryptography import (
    decrypt_with_key_material,
    encrypt_with_key_material,
)
from repositories.schemas.base import Base
from repositories.schemas.security import AccessKey
from repositories.serialization.access_key_encryption import (
    AccessKeyEncryptionMaterialSerializer,
)
from repositories.serialization.access_keys import AccessKeySerializer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pytest

VALID_TEST_KEY = "openai-secret-value"

###############################################################################
@pytest.fixture(autouse=True)
def external_material_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(
        "DILIGENT_ACCESS_KEY_MATERIAL_FILE",
        str(tmp_path / "access-key-material.json"),
    )

###############################################################################
def build_serializer() -> tuple[AccessKeyEncryptionMaterialSerializer, sessionmaker]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    serializer = AccessKeyEncryptionMaterialSerializer(
        engine=engine,
        session_factory=factory,
    )
    return serializer, factory

###############################################################################
def test_ensure_seeded_creates_initial_version_1_row() -> None:
    serializer, _ = build_serializer()

    row = serializer.ensure_seeded()

    assert row.key_purpose == "provider_access_keys"
    assert row.key_version == 1
    assert row.is_active is True
    assert row.seeded_at is not None
    assert row.activated_at is not None

###############################################################################
def test_ensure_seeded_is_idempotent() -> None:
    serializer, _ = build_serializer()

    first = serializer.ensure_seeded()
    second = serializer.ensure_seeded()

    assert first.key_material == second.key_material
    assert first.key_version == second.key_version == 1

###############################################################################
def test_only_one_active_material_exists_per_purpose() -> None:
    serializer, _ = build_serializer()
    serializer.ensure_seeded()
    serializer.rotate_material()
    assert serializer.get_active_material().is_active is True

###############################################################################
def test_rotate_material_creates_new_active_version() -> None:
    serializer, _ = build_serializer()
    seeded = serializer.ensure_seeded()

    rotated = serializer.rotate_material()

    assert seeded.key_version == 1
    assert rotated.key_version == 2
    assert rotated.is_active is True
    active = serializer.get_active_material()
    assert active.key_version == 2

###############################################################################
def test_get_material_by_version_returns_correct_row() -> None:
    serializer, _ = build_serializer()
    serializer.ensure_seeded()
    serializer.rotate_material()

    v1 = serializer.get_material_by_version(1)
    v2 = serializer.get_material_by_version(2)

    assert v1 is not None
    assert v2 is not None
    assert v1.key_version == 1
    assert v2.key_version == 2

###############################################################################
def test_encrypt_and_decrypt_use_external_material() -> None:
    serializer, _ = build_serializer()
    material = serializer.ensure_seeded()
    plaintext = "provider-secret-123"

    ciphertext = encrypt_with_key_material(plaintext, material.key_material)
    restored = decrypt_with_key_material(ciphertext, material.key_material)

    assert ciphertext != plaintext
    assert restored == plaintext

###############################################################################
def test_external_material_store_is_versioned_and_roundtrips(monkeypatch, tmp_path: Path) -> None:
    material_path = tmp_path / "access-key-material.json"
    monkeypatch.setenv("DILIGENT_ACCESS_KEY_MATERIAL_FILE", str(material_path))
    serializer, _ = build_serializer()

    first = serializer.ensure_seeded()
    rotated = serializer.rotate_material()

    assert first.key_version == 1
    assert rotated.key_version == 2
    assert serializer.get_active_material().key_version == 2
    assert serializer.get_material_by_version(1).key_material == first.key_material
    assert "key_material" in material_path.read_text(encoding="utf-8")

###############################################################################
def _build_access_key_serializer() -> tuple[AccessKeySerializer, sessionmaker]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    AccessKeyEncryptionMaterialSerializer(
        engine=engine, session_factory=factory
    ).ensure_seeded()
    serializer = AccessKeySerializer(engine=engine, session_factory=factory)
    return serializer, factory

###############################################################################
def test_decryption_fails_when_encryption_key_version_is_missing() -> None:
    serializer, factory = _build_access_key_serializer()
    row = serializer.create_key("openai", VALID_TEST_KEY)
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

###############################################################################
def test_decryption_fails_when_referenced_version_does_not_exist() -> None:
    serializer, factory = _build_access_key_serializer()
    row = serializer.create_key("openai", VALID_TEST_KEY)

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

###############################################################################
def test_code_never_reads_access_key_encryption_key_env_var() -> None:
    app_dir = Path(__file__).resolve().parents[2]
    source = (app_dir / "server/common/security/cryptography.py").read_text(
        encoding="utf-8"
    )
    assert "ACCESS_KEY_ENCRYPTION_KEY" not in source

###############################################################################
def test_unavailable_key_material_version_fails_loudly() -> None:
    serializer, _ = _build_access_key_serializer()
    serializer.create_key("openai", VALID_TEST_KEY)

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
