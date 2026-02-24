from __future__ import annotations

from cryptography.fernet import Fernet
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from DILIGENT.server.repositories.schemas.models import AccessKey, Base
from DILIGENT.server.repositories.serialization.accesskeys import AccessKeySerializer
from DILIGENT.server.services.keys.cryptography import decrypt, encrypt


# -----------------------------------------------------------------------------
def build_serializer() -> tuple[AccessKeySerializer, sessionmaker]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True)
    serializer = AccessKeySerializer(engine=engine, session_factory=factory)
    return serializer, factory


# -----------------------------------------------------------------------------
def test_encrypt_decrypt_roundtrip(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    fernet_key = Fernet.generate_key().decode("utf-8")
    monkeypatch.setenv("ACCESS_KEY_ENCRYPTION_KEY", fernet_key)

    plaintext = "test-openai-access-key-123"
    ciphertext = encrypt(plaintext)
    restored = decrypt(ciphertext)

    assert ciphertext != plaintext
    assert restored == plaintext


# -----------------------------------------------------------------------------
def test_stored_encrypted_value_never_contains_plaintext(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    fernet_key = Fernet.generate_key().decode("utf-8")
    monkeypatch.setenv("ACCESS_KEY_ENCRYPTION_KEY", fernet_key)
    serializer, factory = build_serializer()
    plaintext = "gemini-test-key-secret"

    created = serializer.create_key("gemini", plaintext)

    with factory() as db_session:
        stored = db_session.execute(
            select(AccessKey).where(AccessKey.id == created.id)
        ).scalar_one()

    assert stored.encrypted_value != plaintext
    assert plaintext not in stored.encrypted_value
    assert stored.fingerprint


# -----------------------------------------------------------------------------
def test_activation_keeps_only_one_active_key_per_provider(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    fernet_key = Fernet.generate_key().decode("utf-8")
    monkeypatch.setenv("ACCESS_KEY_ENCRYPTION_KEY", fernet_key)
    serializer, factory = build_serializer()

    first = serializer.create_key("openai", "openai-key-1")
    second = serializer.create_key("openai", "openai-key-2")
    serializer.activate_key(second.id)

    with factory() as db_session:
        rows = db_session.execute(
            select(AccessKey).where(AccessKey.provider == "openai")
        ).scalars().all()

    active_rows = [row for row in rows if row.is_active]
    assert len(active_rows) == 1
    assert active_rows[0].id == second.id
    assert any(row.id == first.id for row in rows)
