from __future__ import annotations

from sqlalchemy import create_engine, select
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
    factory = sessionmaker(bind=engine, future=True)
    AccessKeyEncryptionMaterialSerializer(
        engine=engine,
        session_factory=factory,
    ).ensure_seeded()
    serializer = AccessKeySerializer(engine=engine, session_factory=factory)
    return serializer, factory


# -----------------------------------------------------------------------------
def test_stored_encrypted_value_never_contains_plaintext() -> None:
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
    assert stored.encryption_key_version == 1


# -----------------------------------------------------------------------------
def test_activation_keeps_only_one_active_key_per_provider() -> None:
    serializer, factory = build_serializer()

    first = serializer.create_key("openai", "openai-key-1-secret")
    second = serializer.create_key("openai", "openai-key-2-secret")
    serializer.activate_key(second.id, provider="openai")

    with factory() as db_session:
        rows = (
            db_session.execute(select(AccessKey).where(AccessKey.provider == "openai"))
            .scalars()
            .all()
        )

    active_rows = [row for row in rows if row.is_active]
    assert len(active_rows) == 1
    assert active_rows[0].id == second.id
    assert any(row.id == first.id for row in rows)


# -----------------------------------------------------------------------------
def test_provider_scoped_activate_and_delete_for_openrouter() -> None:
    serializer, factory = build_serializer()

    openai = serializer.create_key("openai", "openai-key-secret")
    openrouter = serializer.create_key("openrouter", "openrouter-key-secret")
    activated_openrouter = serializer.activate_key(openrouter.id, provider="openrouter")
    assert activated_openrouter.provider == "openrouter"
    assert activated_openrouter.is_active is True

    with factory() as db_session:
        openai_row = db_session.execute(
            select(AccessKey).where(AccessKey.id == openai.id)
        ).scalar_one()
        openrouter_row = db_session.execute(
            select(AccessKey).where(AccessKey.id == openrouter.id)
        ).scalar_one()

    assert openai_row.is_active is False
    assert openrouter_row.is_active is True

    deleted = serializer.delete_key(openrouter.id, provider="openrouter")
    assert deleted is True
    assert serializer.get_active_key("openrouter") is None


# -----------------------------------------------------------------------------
def test_decrypt_key_row_uses_db_seeded_material() -> None:
    serializer, _factory = build_serializer()
    plaintext = "sk-live-example-secret"
    created = serializer.create_key("openai", plaintext)

    restored = serializer.decrypt_key_row(created)

    assert restored == plaintext


# -----------------------------------------------------------------------------
def test_rejects_too_short_access_key() -> None:
    serializer, _factory = build_serializer()

    try:
        serializer.create_key("openai", "short")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected short access key to be rejected")

