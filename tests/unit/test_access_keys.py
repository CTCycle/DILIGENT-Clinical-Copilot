from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from DILIGENT.server.repositories.schemas.models import (
    AccessKey,
    Base,
    ResearchAccessKey,
)
from DILIGENT.server.repositories.serialization.access_key_encryption import (
    AccessKeyEncryptionMaterialSerializer,
)
from DILIGENT.server.repositories.serialization.access_keys import AccessKeySerializer


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

    first = serializer.create_key("openai", "openai-key-1")
    second = serializer.create_key("openai", "openai-key-2")
    serializer.activate_key(second.id, provider="openai")

    with factory() as db_session:
        rows = db_session.execute(
            select(AccessKey).where(AccessKey.provider == "openai")
        ).scalars().all()

    active_rows = [row for row in rows if row.is_active]
    assert len(active_rows) == 1
    assert active_rows[0].id == second.id
    assert any(row.id == first.id for row in rows)


# -----------------------------------------------------------------------------
def test_tavily_keys_are_stored_in_research_table() -> None:
    serializer, factory = build_serializer()

    created = serializer.create_key("tavily", "tvly-secret")

    with factory() as db_session:
        stored = db_session.execute(
            select(ResearchAccessKey).where(ResearchAccessKey.id == created.id)
        ).scalar_one()

    assert stored.provider == "tavily"
    assert stored.encrypted_value != "tvly-secret"
    assert stored.fingerprint
    assert stored.encryption_key_version == 1


# -----------------------------------------------------------------------------
def test_provider_scoped_activate_and_delete_support_tavily() -> None:
    serializer, factory = build_serializer()

    openai = serializer.create_key("openai", "openai-key")
    tavily = serializer.create_key("tavily", "tavily-key")

    # Ids can overlap because keys are stored in different tables.
    assert openai.id == tavily.id

    activated_tavily = serializer.activate_key(tavily.id, provider="tavily")
    assert activated_tavily.provider == "tavily"
    assert activated_tavily.is_active is True

    with factory() as db_session:
        openai_row = db_session.execute(
            select(AccessKey).where(AccessKey.id == openai.id)
        ).scalar_one()
        tavily_row = db_session.execute(
            select(ResearchAccessKey).where(ResearchAccessKey.id == tavily.id)
        ).scalar_one()

    assert openai_row.is_active is False
    assert tavily_row.is_active is True

    deleted = serializer.delete_key(tavily.id, provider="tavily")
    assert deleted is True
    assert serializer.get_active_key("tavily") is None


# -----------------------------------------------------------------------------
def test_decrypt_key_row_uses_db_seeded_material() -> None:
    serializer, _factory = build_serializer()
    plaintext = "sk-live-example"
    created = serializer.create_key("openai", plaintext)

    restored = serializer.decrypt_key_row(created)

    assert restored == plaintext
