# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from pathlib import Path

from repositories.schemas.base import Base
from repositories.schemas.security import AccessKey
from repositories.serialization.access_key_encryption import (
    AccessKeyEncryptionMaterialSerializer,
)
from repositories.serialization.access_keys import AccessKeySerializer
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

###############################################################################
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

###############################################################################
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

###############################################################################
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

###############################################################################
def test_provider_scoped_activate_and_delete_for_brave() -> None:
    serializer, factory = build_serializer()

    openai = serializer.create_key("openai", "openai-key-secret")
    brave = serializer.create_key("brave", "brave-key-secret")
    activated_brave = serializer.activate_key(brave.id, provider="brave")
    assert activated_brave.provider == "brave"
    assert activated_brave.is_active is True

    with factory() as db_session:
        openai_row = db_session.execute(
            select(AccessKey).where(AccessKey.id == openai.id)
        ).scalar_one()
        brave_row = db_session.execute(
            select(AccessKey).where(AccessKey.id == brave.id)
        ).scalar_one()

    assert openai_row.is_active is False
    assert brave_row.is_active is True

    deleted = serializer.delete_key(brave.id, provider="brave")
    assert deleted is True
    assert serializer.get_active_key("brave") is None

###############################################################################
def test_decrypt_key_row_uses_db_seeded_material() -> None:
    serializer, factory = build_serializer()
    plaintext = "sk-live-example-secret"
    created = serializer.create_key("openai", plaintext)

    with factory() as db_session:
        loaded = db_session.get(AccessKey, created.id)
        assert loaded is not None
    restored = serializer.decrypt_key_row(loaded)

    assert restored == plaintext

###############################################################################
def test_file_backed_access_key_survives_serializer_reopen(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    database_path = tmp_path / "access-keys.sqlite3"
    material_path = tmp_path / "access-key-material.json"
    plaintext = "sk-proj-reopen-persistence-20260929"
    monkeypatch.setenv("DILIGENT_ACCESS_KEY_MATERIAL_FILE", str(material_path))

    first_engine = create_engine(
        f"sqlite+pysqlite:///{database_path}", future=True
    )
    Base.metadata.create_all(first_engine)
    first_factory = sessionmaker(
        bind=first_engine, future=True, expire_on_commit=False
    )
    AccessKeyEncryptionMaterialSerializer(
        engine=first_engine,
        session_factory=first_factory,
    ).ensure_seeded()
    first_serializer = AccessKeySerializer(
        engine=first_engine,
        session_factory=first_factory,
    )
    created = first_serializer.create_key("openai", plaintext)
    activated = first_serializer.activate_key(created.id, provider="openai")
    assert activated.is_active is True

    with first_factory() as db_session:
        stored_before_reopen = db_session.get(AccessKey, created.id)
        assert stored_before_reopen is not None
        encrypted_value = stored_before_reopen.encrypted_value
        assert plaintext not in encrypted_value

    first_engine.dispose()

    second_engine = create_engine(
        f"sqlite+pysqlite:///{database_path}", future=True
    )
    second_factory = sessionmaker(
        bind=second_engine, future=True, expire_on_commit=False
    )
    second_serializer = AccessKeySerializer(
        engine=second_engine,
        session_factory=second_factory,
    )
    reopened = second_serializer.get_active_key("openai")
    assert reopened is not None
    assert reopened.id == created.id
    assert reopened.provider == "openai"
    assert reopened.is_active is True

    with second_factory() as db_session:
        stored_after_reopen = db_session.get(AccessKey, created.id)
        assert stored_after_reopen is not None
        assert stored_after_reopen.encrypted_value == encrypted_value
        assert plaintext not in stored_after_reopen.encrypted_value
        assert second_serializer.decrypt_key_row(stored_after_reopen) == plaintext

    assert second_serializer.delete_key(created.id, provider="openai") is True
    second_engine.dispose()

    third_engine = create_engine(
        f"sqlite+pysqlite:///{database_path}", future=True
    )
    third_factory = sessionmaker(
        bind=third_engine, future=True, expire_on_commit=False
    )
    third_serializer = AccessKeySerializer(
        engine=third_engine,
        session_factory=third_factory,
    )
    assert third_serializer.list_keys("openai") == []
    assert third_serializer.get_active_key("openai") is None
    third_engine.dispose()

###############################################################################
def test_rejects_too_short_access_key() -> None:
    serializer, _factory = build_serializer()

    try:
        serializer.create_key("openai", "short")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected short access key to be rejected")

###############################################################################
def test_rejects_placeholder_key_without_displacing_active_key() -> None:
    serializer, factory = build_serializer()
    active = serializer.create_key("openai", "sk-live-valid-secret-value")
    serializer.activate_key(active.id, provider="openai")

    from services.security.access_keys import AccessKeyService

    service = AccessKeyService(serializer=serializer)
    try:
        service.create_access_key("openai", "sk-fake-temporary-placeholder-key")
    except ValueError as exc:
        assert "Openai" in str(exc)
    else:
        raise AssertionError("Expected placeholder key to be rejected")

    with factory() as db_session:
        rows = (
            db_session.execute(select(AccessKey).where(AccessKey.provider == "openai"))
            .scalars()
            .all()
        )
    assert [row.id for row in rows if row.is_active] == [active.id]
