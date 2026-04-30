from __future__ import annotations

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from repositories.schemas.models import (
    AccessKeyEncryptionMaterial,
    Base,
)
from repositories.serialization.access_key_encryption import (
    AccessKeyEncryptionMaterialSerializer,
)
from services.security.cryptography import (
    decrypt_with_key_material,
    encrypt_with_key_material,
)


# -----------------------------------------------------------------------------
def build_serializer() -> tuple[AccessKeyEncryptionMaterialSerializer, sessionmaker]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    serializer = AccessKeyEncryptionMaterialSerializer(
        engine=engine,
        session_factory=factory,
    )
    return serializer, factory


# -----------------------------------------------------------------------------
def test_ensure_seeded_creates_initial_version_1_row() -> None:
    serializer, _ = build_serializer()

    row = serializer.ensure_seeded()

    assert row.key_purpose == "provider_access_keys"
    assert row.key_version == 1
    assert row.is_active is True
    assert row.seeded_at is not None
    assert row.activated_at is not None


# -----------------------------------------------------------------------------
def test_ensure_seeded_is_idempotent() -> None:
    serializer, _ = build_serializer()

    first = serializer.ensure_seeded()
    second = serializer.ensure_seeded()

    assert first.id == second.id
    assert first.key_version == second.key_version == 1


# -----------------------------------------------------------------------------
def test_only_one_active_row_exists_per_purpose() -> None:
    serializer, factory = build_serializer()
    serializer.ensure_seeded()
    serializer.rotate_material()

    with factory() as db_session:
        count_active = db_session.execute(
            select(func.count())
            .select_from(AccessKeyEncryptionMaterial)
            .where(
                AccessKeyEncryptionMaterial.key_purpose == "provider_access_keys",
                AccessKeyEncryptionMaterial.is_active.is_(True),
            )
        ).scalar_one()

    assert int(count_active) == 1


# -----------------------------------------------------------------------------
def test_rotate_material_creates_new_active_version() -> None:
    serializer, _ = build_serializer()
    seeded = serializer.ensure_seeded()

    rotated = serializer.rotate_material()

    assert seeded.key_version == 1
    assert rotated.key_version == 2
    assert rotated.is_active is True
    active = serializer.get_active_material()
    assert active.key_version == 2


# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
def test_encrypt_and_decrypt_use_db_seeded_material() -> None:
    serializer, _ = build_serializer()
    material = serializer.ensure_seeded()
    plaintext = "provider-secret-123"

    ciphertext = encrypt_with_key_material(plaintext, material.key_material)
    restored = decrypt_with_key_material(ciphertext, material.key_material)

    assert ciphertext != plaintext
    assert restored == plaintext

