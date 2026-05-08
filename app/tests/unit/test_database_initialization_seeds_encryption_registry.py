from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from domain.settings.configuration import DatabaseSettings
from repositories.database import initializer
from repositories.database.sqlite import SQLiteRepository
from repositories.schemas.models import AccessKeyEncryptionMaterial


# -----------------------------------------------------------------------------
def make_sqlite_settings() -> DatabaseSettings:
    return DatabaseSettings(
        embedded_database=True,
        engine="postgres",
        host="127.0.0.1",
        port=5432,
        database_name="diligent",
        username="postgres",
        password="",
        ssl=False,
        ssl_ca=None,
        connect_timeout=10,
        insert_batch_size=1000,
        insert_commit_interval=100,
        select_page_size=1000,
    )


# -----------------------------------------------------------------------------
def test_sqlite_fresh_creation_seeds_registry_once(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "repositories.database.sqlite.RESOURCES_PATH",
        str(tmp_path),
    )

    repository = SQLiteRepository(make_sqlite_settings())
    assert repository.db_path is not None
    assert Path(repository.db_path).exists()

    factory = sessionmaker(bind=repository.engine, future=True)
    with factory() as db_session:
        count_rows = db_session.execute(
            select(func.count()).select_from(AccessKeyEncryptionMaterial)
        ).scalar_one()
        active_rows = db_session.execute(
            select(func.count())
            .select_from(AccessKeyEncryptionMaterial)
            .where(AccessKeyEncryptionMaterial.is_active.is_(True))
        ).scalar_one()

    assert int(count_rows) == 1
    assert int(active_rows) == 1


# -----------------------------------------------------------------------------
def test_sqlite_reopen_with_existing_db_does_not_reseed(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "repositories.database.sqlite.RESOURCES_PATH",
        str(tmp_path),
    )

    first = SQLiteRepository(make_sqlite_settings())
    second = SQLiteRepository(make_sqlite_settings())
    factory = sessionmaker(bind=second.engine, future=True)

    with factory() as db_session:
        count_rows = db_session.execute(
            select(func.count()).select_from(AccessKeyEncryptionMaterial)
        ).scalar_one()

    assert int(count_rows) == 1
    assert first.db_path == second.db_path


# -----------------------------------------------------------------------------
def test_postgresql_initialization_path_seeds_after_schema_creation(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    settings = DatabaseSettings(
        embedded_database=False,
        engine="postgres",
        host="127.0.0.1",
        port=5432,
        database_name="diligent",
        username="postgres",
        password="",
        ssl=False,
        ssl_ca=None,
        connect_timeout=10,
        insert_batch_size=1000,
        insert_commit_interval=100,
        select_page_size=1000,
    )

    order: list[str] = []
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, _stmt):
            class ScalarResult:
                @staticmethod
                def scalar():
                    return 1

            return ScalarResult()

    class FakeAdminEngine:
        @staticmethod
        def connect():
            return FakeConnection()

    class FakePostgresRepository:
        def __init__(self, _settings) -> None:
            self.engine = engine

    def fake_create_all(_engine):
        order.append("create_all")

    class FakeMaterialSerializer:
        def __init__(self, **_kwargs) -> None:
            pass

        def ensure_seeded(self, purpose: str):
            assert purpose == "provider_access_keys"
            order.append("seeded")

    class FakeTextNormalizationSerializer:
        def __init__(self, **_kwargs) -> None:
            pass

        def ensure_seeded(self):
            order.append("text_seeded")

    monkeypatch.setattr(
        initializer.sqlalchemy, "create_engine", lambda *a, **k: FakeAdminEngine()
    )
    monkeypatch.setattr(initializer, "PostgresRepository", FakePostgresRepository)
    monkeypatch.setattr(initializer.Base.metadata, "create_all", fake_create_all)
    monkeypatch.setattr(
        initializer, "AccessKeyEncryptionMaterialSerializer", FakeMaterialSerializer
    )
    monkeypatch.setattr(
        initializer,
        "TextNormalizationVocabularySerializer",
        FakeTextNormalizationSerializer,
    )

    db_name = initializer.ensure_postgres_database(settings)

    assert db_name == "diligent"
    assert order == ["create_all", "seeded", "text_seeded"]


# -----------------------------------------------------------------------------
def test_seeding_does_not_create_duplicate_active_rows(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "repositories.database.sqlite.RESOURCES_PATH",
        str(tmp_path),
    )

    repository = SQLiteRepository(make_sqlite_settings())
    factory = sessionmaker(bind=repository.engine, future=True)
    with factory() as db_session:
        count_rows = db_session.execute(
            select(func.count())
            .select_from(AccessKeyEncryptionMaterial)
            .where(AccessKeyEncryptionMaterial.is_active.is_(True))
        ).scalar_one()

    assert int(count_rows) == 1


