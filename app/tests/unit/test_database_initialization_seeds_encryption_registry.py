from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from domain.settings.configuration import DatabaseSettings
from repositories.database import initializer
from repositories.database.sqlite import SQLiteRepository
from repositories.serialization.access_key_encryption import (
    AccessKeyEncryptionMaterialSerializer,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

###############################################################################
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

###############################################################################
def _make_temp_db_root(prefix: str) -> Path:
    temp_root = Path(tempfile.gettempdir()) / f"{prefix}-{uuid.uuid4().hex}"
    temp_root.mkdir(parents=True, exist_ok=True)
    return temp_root

###############################################################################
def test_sqlite_fresh_creation_uses_external_material(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    temp_root = _make_temp_db_root("sqlite-seed-fresh")
    try:
        monkeypatch.setattr(
            "repositories.database.sqlite.DATABASE_FILE_PATH",
            temp_root / "database.db",
        )
        material_path = temp_root / "access-key-material.json"
        monkeypatch.setenv("DILIGENT_ACCESS_KEY_MATERIAL_FILE", str(material_path))

        repository = SQLiteRepository(make_sqlite_settings())
        assert repository.db_path is not None
        assert Path(repository.db_path).exists()

        assert not material_path.exists()
        material = AccessKeyEncryptionMaterialSerializer(
            engine=repository.engine,
            session_factory=repository.session_factory,
        ).ensure_seeded()
        assert material.key_version == 1
        assert material_path.exists()
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

###############################################################################
def test_sqlite_reopen_with_existing_db_reuses_external_material(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    temp_root = _make_temp_db_root("sqlite-seed-reopen")
    try:
        monkeypatch.setattr(
            "repositories.database.sqlite.DATABASE_FILE_PATH",
            temp_root / "database.db",
        )
        material_path = temp_root / "access-key-material.json"
        monkeypatch.setenv("DILIGENT_ACCESS_KEY_MATERIAL_FILE", str(material_path))

        first = SQLiteRepository(make_sqlite_settings())
        second = SQLiteRepository(make_sqlite_settings())
        first_material = AccessKeyEncryptionMaterialSerializer(
            engine=first.engine,
            session_factory=first.session_factory,
        ).ensure_seeded()
        second_material = AccessKeyEncryptionMaterialSerializer(
            engine=second.engine,
            session_factory=second.session_factory,
        ).ensure_seeded()
        assert first_material.key_material == second_material.key_material
        assert first.db_path == second.db_path
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

###############################################################################
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

    ###############################################################################
    class FakeConnection:

        # -------------------------------------------------------------------------
        def __enter__(self):
            return self

        # -------------------------------------------------------------------------
        def __exit__(self, exc_type, exc, tb):
            return False

        # -------------------------------------------------------------------------
        def execute(self, _stmt):

            ###############################################################################
            class ScalarResult:

                # -------------------------------------------------------------------------
                @staticmethod
                def scalar():
                    return 1

            return ScalarResult()

    ###############################################################################
    class FakeAdminEngine:

        # -------------------------------------------------------------------------
        @staticmethod
        def connect():
            return FakeConnection()

    ###############################################################################
    class FakePostgresRepository:

        # -------------------------------------------------------------------------
        def __init__(self, _settings) -> None:
            self.engine = engine
            self.session_factory = sessionmaker(bind=engine, future=True)

    def fake_create_all(_engine):
        order.append("create_all")

    ###############################################################################
    class FakeCatalogSerializer:

        # -------------------------------------------------------------------------
        def __init__(self, **_kwargs) -> None:
            pass

    ###############################################################################
    class FakeCatalogSeedResult:
        manifests_seen = 1
        manifests_seeded = 1
        entries_written = 1

    def fake_seed_catalogs(_serializer, *, force: bool = False):
        assert force is False
        order.append("catalog_seeded")
        return FakeCatalogSeedResult()

    monkeypatch.setattr(
        initializer.sqlalchemy, "create_engine", lambda *a, **k: FakeAdminEngine()
    )
    monkeypatch.setattr(initializer, "PostgresRepository", FakePostgresRepository)
    monkeypatch.setattr(initializer.Base.metadata, "create_all", fake_create_all)
    monkeypatch.setattr(
        initializer, "ReferenceCatalogSerializer", FakeCatalogSerializer
    )
    monkeypatch.setattr(initializer, "_seed_catalogs", fake_seed_catalogs)

    db_name = initializer.ensure_postgres_database(settings)

    assert db_name == "diligent"
    assert order == ["create_all", "catalog_seeded"]

###############################################################################
def test_external_material_does_not_duplicate_on_reopen(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    temp_root = _make_temp_db_root("sqlite-seed-active")
    try:
        monkeypatch.setattr(
            "repositories.database.sqlite.DATABASE_FILE_PATH",
            temp_root / "database.db",
        )
        material_path = temp_root / "access-key-material.json"
        monkeypatch.setenv("DILIGENT_ACCESS_KEY_MATERIAL_FILE", str(material_path))

        repository = SQLiteRepository(make_sqlite_settings())
        material = AccessKeyEncryptionMaterialSerializer(
            engine=repository.engine,
            session_factory=repository.session_factory,
        ).ensure_seeded()
        repository_again = SQLiteRepository(make_sqlite_settings())
        material_again = AccessKeyEncryptionMaterialSerializer(
            engine=repository_again.engine,
            session_factory=repository_again.session_factory,
        ).ensure_seeded()
        assert material.key_material == material_again.key_material
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

###############################################################################
def _sqlite_settings() -> DatabaseSettings:
    return DatabaseSettings(
        backend="sqlite",
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

###############################################################################
def _postgres_settings() -> DatabaseSettings:
    return DatabaseSettings(
        backend="postgresql",
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

###############################################################################
def test_run_database_initialization_uses_sqlite_path_when_embedded(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    calls: list[str] = []
    settings = _sqlite_settings()

    monkeypatch.setattr(
        initializer,
        "get_server_settings",
        lambda: type("S", (), {"database": settings})(),
    )
    monkeypatch.setattr(
        initializer,
        "initialize_sqlite_database",
        lambda _settings: calls.append("sqlite"),
    )
    monkeypatch.setattr(
        initializer,
        "ensure_postgres_database",
        lambda _settings: calls.append("postgres"),
    )

    initializer.run_database_initialization()

    assert calls == ["sqlite"]

###############################################################################
def test_run_database_initialization_uses_postgres_path_when_external(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    calls: list[str] = []
    settings = _postgres_settings()

    monkeypatch.setattr(
        initializer,
        "get_server_settings",
        lambda: type("S", (), {"database": settings})(),
    )
    monkeypatch.setattr(
        initializer,
        "initialize_sqlite_database",
        lambda _settings: calls.append("sqlite"),
    )
    monkeypatch.setattr(
        initializer,
        "ensure_postgres_database",
        lambda _settings: calls.append("postgres"),
    )

    initializer.run_database_initialization()

    assert calls == ["postgres"]
