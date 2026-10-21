# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

import pytest
from domain.settings.configuration import DatabaseSettings
from repositories.database import initializer
from repositories.database.sqlite import SQLiteRepository
from repositories.serialization.access_key_encryption import (
    AccessKeyEncryptionMaterialSerializer,
)
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
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

        settings = make_sqlite_settings()
        initializer.initialize_sqlite_database(settings, seed_catalogs=False)
        repository = SQLiteRepository(settings)
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

        settings = make_sqlite_settings()
        initializer.initialize_sqlite_database(settings, seed_catalogs=False)
        first = SQLiteRepository(settings)
        second = SQLiteRepository(settings)
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

    def fake_migrate_database(_engine, **_kwargs):
        order.append("migration")

    def fake_seed_model_configuration(_repository):
        order.append("model_config_seeded")

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

    monkeypatch.setattr(initializer, "PostgresRepository", FakePostgresRepository)
    monkeypatch.setattr(initializer, "migrate_database", fake_migrate_database)
    monkeypatch.setattr(
        initializer, "_seed_model_configuration", fake_seed_model_configuration
    )
    monkeypatch.setattr(
        initializer, "ReferenceCatalogSerializer", FakeCatalogSerializer
    )
    monkeypatch.setattr(initializer, "_seed_catalogs", fake_seed_catalogs)

    db_name = initializer.ensure_postgres_database(settings)

    assert db_name == "diligent"
    assert order == ["migration", "model_config_seeded", "catalog_seeded"]

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

        settings = make_sqlite_settings()
        initializer.initialize_sqlite_database(settings, seed_catalogs=False)
        repository = SQLiteRepository(settings)
        material = AccessKeyEncryptionMaterialSerializer(
            engine=repository.engine,
            session_factory=repository.session_factory,
        ).ensure_seeded()
        repository_again = SQLiteRepository(settings)
        material_again = AccessKeyEncryptionMaterialSerializer(
            engine=repository_again.engine,
            session_factory=repository_again.session_factory,
        ).ensure_seeded()
        assert material.key_material == material_again.key_material
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

###############################################################################
def test_sqlite_startup_initializes_and_seeds_when_database_file_is_missing(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    database_path = tmp_path / "missing.db"
    settings = make_sqlite_settings()
    monkeypatch.setattr(
        "repositories.database.sqlite.DATABASE_FILE_PATH", database_path
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        initializer,
        "initialize_sqlite_database",
        lambda _settings, **kwargs: calls.append(kwargs) or True,
    )

    assert initializer.ensure_database_ready(settings) is True
    assert calls == [{"seed_catalogs": True}]

###############################################################################
def test_sqlite_startup_initializes_existing_empty_database(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    database_path = tmp_path / "existing.db"
    database_path.touch()
    settings = make_sqlite_settings()
    monkeypatch.setattr(
        "repositories.database.sqlite.DATABASE_FILE_PATH", database_path
    )

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        initializer,
        "initialize_sqlite_database",
        lambda _settings, **kwargs: calls.append(kwargs) or True,
    )

    assert initializer.ensure_database_ready(settings) is True
    assert calls == [{"seed_catalogs": True}]

###############################################################################
def test_sqlite_initialization_seeds_an_existing_file_without_app_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "empty-existing.db"
    database_path.touch()
    monkeypatch.setattr(
        "repositories.database.sqlite.DATABASE_FILE_PATH", database_path
    )
    migration_calls: list[dict[str, object]] = []
    catalog_calls: list[str] = []
    monkeypatch.setattr(
        initializer,
        "migrate_database",
        lambda _engine, **kwargs: migration_calls.append(kwargs),
    )
    monkeypatch.setattr(initializer, "_seed_model_configuration", lambda _repo: None)
    monkeypatch.setattr(
        initializer,
        "_seed_repository_catalogs",
        lambda _repo, **_kwargs: catalog_calls.append("seeded"),
    )

    assert initializer.initialize_sqlite_database(make_sqlite_settings()) is True
    assert migration_calls == [{"database_was_empty": True, "drop_existing": False}]
    assert catalog_calls == ["seeded"]

###############################################################################
def test_postgres_startup_seeds_catalogs_for_existing_empty_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_engine("sqlite:///:memory:")
    events: list[tuple[str, object]] = []

    ###############################################################################
    class FakePostgresRepository:

        # -------------------------------------------------------------------------
        def __init__(self, _settings) -> None:
            self.engine = engine

    monkeypatch.setattr(
        initializer,
        "_create_postgres_database_if_missing",
        lambda _settings: ("diligent", False),
    )
    monkeypatch.setattr(initializer, "PostgresRepository", FakePostgresRepository)
    monkeypatch.setattr(
        initializer,
        "migrate_database",
        lambda _engine, **kwargs: events.append(("migration", kwargs)),
    )
    monkeypatch.setattr(
        initializer,
        "_seed_model_configuration",
        lambda _repository: events.append(("model", None)),
    )
    monkeypatch.setattr(
        initializer,
        "_seed_repository_catalogs",
        lambda _repository, **kwargs: events.append(("catalogs", kwargs)),
    )

    assert initializer.ensure_database_ready(_postgres_settings()) is True
    assert events == [
        ("migration", {"database_was_empty": True}),
        ("model", None),
        ("catalogs", {"backend_label": "PostgreSQL", "force": False}),
    ]

###############################################################################
def test_database_initialization_converts_unavailable_connection_to_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_initialization(**_kwargs):
        raise SQLAlchemyError("connection refused")

    monkeypatch.setattr(initializer, "run_database_initialization", fail_initialization)

    with pytest.raises(SystemExit) as exc_info:
        initializer.initialize_database()

    assert exc_info.value.code == 1

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
def test_postgres_database_creation_uses_server_catalog_for_localized_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    localized_error = SQLAlchemyError(
        'connection failed: FATALE: il database "diligent" non esiste'
    )

    ###############################################################################
    class FakeTargetEngine:

        # -------------------------------------------------------------------------
        def connect(self):
            events.append("target_connect")
            raise localized_error

        # -------------------------------------------------------------------------
        def dispose(self) -> None:
            events.append("target_dispose")

    ###############################################################################
    class FakePostgresRepository:

        # -------------------------------------------------------------------------
        def __init__(self, _settings) -> None:
            self.engine = FakeTargetEngine()

    ###############################################################################
    class FakeResult:

        # -------------------------------------------------------------------------
        def scalar(self):
            return None

    ###############################################################################
    class FakeAdminConnection:

        # -------------------------------------------------------------------------
        def __enter__(self):
            return self

        # -------------------------------------------------------------------------
        def __exit__(self, *_args) -> None:
            return None

        # -------------------------------------------------------------------------
        def execute(self, statement, _parameters=None):
            sql = str(statement)
            if "pg_advisory_lock" in sql:
                events.append("lock")
            elif "FROM pg_database" in sql:
                events.append("catalog_check")
                return FakeResult()
            elif sql.startswith("CREATE DATABASE"):
                events.append("create")
            elif "pg_advisory_unlock" in sql:
                events.append("unlock")
            return None

    ###############################################################################
    class FakeAdminEngine:

        # -------------------------------------------------------------------------
        def connect(self):
            events.append("admin_connect")
            return FakeAdminConnection()

        # -------------------------------------------------------------------------
        def dispose(self) -> None:
            events.append("admin_dispose")

    admin_engine = FakeAdminEngine()
    monkeypatch.setattr(initializer, "PostgresRepository", FakePostgresRepository)
    monkeypatch.setattr(
        initializer.sqlalchemy,
        "create_engine",
        lambda *_args, **_kwargs: admin_engine,
    )

    assert initializer._create_postgres_database_if_missing(_postgres_settings()) == (
        "diligent",
        True,
    )
    assert events == [
        "target_connect",
        "target_dispose",
        "admin_connect",
        "lock",
        "catalog_check",
        "create",
        "unlock",
        "admin_dispose",
    ]

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
        lambda _settings, **_kwargs: calls.append("sqlite"),
    )
    monkeypatch.setattr(
        initializer,
        "ensure_postgres_database",
        lambda _settings, **_kwargs: calls.append("postgres"),
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
        lambda _settings, **_kwargs: calls.append("sqlite"),
    )
    monkeypatch.setattr(
        initializer,
        "ensure_postgres_database",
        lambda _settings, **_kwargs: calls.append("postgres"),
    )

    initializer.run_database_initialization()

    assert calls == ["postgres"]
