from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier

import pytest
from alembic import command
from sqlalchemy import create_engine, inspect, text

import repositories.database.migrations as migration_coordinator
from repositories.database.migrations import (
    HEAD_REVISION,
    MigrationError,
    build_alembic_config,
    migrate_database,
)

###############################################################################
def _engine(path: Path):
    return create_engine(
        f"sqlite+pysqlite:///{path}",
        future=True,
        connect_args={"timeout": 30.0, "autocommit": False},
    )

###############################################################################
def _upgrade_to(engine, revision: str) -> None:  # type: ignore[no-untyped-def]
    config = build_alembic_config()
    with engine.connect() as connection:
        with connection.begin():
            config.attributes["connection"] = connection
            command.upgrade(config, revision)

###############################################################################
def test_fresh_sqlite_database_reaches_head_and_is_idempotent(tmp_path: Path) -> None:
    database_path = tmp_path / "fresh.db"
    engine = _engine(database_path)
    try:
        first = migrate_database(engine, database_was_empty=True)
        second = migrate_database(engine, database_was_empty=False)

        assert first.target_heads == (HEAD_REVISION,)
        assert second.current_heads == (HEAD_REVISION,)
        assert second.upgraded is False
        assert inspect(engine).has_table("clinical_sessions")
        assert inspect(engine).has_table("alembic_version")
        with engine.connect() as connection:
            assert connection.execute(text("select version_num from alembic_version")).scalar_one() == HEAD_REVISION
    finally:
        engine.dispose()

###############################################################################
def test_populated_unversioned_schema_is_rejected_without_stamping(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.db"
    engine = _engine(database_path)
    try:
        _upgrade_to(engine, "202608200002")
        with engine.begin() as connection:
            connection.execute(
                text(
                    "insert into application_configuration "
                    "(id, revision, payload) values (1, 0, :payload)"
                ),
                {"payload": "{\"clinical_model\": \"legacy\"}"},
            )
            connection.execute(text("drop table alembic_version"))

        with pytest.raises(MigrationError, match="no Alembic revision"):
            migrate_database(engine, database_was_empty=False)
        with engine.connect() as connection:
            assert not inspect(connection).has_table("alembic_version")
            assert connection.execute(
                text("select payload from application_configuration where id = 1")
            ).scalar_one() == '{"clinical_model": "legacy"}'
            assert "schema_version" not in {
                column["name"]
                for column in inspect(connection).get_columns("application_configuration")
            }
    finally:
        engine.dispose()

###############################################################################
def test_unversioned_current_schema_is_rejected_without_recreating_tables(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "current-unversioned.db"
    engine = _engine(database_path)
    try:
        migrate_database(engine, database_was_empty=True)
        with engine.begin() as connection:
            connection.execute(
                text(
                    "insert into application_configuration "
                    "(id, revision, payload) values (1, 0, :payload)"
                ),
                {"payload": "{\"preserve\": true}"},
            )
            connection.execute(text("drop table alembic_version"))

        with pytest.raises(MigrationError, match="no Alembic revision"):
            migrate_database(engine, database_was_empty=False)
        with engine.connect() as connection:
            assert not inspect(connection).has_table("alembic_version")
            assert connection.execute(
                text("select payload from application_configuration where id = 1")
            ).scalar_one() == '{"preserve": true}'
    finally:
        engine.dispose()

###############################################################################
def test_unknown_unversioned_schema_is_rejected_before_version_table_creation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "unknown.db"
    engine = _engine(database_path)
    try:
        with engine.begin() as connection:
            connection.execute(text("create table unrelated (id integer primary key)"))

        with pytest.raises(MigrationError, match="no Alembic revision"):
            migrate_database(engine, database_was_empty=False)

        with engine.connect() as connection:
            assert not inspect(connection).has_table("alembic_version")
    finally:
        engine.dispose()

###############################################################################
def test_model_role_data_migration_populates_only_missing_roles(tmp_path: Path) -> None:
    database_path = tmp_path / "model-role-migration.db"
    engine = _engine(database_path)
    try:
        _upgrade_to(engine, "202608200002")
        with engine.begin() as connection:
            connection.execute(
                text(
                    "insert into application_configuration "
                    "(id, revision, payload) values (1, 0, :payload)"
                ),
                {
                    "payload": json.dumps(
                        {
                            "clinical_model": "clinical-model",
                            "text_extraction_model": "parser-model",
                            "revision_model": "saved-revision-model",
                        }
                    )
                },
            )

        migrate_database(engine, database_was_empty=False)

        with engine.connect() as connection:
            payload = json.loads(
                connection.execute(
                    text("select payload from application_configuration where id = 1")
                ).scalar_one()
            )
        assert payload == {
            "clinical_model": "clinical-model",
            "text_extraction_model": "parser-model",
            "revision_model": "saved-revision-model",
            "timeline_model": "parser-model",
        }
    finally:
        engine.dispose()

###############################################################################
def test_unknown_versioned_schema_is_rejected_without_changing_revision(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "unknown-version.db"
    engine = _engine(database_path)
    try:
        migrate_database(engine, database_was_empty=True)
        with engine.begin() as connection:
            connection.execute(
                text("update alembic_version set version_num = 'unknown_revision'")
            )

        with pytest.raises(MigrationError, match="unknown Alembic revision"):
            migrate_database(engine, database_was_empty=False)

        with engine.connect() as connection:
            assert connection.execute(
                text("select version_num from alembic_version")
            ).scalar_one() == "unknown_revision"
    finally:
        engine.dispose()

###############################################################################
def test_failed_migration_rolls_back_schema_and_version_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "failed.db"
    engine = _engine(database_path)

    def fail_after_ddl(config, _revision: str) -> None:  # type: ignore[no-untyped-def]
        connection = config.attributes["connection"]
        connection.execute(text("create table migration_failure_probe (id integer)"))
        raise RuntimeError("injected migration failure")

    monkeypatch.setattr(migration_coordinator.command, "upgrade", fail_after_ddl)
    try:
        with pytest.raises(MigrationError, match="injected migration failure"):
            migrate_database(engine, database_was_empty=True)

        with engine.connect() as connection:
            inspector = inspect(connection)
            assert not inspector.has_table("migration_failure_probe")
            assert not inspector.has_table("alembic_version")
    finally:
        engine.dispose()

###############################################################################
def test_drop_existing_resets_managed_database_through_alembic(tmp_path: Path) -> None:
    database_path = tmp_path / "reset.db"
    engine = _engine(database_path)
    try:
        migrate_database(engine, database_was_empty=True)
        with engine.begin() as connection:
            connection.execute(
                text(
                    "insert into application_configuration "
                    "(id, revision, payload) values (1, 0, :payload)"
                ),
                {"payload": "{\"reset\": true}"},
            )

        result = migrate_database(engine, database_was_empty=False, drop_existing=True)

        assert result.reset is True
        with engine.connect() as connection:
            assert connection.execute(
                text("select count(*) from application_configuration")
            ).scalar_one() == 0
            assert connection.execute(text("select version_num from alembic_version")).scalar_one() == HEAD_REVISION
    finally:
        engine.dispose()

###############################################################################
def test_concurrent_sqlite_startup_attempts_serialize(tmp_path: Path) -> None:
    database_path = tmp_path / "concurrent.db"
    barrier = Barrier(2)

    def run_startup() -> tuple[str, ...]:
        engine = _engine(database_path)
        try:
            barrier.wait(timeout=30)
            return migrate_database(engine, database_was_empty=not database_path.exists()).target_heads
        finally:
            engine.dispose()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _item: run_startup(), range(2)))

    assert results == [(HEAD_REVISION,), (HEAD_REVISION,)]
