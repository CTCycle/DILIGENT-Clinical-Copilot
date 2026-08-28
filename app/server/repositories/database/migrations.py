from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import event, inspect, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.schema import Column
from sqlalchemy.sql.sqltypes import Integer

from common.utils.logger import logger
from repositories.schemas import Base

BASELINE_REVISION = "202608200001"
HEAD_REVISION = "202608200002"
MIGRATION_LOCK_KEY = 7362381

###############################################################################
class MigrationError(RuntimeError):
    """Raised when a database cannot be safely adopted or migrated."""

###############################################################################
@dataclass(frozen=True, slots=True)
class MigrationResult:
    current_heads: tuple[str, ...]
    target_heads: tuple[str, ...]
    database_was_empty: bool
    adopted_revision: str | None
    reset: bool

    # -------------------------------------------------------------------------
    @property
    def upgraded(self) -> bool:
        return self.current_heads != self.target_heads or self.reset

###############################################################################
def resolve_migrations_path() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        frozen_path = Path(frozen_root) / "migrations"
        if frozen_path.is_dir():
            return frozen_path

    source_path = Path(__file__).resolve().parents[2] / "migrations"
    if source_path.is_dir():
        return source_path
    raise MigrationError(f"Alembic migration directory is missing: {source_path}")

###############################################################################
def build_alembic_config() -> Config:
    migrations_path = resolve_migrations_path()
    ini_path = migrations_path.parent / "alembic.ini"
    config = Config(str(ini_path) if ini_path.is_file() else None)
    config.set_main_option("script_location", str(migrations_path))
    config.attributes["connection"] = None
    return config

###############################################################################
def _script_directory(config: Config) -> ScriptDirectory:
    return ScriptDirectory.from_config(config)

###############################################################################
def _target_heads(config: Config) -> tuple[str, ...]:
    heads = tuple(sorted(_script_directory(config).get_heads()))
    if heads != (HEAD_REVISION,):
        raise MigrationError(
            "Expected one Alembic head at "
            f"{HEAD_REVISION}, found {heads or 'none'}."
        )
    return heads

###############################################################################
def _current_heads(connection: Connection) -> tuple[str, ...]:
    context = MigrationContext.configure(connection)
    return tuple(sorted(context.get_current_heads()))

###############################################################################
def _user_tables(connection: Connection) -> set[str]:
    return set(inspect(connection).get_table_names()) - {"alembic_version"}

###############################################################################
def _schema_differences(connection: Connection) -> list[tuple[object, ...]]:
    migration_context = MigrationContext.configure(
        connection,
        opts={
            "target_metadata": Base.metadata,
            "compare_type": True,
            "compare_server_default": True,
            "render_as_batch": True,
        },
    )
    return list(compare_metadata(migration_context, Base.metadata))

###############################################################################
def _is_legacy_v24_schema(differences: list[tuple[object, ...]]) -> bool:
    if len(differences) != 1:
        return False
    difference = differences[0]
    if len(difference) != 4 or difference[0] != "remove_column":
        return False
    _, schema, table_name, column = difference
    return (
        schema is None
        and table_name == "application_configuration"
        and isinstance(column, Column)
        and column.name == "schema_version"
        and isinstance(column.type, Integer)
        and not column.nullable
    )

###############################################################################
def _format_differences(differences: list[tuple[object, ...]]) -> str:
    if not differences:
        return "none"
    return "; ".join(str(difference) for difference in differences[:5])

###############################################################################
def _begin_sqlite_exclusive(connection: Connection) -> None:
    driver_connection: Any = connection.connection.driver_connection
    previous_autocommit = getattr(driver_connection, "autocommit", None)
    if previous_autocommit is not None:
        driver_connection.autocommit = True
    try:
        driver_connection.execute("BEGIN EXCLUSIVE")
    finally:
        if previous_autocommit is not None:
            driver_connection.autocommit = previous_autocommit

###############################################################################
@contextmanager
def _migration_transaction(engine: Engine) -> Iterator[Connection]:
    exclusive_listener = None
    if engine.dialect.name == "sqlite":
        exclusive_listener = _begin_sqlite_exclusive
        event.listen(engine, "begin", exclusive_listener)

    try:
        with engine.connect() as connection:
            with connection.begin():
                if engine.dialect.name == "postgresql":
                    connection.execute(
                        text("SELECT pg_advisory_xact_lock(:lock_key)"),
                        {"lock_key": MIGRATION_LOCK_KEY},
                    )
                    logger.info("Acquired PostgreSQL Alembic migration lock")
                elif engine.dialect.name == "sqlite":
                    logger.info("Acquired SQLite exclusive Alembic migration lock")
                yield connection
    finally:
        if exclusive_listener is not None:
            event.remove(engine, "begin", exclusive_listener)

###############################################################################
def _run_command(connection: Connection, action: Literal["upgrade", "stamp", "downgrade"], revision: str) -> None:
    config = build_alembic_config()
    config.attributes["connection"] = connection
    logger.info("Running Alembic %s %s", action, revision)
    try:
        if action == "upgrade":
            command.upgrade(config, revision)
        elif action == "stamp":
            command.stamp(config, revision)
        else:
            command.downgrade(config, revision)
    except Exception as exc:
        raise MigrationError(
            f"Alembic {action} {revision} failed: {exc}"
        ) from exc

###############################################################################
def _validate_known_revision(config: Config, current_heads: tuple[str, ...]) -> None:
    script = _script_directory(config)
    known_revisions = {revision.revision for revision in script.walk_revisions()}
    unknown = [head for head in current_heads if head not in known_revisions]
    if unknown:
        raise MigrationError(
            "Database references unknown Alembic revision(s): "
            f"{', '.join(unknown)}. Restore a compatible migration set or obtain a backup."
        )

###############################################################################
def _adopt_unversioned_database(
    connection: Connection,
    config: Config,
    *,
    database_was_empty: bool,
) -> str | None:
    if database_was_empty:
        return None

    differences = _schema_differences(connection)
    if not differences:
        _run_command(connection, "stamp", HEAD_REVISION)
        logger.info("Stamped unversioned database at current Alembic head")
        return HEAD_REVISION

    if _is_legacy_v24_schema(differences):
        _run_command(connection, "stamp", BASELINE_REVISION)
        logger.info(
            "Adopted unversioned v2.4-v3.0 database at Alembic revision %s",
            BASELINE_REVISION,
        )
        return BASELINE_REVISION

    raise MigrationError(
        "Unversioned database schema is not a recognized v2.4+ DILIGENT schema; "
        "no migration changes were applied. Differences: "
        f"{_format_differences(differences)}"
    )

###############################################################################
def migrate_database(
    engine: Engine,
    *,
    database_was_empty: bool,
    drop_existing: bool = False,
) -> MigrationResult:
    config = build_alembic_config()
    target_heads = _target_heads(config)

    try:
        with _migration_transaction(engine) as connection:
            initial_heads = _current_heads(connection)
            user_tables = _user_tables(connection)
            database_is_empty = not user_tables
            adopted_revision: str | None = None
            reset = False

            logger.info(
                "Alembic schema check: current=%s target=%s tables=%s",
                initial_heads or "base",
                target_heads,
                len(user_tables),
            )

            if not initial_heads:
                if database_is_empty:
                    logger.info("Applying Alembic migrations to an empty database")
                elif inspect(connection).has_table("alembic_version"):
                    raise MigrationError(
                        "Database has an empty Alembic version table but contains "
                        "application tables; refusing to guess its migration history."
                    )
                else:
                    adopted_revision = _adopt_unversioned_database(
                        connection,
                        config,
                        database_was_empty=database_is_empty,
                    )

            current_heads = _current_heads(connection)
            _validate_known_revision(config, current_heads)

            if drop_existing and (current_heads or user_tables):
                if not current_heads:
                    raise MigrationError(
                        "The database must be a recognized Alembic schema before "
                        "--drop-existing can reset it."
                    )
                _run_command(connection, "downgrade", "base")
                reset = True
                current_heads = _current_heads(connection)

            if current_heads != target_heads:
                _run_command(connection, "upgrade", "head")

            final_heads = _current_heads(connection)
            if final_heads != target_heads:
                raise MigrationError(
                    f"Alembic upgrade completed at {final_heads or 'base'}, "
                    f"expected {target_heads}."
                )

            return MigrationResult(
                current_heads=initial_heads,
                target_heads=target_heads,
                database_was_empty=database_was_empty,
                adopted_revision=adopted_revision,
                reset=reset,
            )
    except MigrationError:
        raise
    except SQLAlchemyError as exc:
        raise MigrationError(f"Database migration connection failed: {exc}") from exc
