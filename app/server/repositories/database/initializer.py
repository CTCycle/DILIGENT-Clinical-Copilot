from __future__ import annotations

import urllib.parse

import sqlalchemy
from sqlalchemy import column, literal, select, table
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.elements import TextClause

from common.catalogs.manifest_loader import (
    compute_manifest_hash,
    iter_catalog_manifest_paths,
    load_catalog_manifest,
)
from common.catalogs.provider import get_catalog_provider
from common.utils.logger import logger
from configurations.startup import get_server_settings
from domain.catalogs import CatalogSeedResult
from domain.settings.configuration import DatabaseSettings
from repositories.database.migrations import MigrationError, migrate_database
from repositories.database.postgres import PostgresRepository
from repositories.database.sqlite import (
    SQLiteRepository,
    resolve_sqlite_database_path,
)
from repositories.database.utils import (
    normalize_postgres_engine,
    validate_postgres_database_name,
)
from repositories.serialization.catalogs import ReferenceCatalogSerializer

POSTGRES_CREATE_DATABASE_LOCK_KEY = 7362382


def build_postgres_connect_args(settings: DatabaseSettings) -> dict[str, str | int]:
    connect_args: dict[str, str | int] = {
        "connect_timeout": settings.connect_timeout,
        "client_encoding": "utf8",
    }
    if settings.ssl:
        connect_args["sslmode"] = "require"
        if settings.ssl_ca:
            connect_args["sslrootcert"] = settings.ssl_ca
    return connect_args


def build_postgres_url(settings: DatabaseSettings, database_name: str) -> str:
    port = settings.port or 5432
    engine_name = normalize_postgres_engine(settings.engine)
    safe_username = urllib.parse.quote_plus(settings.username or "")
    safe_password = urllib.parse.quote_plus(settings.password or "")
    safe_database_name = validate_postgres_database_name(database_name)
    return (
        f"{engine_name}://{safe_username}:{safe_password}"
        f"@{settings.host}:{port}/{safe_database_name}"
    )


def clone_settings_with_database(
    settings: DatabaseSettings, database_name: str
) -> DatabaseSettings:
    safe_database_name = validate_postgres_database_name(database_name)
    return DatabaseSettings(
        embedded_database=False,
        engine=settings.engine,
        host=settings.host,
        port=settings.port,
        database_name=safe_database_name,
        username=settings.username,
        password=settings.password,
        ssl=settings.ssl,
        ssl_ca=settings.ssl_ca,
        connect_timeout=settings.connect_timeout,
        insert_batch_size=settings.insert_batch_size,
        insert_commit_interval=settings.insert_commit_interval,
        select_page_size=settings.select_page_size,
    )


def build_postgres_create_database_sql(database_name: str) -> TextClause:
    safe_database_name = validate_postgres_database_name(database_name)
    return sqlalchemy.text(
        f'CREATE DATABASE "{safe_database_name}" WITH ENCODING \'UTF8\' TEMPLATE template0'
    )


def _seed_catalogs(
    serializer: ReferenceCatalogSerializer,
    force: bool = False,
) -> CatalogSeedResult:
    paths = iter_catalog_manifest_paths()
    manifests_seeded = 0
    entries_written = 0
    for path in paths:
        manifest_hash = compute_manifest_hash(path)
        manifest = load_catalog_manifest(path)
        if (not force) and serializer.has_successful_seed(
            manifest.manifest, manifest_hash
        ):
            continue
        written = serializer.replace_manifest_entries(
            manifest=manifest,
            manifest_hash=manifest_hash,
            source_path=str(path),
        )
        if written > 0:
            manifests_seeded += 1
            entries_written += written
    return CatalogSeedResult(
        manifests_seen=len(paths),
        manifests_seeded=manifests_seeded,
        entries_written=entries_written,
    )


def _seed_repository_catalogs(
    repository: SQLiteRepository | PostgresRepository,
    *,
    backend_label: str,
    force: bool,
) -> None:
    serializer = ReferenceCatalogSerializer(session_factory=repository.session_factory)
    result = _seed_catalogs(serializer, force=force)
    logger.info(
        "Catalog seeding completed for %s: seen=%s seeded=%s entries=%s",
        backend_label,
        result.manifests_seen,
        result.manifests_seeded,
        result.entries_written,
    )
    get_catalog_provider().invalidate()


def initialize_sqlite_database(
    settings: DatabaseSettings,
    *,
    drop_existing: bool = False,
    seed_catalogs: bool = True,
    force_reseed_catalogs: bool = False,
) -> None:
    database_path = resolve_sqlite_database_path(settings)
    database_was_missing = not database_path.is_file()
    repository = SQLiteRepository(settings)
    try:
        migrate_database(
            repository.engine,
            database_was_empty=database_was_missing,
            drop_existing=drop_existing,
        )
        if seed_catalogs:
            _seed_repository_catalogs(
                repository,
                backend_label="SQLite",
                force=force_reseed_catalogs,
            )
        logger.info("SQLite database is synchronized at %s", repository.db_path)
    finally:
        repository.engine.dispose()


def _is_missing_postgres_database(error: SQLAlchemyError) -> bool:
    original = getattr(error, "orig", error)
    state = getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)
    if state == "3D000":
        return True
    message = str(original).lower()
    return "database" in message and "does not exist" in message


def _create_postgres_database_if_missing(settings: DatabaseSettings) -> tuple[str, bool]:
    if not settings.host:
        raise ValueError("Database host is required for PostgreSQL initialization.")
    if not settings.username:
        raise ValueError("Database username is required for PostgreSQL initialization.")
    if not settings.database_name:
        raise ValueError("Database name is required for PostgreSQL initialization.")

    target_database = validate_postgres_database_name(settings.database_name)
    normalized_settings = clone_settings_with_database(settings, target_database)
    target_repository = PostgresRepository(normalized_settings)
    try:
        try:
            with target_repository.engine.connect():
                logger.info("PostgreSQL database %s is reachable", target_database)
                return target_database, False
        except SQLAlchemyError as exc:
            if not _is_missing_postgres_database(exc):
                raise
            logger.info(
                "PostgreSQL database %s is not available; attempting creation",
                target_database,
            )
    finally:
        target_repository.engine.dispose()

    connect_args = build_postgres_connect_args(settings)
    admin_engine = sqlalchemy.create_engine(
        build_postgres_url(settings, "postgres"),
        echo=False,
        future=True,
        connect_args=connect_args,
        isolation_level="AUTOCOMMIT",
        pool_pre_ping=True,
    )
    pg_database = table("pg_database", column("datname"))
    exists_stmt = (
        select(literal(1))
        .select_from(pg_database)
        .where(pg_database.c.datname == target_database)
        .limit(1)
    )
    try:
        with admin_engine.connect() as connection:
            connection.execute(
                sqlalchemy.text("SELECT pg_advisory_lock(:lock_key)"),
                {"lock_key": POSTGRES_CREATE_DATABASE_LOCK_KEY},
            )
            try:
                exists = connection.execute(exists_stmt).scalar()
                if exists:
                    logger.info(
                        "PostgreSQL database %s was created by another initializer",
                        target_database,
                    )
                    return target_database, False
                connection.execute(build_postgres_create_database_sql(target_database))
                logger.info("Created PostgreSQL database %s", target_database)
                return target_database, True
            finally:
                connection.execute(
                    sqlalchemy.text("SELECT pg_advisory_unlock(:lock_key)"),
                    {"lock_key": POSTGRES_CREATE_DATABASE_LOCK_KEY},
                )
    finally:
        admin_engine.dispose()


def ensure_postgres_database(
    settings: DatabaseSettings,
    *,
    drop_existing: bool = False,
    seed_catalogs: bool = True,
    force_reseed_catalogs: bool = False,
) -> str:
    target_database, _ = _create_postgres_database_if_missing(settings)
    normalized_settings = clone_settings_with_database(settings, target_database)
    repository = PostgresRepository(normalized_settings)
    try:
        migrate_database(
            repository.engine,
            database_was_empty=False,
            drop_existing=drop_existing,
        )
        if seed_catalogs:
            _seed_repository_catalogs(
                repository,
                backend_label="PostgreSQL",
                force=force_reseed_catalogs,
            )
        logger.info("PostgreSQL database %s is synchronized", target_database)
    finally:
        repository.engine.dispose()
    return target_database


def ensure_database_ready(settings: DatabaseSettings) -> bool:
    """Synchronize startup schema and seed only a newly created database."""

    if settings.backend == "sqlite":
        database_path = resolve_sqlite_database_path(settings)
        database_was_missing = not database_path.is_file()
        initialize_sqlite_database(
            settings,
            seed_catalogs=database_was_missing,
        )
        return database_was_missing

    if settings.backend != "postgresql":
        raise ValueError(f"Unsupported database backend: {settings.backend}")

    target_database, database_was_created = _create_postgres_database_if_missing(settings)
    normalized_settings = clone_settings_with_database(settings, target_database)
    repository = PostgresRepository(normalized_settings)
    try:
        migrate_database(repository.engine, database_was_empty=False)
        if database_was_created:
            _seed_repository_catalogs(
                repository,
                backend_label="PostgreSQL",
                force=False,
            )
        logger.info("PostgreSQL database %s is synchronized", target_database)
    finally:
        repository.engine.dispose()
    return database_was_created


def run_database_initialization(
    *,
    drop_existing: bool = False,
    seed_catalogs: bool = True,
    force_reseed_catalogs: bool = False,
) -> None:
    settings = get_server_settings().database
    if settings.backend == "sqlite":
        logger.info("Running SQLite Alembic initialization path")
        initialize_sqlite_database(
            settings,
            drop_existing=drop_existing,
            seed_catalogs=seed_catalogs,
            force_reseed_catalogs=force_reseed_catalogs,
        )
        return

    logger.info("Running PostgreSQL Alembic initialization path")
    engine_name = normalize_postgres_engine(settings.engine).lower()
    if engine_name not in {"postgres", "postgresql", "postgresql+psycopg"}:
        raise ValueError(f"Unsupported database engine: {settings.engine}")
    ensure_postgres_database(
        settings,
        drop_existing=drop_existing,
        seed_catalogs=seed_catalogs,
        force_reseed_catalogs=force_reseed_catalogs,
    )


def initialize_database(
    drop_existing: bool = False,
    seed_catalogs: bool = True,
    force_reseed_catalogs: bool = False,
) -> None:
    try:
        run_database_initialization(
            drop_existing=drop_existing,
            seed_catalogs=seed_catalogs,
            force_reseed_catalogs=force_reseed_catalogs,
        )
    except (MigrationError, SQLAlchemyError, ValueError) as exc:
        logger.error("Database initialization failed: %s", exc)
        raise SystemExit(1) from exc
    except Exception as exc:
        logger.exception("Unexpected error during database initialization.")
        raise SystemExit(1) from exc
