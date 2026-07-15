from __future__ import annotations

import urllib.parse
import sqlalchemy
from sqlalchemy import column, literal, select, table
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.elements import TextClause

from common.catalogs.manifest_loader import (
    compute_manifest_hash,
    iter_catalog_manifest_paths,
    load_catalog_manifest,
)
from common.catalogs.provider import catalog_provider
from common.utils.logger import logger
from configurations.startup import get_server_settings
from domain.catalogs import CatalogSeedResult
from domain.settings.configuration import DatabaseSettings
from repositories.database.postgres import PostgresRepository
from repositories.database.sqlite import SQLiteRepository
from repositories.database.utils import (
    normalize_postgres_engine,
    validate_postgres_database_name,
)
from repositories.schemas.base import Base
from repositories.serialization.catalogs import (
    ReferenceCatalogSerializer,
)

###############################################################################
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

###############################################################################
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

###############################################################################
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

###############################################################################
def build_postgres_create_database_sql(
    database_name: str,
) -> TextClause:
    safe_database_name = validate_postgres_database_name(database_name)
    return sqlalchemy.text(
        f"CREATE DATABASE \"{safe_database_name}\" WITH ENCODING 'UTF8' TEMPLATE template0"
    )

###############################################################################
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
        try:
            written = serializer.replace_manifest_entries(
                manifest=manifest,
                manifest_hash=manifest_hash,
                source_path=str(path),
            )
            if written > 0:
                manifests_seeded += 1
                entries_written += written
        except Exception as exc:
            _ = exc
            raise
    return CatalogSeedResult(
        manifests_seen=len(paths),
        manifests_seeded=manifests_seeded,
        entries_written=entries_written,
    )

###############################################################################
def initialize_sqlite_database(
    settings: DatabaseSettings,
    *,
    drop_existing: bool = False,
    seed_catalogs: bool = True,
    force_reseed_catalogs: bool = False,
) -> None:
    repository = SQLiteRepository(settings)
    if drop_existing:
        Base.metadata.drop_all(repository.engine)
    Base.metadata.create_all(repository.engine)
    if seed_catalogs:
        session_factory = getattr(
            repository,
            "session_factory",
            sessionmaker(bind=repository.engine, future=True),
        )
        serializer = ReferenceCatalogSerializer(session_factory=session_factory)
        result = _seed_catalogs(serializer, force=force_reseed_catalogs)
        logger.info(
            "Catalog seeding completed for SQLite: seen=%s seeded=%s entries=%s",
            result.manifests_seen,
            result.manifests_seeded,
            result.entries_written,
        )
        catalog_provider.invalidate()
    logger.info("Initialized SQLite database schema at %s", repository.db_path)

###############################################################################
def ensure_postgres_database(
    settings: DatabaseSettings,
    *,
    drop_existing: bool = False,
    seed_catalogs: bool = True,
    force_reseed_catalogs: bool = False,
) -> str:
    if not settings.host:
        raise ValueError("Database host is required for PostgreSQL initialization.")
    if not settings.username:
        raise ValueError("Database username is required for PostgreSQL initialization.")
    if not settings.database_name:
        raise ValueError("Database name is required for PostgreSQL initialization.")

    target_database = validate_postgres_database_name(settings.database_name)
    connect_args = build_postgres_connect_args(settings)

    admin_url = build_postgres_url(settings, "postgres")
    admin_engine = sqlalchemy.create_engine(
        admin_url,
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

    with admin_engine.connect() as conn:
        exists = conn.execute(exists_stmt).scalar()
        if exists:
            logger.info("PostgreSQL database %s already exists", target_database)
        else:
            conn.execute(build_postgres_create_database_sql(target_database))
            logger.info("Created PostgreSQL database %s", target_database)

    normalized_settings = clone_settings_with_database(settings, target_database)
    repository = PostgresRepository(normalized_settings)
    if drop_existing:
        Base.metadata.drop_all(repository.engine)
    Base.metadata.create_all(repository.engine)
    if seed_catalogs:
        session_factory = getattr(repository, "session_factory", None)
        if session_factory is not None:
            serializer = ReferenceCatalogSerializer(session_factory=session_factory)
            result = _seed_catalogs(serializer, force=force_reseed_catalogs)
        else:
            result = None
        if result is not None:
            logger.info(
                "Catalog seeding completed for PostgreSQL: seen=%s seeded=%s entries=%s",
                result.manifests_seen,
                result.manifests_seeded,
                result.entries_written,
            )
            catalog_provider.invalidate()
    logger.info("Ensured PostgreSQL tables exist in %s", target_database)

    return target_database

###############################################################################
def run_database_initialization(
    *,
    drop_existing: bool = False,
    seed_catalogs: bool = True,
    force_reseed_catalogs: bool = False,
) -> None:
    settings = get_server_settings().database
    init_kwargs = {
        "drop_existing": drop_existing,
        "seed_catalogs": seed_catalogs,
        "force_reseed_catalogs": force_reseed_catalogs,
    }
    use_default_init_kwargs = (
        not drop_existing and seed_catalogs and not force_reseed_catalogs
    )
    if settings.backend == "sqlite":
        logger.info("Running SQLite initialization path.")
        if use_default_init_kwargs:
            initialize_sqlite_database(settings)
        else:
            initialize_sqlite_database(settings, **init_kwargs)
        return

    logger.info("Running PostgreSQL initialization path (manual trigger expected).")
    engine_name = normalize_postgres_engine(settings.engine).lower()
    if engine_name not in {
        "postgres",
        "postgresql",
        "postgresql+psycopg",
    }:
        raise ValueError(f"Unsupported database engine: {settings.engine}")

    if use_default_init_kwargs:
        ensure_postgres_database(settings)
    else:
        ensure_postgres_database(settings, **init_kwargs)

###############################################################################
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
    except (SQLAlchemyError, ValueError) as exc:
        logger.error("Database initialization failed: %s", exc)
        raise SystemExit(1) from exc
    except Exception as exc:
        logger.exception("Unexpected error during database initialization.")
        raise SystemExit(1) from exc
