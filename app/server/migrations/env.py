from __future__ import annotations

from alembic import context
from sqlalchemy.engine import Connection

from configurations.startup import get_server_settings
from repositories.database.engine import build_postgres_engine, build_sqlite_engine
from repositories.database.sqlite import resolve_sqlite_database_path
from repositories.schemas import Base

config = context.config

target_metadata = Base.metadata

###############################################################################
def _build_connectable():
    settings = get_server_settings().database
    if settings.backend == "sqlite":
        database_path = resolve_sqlite_database_path(settings)
        return build_sqlite_engine(str(database_path))
    if settings.backend == "postgresql":
        return build_postgres_engine(settings)
    raise ValueError(f"Unsupported database backend: {settings.backend}")

###############################################################################
def _configure(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        render_as_batch=True,
        transactional_ddl=True,
        transaction_per_migration=False,
    )

###############################################################################
def _run_migrations(connection: Connection) -> None:
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()

###############################################################################
def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError(
            "Offline Alembic generation requires sqlalchemy.url in the command configuration."
        )
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        render_as_batch=True,
        transactional_ddl=True,
    )
    with context.begin_transaction():
        context.run_migrations()

###############################################################################
def run_migrations_online() -> None:
    supplied_connection = config.attributes.get("connection")
    if supplied_connection is not None:
        _run_migrations(supplied_connection)
        return

    connectable = _build_connectable()
    try:
        with connectable.connect() as connection:
            _run_migrations(connection)
    finally:
        connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
