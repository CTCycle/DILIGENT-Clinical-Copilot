from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from repositories.database.migrations import migrate_database


###############################################################################
@pytest.fixture(params=["sqlite", "postgresql"])
def persistence_engine(request: pytest.FixtureRequest, tmp_path: Path) -> Engine:
    if request.param == "sqlite":
        engine = create_engine(
            f"sqlite+pysqlite:///{tmp_path / 'persistence.db'}",
            future=True,
            connect_args={"timeout": 30.0, "autocommit": False},
        )

        @event.listens_for(engine, "connect")
        def configure_sqlite(dbapi_connection, _connection_record) -> None:
            previous_autocommit = getattr(dbapi_connection, "autocommit", None)
            if previous_autocommit is not None:
                dbapi_connection.autocommit = True
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.execute("PRAGMA journal_mode=WAL")
            finally:
                cursor.close()
                if previous_autocommit is not None:
                    dbapi_connection.autocommit = previous_autocommit
    else:
        url = os.getenv("TEST_DATABASE_URL")
        if not url:
            pytest.skip("TEST_DATABASE_URL is not configured")
        engine = create_engine(url, future=True, pool_pre_ping=True)

    migrate_database(engine, database_was_empty=True)
    try:
        yield engine
    finally:
        migrate_database(engine, database_was_empty=False, drop_existing=True)
        engine.dispose()


###############################################################################
@pytest.fixture
def persistence_session(persistence_engine: Engine) -> Session:
    factory = sessionmaker(bind=persistence_engine, future=True, expire_on_commit=False)
    with factory() as db_session:
        yield db_session
