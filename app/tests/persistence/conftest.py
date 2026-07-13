from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from repositories.schemas.models import Base


###############################################################################
@pytest.fixture(params=["sqlite", "postgresql"])
def persistence_engine(request: pytest.FixtureRequest, tmp_path: Path) -> Engine:
    if request.param == "sqlite":
        engine = create_engine(
            f"sqlite+pysqlite:///{tmp_path / 'persistence.db'}",
            future=True,
            connect_args={"timeout": 30.0},
        )

        @event.listens_for(engine, "connect")
        def configure_sqlite(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.execute("PRAGMA journal_mode=WAL")
            finally:
                cursor.close()
    else:
        url = os.getenv("TEST_DATABASE_URL")
        if not url:
            pytest.skip("TEST_DATABASE_URL is not configured")
        engine = create_engine(url, future=True, pool_pre_ping=True)

    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


###############################################################################
@pytest.fixture
def persistence_session(persistence_engine: Engine) -> Session:
    factory = sessionmaker(bind=persistence_engine, future=True, expire_on_commit=False)
    with factory() as db_session:
        yield db_session
