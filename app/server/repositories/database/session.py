# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from configurations.startup import get_server_settings
from repositories.database.postgres import PostgresRepository
from repositories.database.sqlite import SQLiteRepository


###############################################################################
@lru_cache(maxsize=1)
def get_default_repository():
    settings = get_server_settings().database
    repository_cls = (
        SQLiteRepository if settings.backend == "sqlite" else PostgresRepository
    )
    return repository_cls(settings)


###############################################################################
def resolve_engine(engine: Engine | None = None) -> Engine:
    if engine is not None:
        return engine
    return get_default_repository().engine


###############################################################################
def resolve_session_factory(
    *,
    engine: Engine | None = None,
    session_factory: sessionmaker | None = None,
    expire_on_commit: bool = False,
) -> sessionmaker:
    if session_factory is not None:
        return session_factory
    return sessionmaker(
        bind=resolve_engine(engine),
        future=True,
        expire_on_commit=expire_on_commit,
    )


###############################################################################
def begin_immediate_write(db_session: Session) -> None:
    """Take SQLite's write lock before the transaction's first read.

    Deferred transactions that read and then write fail at once with "database
    is locked" when another writer commits in between. BEGIN IMMEDIATE waits on
    the busy timeout instead. Must run before any other statement in the
    session; PostgreSQL is left unchanged.
    """
    connection = db_session.connection()
    if connection.dialect.name != "sqlite":
        return
    driver_connection: Any = connection.connection.driver_connection
    previous_autocommit = getattr(driver_connection, "autocommit", None)
    if previous_autocommit is not None:
        driver_connection.autocommit = True
    try:
        driver_connection.execute("BEGIN IMMEDIATE")
    finally:
        if previous_autocommit is not None:
            driver_connection.autocommit = previous_autocommit


###############################################################################
@contextmanager
def unit_of_work(
    *,
    engine: Engine | None = None,
    session_factory: sessionmaker | None = None,
    immediate: bool = False,
) -> Iterator:
    """Own one transaction and never commit behind the caller's back.

    Set ``immediate`` for transactions that read before they write.
    """
    factory = resolve_session_factory(
        engine=engine,
        session_factory=session_factory,
        expire_on_commit=False,
    )
    db_session = factory()
    try:
        if immediate:
            begin_immediate_write(db_session)
        yield db_session
        db_session.commit()
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()
