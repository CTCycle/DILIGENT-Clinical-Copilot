from __future__ import annotations

from functools import lru_cache

from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from configurations.startup import get_server_settings
from repositories.database.postgres import PostgresRepository
from repositories.database.sqlite import SQLiteRepository

###############################################################################
@lru_cache(maxsize=1)
def get_default_repository():
    settings = get_server_settings().database
    repository_cls = SQLiteRepository if settings.embedded_database else PostgresRepository
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

