from __future__ import annotations

from functools import lru_cache

from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from DILIGENT.server.configurations.bootstrap import server_settings


@lru_cache(maxsize=1)
def get_default_repository():
    settings = server_settings.database
    if settings.embedded_database:
        from DILIGENT.server.repositories.database.sqlite import SQLiteRepository

        return SQLiteRepository(settings)
    from DILIGENT.server.repositories.database.postgres import PostgresRepository

    return PostgresRepository(settings)


def resolve_engine(engine: Engine | None = None) -> Engine:
    if engine is not None:
        return engine
    return get_default_repository().engine


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
