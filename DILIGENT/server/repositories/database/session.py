from __future__ import annotations

import importlib
from functools import lru_cache

from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from DILIGENT.server.configurations.startup import server_settings


REPOSITORY_CLASS_PATHS = {
    True: ("DILIGENT.server.repositories.database.sqlite", "SQLiteRepository"),
    False: ("DILIGENT.server.repositories.database.postgres", "PostgresRepository"),
}


def _resolve_repository_class(embedded_database: bool):
    module_name, class_name = REPOSITORY_CLASS_PATHS[bool(embedded_database)]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


@lru_cache(maxsize=1)
def get_default_repository():
    settings = server_settings.database
    repository_cls = _resolve_repository_class(settings.embedded_database)
    return repository_cls(settings)


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
