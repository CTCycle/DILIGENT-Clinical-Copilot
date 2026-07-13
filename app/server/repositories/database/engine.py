from __future__ import annotations

import urllib.parse
from typing import Any

import sqlalchemy
from sqlalchemy.engine import Engine

from domain.settings.configuration import DatabaseSettings
from repositories.database.utils import normalize_postgres_engine, validate_postgres_database_name


def build_sqlite_engine(database_path: str, *, timeout: float = 30.0) -> Engine:
    return sqlalchemy.create_engine(
        f"sqlite:///{database_path}",
        echo=False,
        future=True,
        connect_args={"timeout": timeout},
    )


def build_postgres_engine(settings: DatabaseSettings) -> Engine:
    if not settings.host or not settings.database_name or not settings.username:
        raise ValueError("PostgreSQL host, database name, and username are required")
    engine_name = normalize_postgres_engine(settings.engine)
    username = urllib.parse.quote_plus(settings.username)
    password = urllib.parse.quote_plus(settings.password or "")
    database_name = validate_postgres_database_name(settings.database_name)
    connect_args: dict[str, Any] = {"connect_timeout": settings.connect_timeout}
    if settings.ssl:
        connect_args["sslmode"] = "require"
        if settings.ssl_ca:
            connect_args["sslrootcert"] = settings.ssl_ca
    return sqlalchemy.create_engine(
        f"{engine_name}://{username}:{password}@{settings.host}:{settings.port or 5432}/{database_name}",
        echo=False,
        future=True,
        connect_args=connect_args,
        pool_pre_ping=True,
        pool_timeout=30,
        pool_recycle=1800,
    )
