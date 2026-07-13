from __future__ import annotations

from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from domain.settings.configuration import DatabaseSettings
from repositories.database.engine import build_postgres_engine
from repositories.serialization.catalogs import ReferenceCatalogSerializer

###############################################################################
class PostgresRepository:

    # -------------------------------------------------------------------------
    def __init__(self, settings: DatabaseSettings) -> None:
        if not settings.host:
            raise ValueError("Database host must be provided for external database.")
        if not settings.database_name:
            raise ValueError("Database name must be provided for external database.")
        if not settings.username:
            raise ValueError(
                "Database username must be provided for external database."
            )

        self.db_path: str | None = None
        self.engine: Engine = build_postgres_engine(settings)
        self.session_factory = sessionmaker(bind=self.engine, future=True)
        self.catalogs = ReferenceCatalogSerializer(session_factory=self.session_factory)
