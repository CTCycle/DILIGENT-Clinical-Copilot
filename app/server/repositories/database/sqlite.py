from __future__ import annotations

from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from common.paths import DATABASE_FILE_PATH
from domain.settings.configuration import DatabaseSettings
from repositories.database.engine import build_sqlite_engine
from repositories.serialization.catalogs import ReferenceCatalogSerializer

###############################################################################
def resolve_sqlite_database_path(settings: DatabaseSettings) -> Path:
    return Path(settings.sqlite_path) if settings.sqlite_path else DATABASE_FILE_PATH

###############################################################################
class SQLiteRepository:

    # -------------------------------------------------------------------------
    def __init__(self, settings: DatabaseSettings) -> None:
        self.db_path = resolve_sqlite_database_path(settings)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = build_sqlite_engine(str(self.db_path), timeout=30.0)
        event.listen(self.engine, "connect", self._configure_connection)
        self.session_factory = sessionmaker(bind=self.engine, future=True)
        self.catalogs = ReferenceCatalogSerializer(session_factory=self.session_factory)

    # -------------------------------------------------------------------------
    @staticmethod
    def _configure_connection(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
            # WAL keeps readers independent from the single SQLite writer and
            # remains durable across connections and process restarts.
            cursor.execute("PRAGMA journal_mode=WAL")
        finally:
            cursor.close()
