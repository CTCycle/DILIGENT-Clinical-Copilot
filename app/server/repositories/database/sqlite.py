from __future__ import annotations

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from common.paths import DATABASE_FILE_PATH
from common.utils.logger import logger
from domain.settings.configuration import DatabaseSettings
from repositories.schemas.models import Base
from repositories.database.engine import build_sqlite_engine
from repositories.serialization.access_key_encryption import (
    AccessKeyEncryptionMaterialSerializer,
)
from repositories.serialization.catalogs import ReferenceCatalogSerializer

###############################################################################
class SQLiteRepository:

    # -------------------------------------------------------------------------
    def __init__(self, settings: DatabaseSettings) -> None:
        self.db_path = DATABASE_FILE_PATH
        db_file_missing = bool(self.db_path and not self.db_path.exists())
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = build_sqlite_engine(str(self.db_path), timeout=30.0)
        event.listen(self.engine, "connect", self._configure_connection)
        seed_session_factory = sessionmaker(
            bind=self.engine,
            future=True,
            expire_on_commit=False,
        )
        if db_file_missing:
            Base.metadata.create_all(self.engine)
        AccessKeyEncryptionMaterialSerializer(
            engine=self.engine,
            session_factory=seed_session_factory,
        ).ensure_seeded("provider_access_keys")
        if db_file_missing:
            logger.info(
                "SQLite DB file was missing; created and initialized schema at %s",
                str(self.db_path),
            )
        else:
            logger.info(
                "SQLite DB file already existed; ensured encryption material at %s",
                str(self.db_path),
            )
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
