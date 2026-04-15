from __future__ import annotations

import os

import sqlalchemy
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from DILIGENT.server.domain.settings.configuration import DatabaseSettings
from DILIGENT.server.repositories.serialization.access_key_encryption import (
    AccessKeyEncryptionMaterialSerializer,
)
from DILIGENT.server.repositories.schemas.models import Base
from DILIGENT.server.common.constants import DATABASE_FILENAME, RESOURCES_PATH
from DILIGENT.server.common.utils.logger import logger


class SQLiteRepository:
    def __init__(self, settings: DatabaseSettings) -> None:
        self.db_path: str | None = os.path.join(RESOURCES_PATH, DATABASE_FILENAME)
        should_initialize_schema = bool(self.db_path and not os.path.exists(self.db_path))
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.engine: Engine = sqlalchemy.create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            future=True,
        )
        event.listen(self.engine, "connect", self._enable_foreign_keys)
        if should_initialize_schema:
            Base.metadata.create_all(self.engine)
            seed_session_factory = sessionmaker(
                bind=self.engine,
                future=True,
                expire_on_commit=False,
            )
            AccessKeyEncryptionMaterialSerializer(
                engine=self.engine,
                session_factory=seed_session_factory,
            ).ensure_seeded("provider_access_keys")
            logger.info(
                "SQLite DB file was missing; created and initialized schema at %s",
                self.db_path,
            )
        else:
            logger.info(
                "SQLite DB file already present at %s; skipping automatic schema initialization.",
                self.db_path,
            )
        self.session_factory = sessionmaker(bind=self.engine, future=True)

    @staticmethod
    def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()
