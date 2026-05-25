from __future__ import annotations

import os

import sqlalchemy
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from domain.settings.configuration import DatabaseSettings
from repositories.serialization.access_key_encryption import (
    AccessKeyEncryptionMaterialSerializer,
)
from repositories.serialization.text_normalization import (
    TextNormalizationVocabularySerializer,
)
from repositories.schemas.models import Base
from common.constants import DATABASE_FILENAME, RESOURCES_PATH
from common.utils.logger import logger


class SQLiteRepository:
    def __init__(self, settings: DatabaseSettings) -> None:
        override_path = os.getenv("DILIGENT_SQLITE_PATH", "").strip()
        if override_path:
            self.db_path = os.path.abspath(override_path)
        else:
            self.db_path = os.path.join(RESOURCES_PATH, DATABASE_FILENAME)
        should_initialize_schema = bool(
            self.db_path and not os.path.exists(self.db_path)
        )
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.engine: Engine = sqlalchemy.create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            future=True,
        )
        event.listen(self.engine, "connect", self._enable_foreign_keys)
        seed_session_factory = sessionmaker(
            bind=self.engine,
            future=True,
            expire_on_commit=False,
        )
        if should_initialize_schema:
            Base.metadata.create_all(self.engine)
            AccessKeyEncryptionMaterialSerializer(
                engine=self.engine,
                session_factory=seed_session_factory,
            ).ensure_seeded("provider_access_keys")
            logger.info(
                "SQLite DB file was missing; created and initialized schema at %s",
                self.db_path,
            )
        TextNormalizationVocabularySerializer(
            engine=self.engine,
            session_factory=seed_session_factory,
        ).ensure_seeded()
        self.session_factory = sessionmaker(bind=self.engine, future=True)

    @staticmethod
    def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()


