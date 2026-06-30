from __future__ import annotations

import sqlalchemy
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from common.paths import DATABASE_FILE_PATH
from common.utils.logger import logger
from domain.settings.configuration import DatabaseSettings
from repositories.schemas.models import Base, ClinicalSessionLab
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
        self.engine: Engine = sqlalchemy.create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            future=True,
            connect_args={"timeout": 30.0},
        )
        event.listen(self.engine, "connect", self._configure_connection)
        seed_session_factory = sessionmaker(
            bind=self.engine,
            future=True,
            expire_on_commit=False,
        )
        Base.metadata.create_all(self.engine)
        ensure_clinical_session_labs_observation_schema(self.engine)
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
                "SQLite DB file already existed; ensured additive schema and encryption material at %s",
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
            cursor.execute("PRAGMA journal_mode=MEMORY")
        finally:
            cursor.close()


###############################################################################
def ensure_clinical_session_labs_observation_schema(engine: Engine) -> None:
    with engine.begin() as connection:
        rows = connection.execute(
            sqlalchemy.text("PRAGMA table_info(clinical_session_labs)")
        ).mappings().all()
        columns = {str(row["name"]) for row in rows}
        if not columns or "observation_index" in columns:
            return
        connection.execute(
            sqlalchemy.text(
                "ALTER TABLE clinical_session_labs "
                "RENAME TO clinical_session_labs_legacy"
            )
        )
        ClinicalSessionLab.__table__.create(bind=connection)
        connection.execute(
            sqlalchemy.text(
                """
                INSERT INTO clinical_session_labs (
                    id,
                    session_id,
                    lab_code,
                    observation_index,
                    value_raw,
                    upper_limit_raw
                )
                SELECT
                    id,
                    session_id,
                    lab_code,
                    0,
                    value_raw,
                    upper_limit_raw
                FROM clinical_session_labs_legacy
                """
            )
        )
        connection.execute(sqlalchemy.text("DROP TABLE clinical_session_labs_legacy"))
