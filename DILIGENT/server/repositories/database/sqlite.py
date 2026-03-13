from __future__ import annotations

import os
from typing import Any, Iterator

import pandas as pd
import sqlalchemy
from sqlalchemy import UniqueConstraint, event, inspect, text
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from DILIGENT.server.configurations import DatabaseSettings
from DILIGENT.server.repositories.database.utils import (
    MISSING_TABLE_MESSAGE,
    validate_sql_identifier,
)
from DILIGENT.server.repositories.schemas.models import Base
from DILIGENT.server.common.constants import DATABASE_FILENAME, RESOURCES_PATH
from DILIGENT.server.common.utils.logger import logger


# [SQLITE DATABASE]
###############################################################################
class SQLiteRepository:
    def __init__(self, settings: DatabaseSettings) -> None:
        self.db_path: str | None = os.path.join(RESOURCES_PATH, DATABASE_FILENAME)
        should_initialize_schema = bool(self.db_path and not os.path.exists(self.db_path))
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.engine: Engine = sqlalchemy.create_engine(
            f"sqlite:///{self.db_path}", echo=False, future=True
        )
        event.listen(self.engine, "connect", self._enable_foreign_keys)
        if should_initialize_schema:
            Base.metadata.create_all(self.engine)
            logger.info(
                "SQLite DB file was missing; created and initialized schema at %s",
                self.db_path,
            )
        self.session_factory = sessionmaker(bind=self.engine, future=True)
        self.insert_batch_size = settings.insert_batch_size
        self.insert_commit_interval = settings.insert_commit_interval
        self.select_page_size = settings.select_page_size

    # -------------------------------------------------------------------------
    @staticmethod
    def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    def get_table_class(self, table_name: str) -> Any:
        for cls in Base.__subclasses__():
            if getattr(cls, "__tablename__", None) == table_name:
                return cls
        raise ValueError(f"No table class found for name {table_name}")

    # -------------------------------------------------------------------------
    def sanitize_table_name(self, table_name: str) -> str:
        normalized_name = validate_sql_identifier(table_name, label="table name")
        self.get_table_class(normalized_name)
        return normalized_name

    # -------------------------------------------------------------------------
    def upsert_dataframe(self, df: pd.DataFrame, table_cls) -> None:
        table = table_cls.__table__
        session = self.session_factory()
        try:
            unique_cols = []
            for uc in table.constraints:
                if isinstance(uc, UniqueConstraint):
                    unique_cols = uc.columns.keys()
                    break
            if not unique_cols:
                raise ValueError(f"No unique constraint found for {table_cls.__name__}")
            records = df.to_dict(orient="records")
            pending = 0
            for i in range(0, len(records), self.insert_batch_size):
                batch = records[i : i + self.insert_batch_size]
                if not batch:
                    continue
                stmt = insert(table).values(batch)
                update_cols = {
                    col: getattr(stmt.excluded, col)  # type: ignore[attr-defined]
                    for col in batch[0]
                    if col not in unique_cols
                }
                stmt = stmt.on_conflict_do_update(
                    index_elements=unique_cols, set_=update_cols
                )
                session.execute(stmt)
                pending += 1
                if pending >= self.insert_commit_interval:
                    session.commit()
                    pending = 0
            if pending:
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def load_from_database(self, table_name: str) -> pd.DataFrame:
        safe_table_name = self.sanitize_table_name(table_name)
        with self.engine.connect() as conn:
            inspector = inspect(conn)
            if not inspector.has_table(safe_table_name):
                logger.warning(MISSING_TABLE_MESSAGE, safe_table_name)
                return pd.DataFrame()
            data = pd.read_sql_table(safe_table_name, conn)
        return data


    # -------------------------------------------------------------------------
    def upsert_into_database(self, df: pd.DataFrame, table_name: str) -> None:
        safe_table_name = self.sanitize_table_name(table_name)
        table_cls = self.get_table_class(safe_table_name)
        self.upsert_dataframe(df, table_cls)

    # -----------------------------------------------------------------------------
    def count_rows(self, table_name: str) -> int:
        safe_table_name = self.sanitize_table_name(table_name)
        with self.engine.connect() as conn:
            result = conn.execute(
                sqlalchemy.text(f'SELECT COUNT(*) FROM "{safe_table_name}"')
            )
            value = result.scalar() or 0
        return int(value)

    # -------------------------------------------------------------------------
    def stream_rows(self, table_name: str, page_size: int) -> Iterator[pd.DataFrame]:
        safe_table_name = self.sanitize_table_name(table_name)
        chunk_size = page_size if page_size > 0 else self.select_page_size
        if chunk_size <= 0:
            yield self.load_from_database(safe_table_name)
            return
        with self.engine.connect() as conn:
            inspector = inspect(conn)
            if not inspector.has_table(safe_table_name):
                logger.warning(MISSING_TABLE_MESSAGE, safe_table_name)
                return
            query = text(f'SELECT * FROM "{safe_table_name}"')
            for chunk in pd.read_sql_query(query, conn, chunksize=chunk_size):
                yield chunk

    # -------------------------------------------------------------------------
    def load_paginated(
        self, table_name: str, offset: int, limit: int
    ) -> pd.DataFrame:
        safe_table_name = self.sanitize_table_name(table_name)
        safe_offset = max(int(offset), 0)
        safe_limit = max(int(limit), 1)
        with self.engine.connect() as conn:
            inspector = inspect(conn)
            if not inspector.has_table(safe_table_name):
                logger.warning(MISSING_TABLE_MESSAGE, safe_table_name)
                return pd.DataFrame()
            query = text(f'SELECT * FROM "{safe_table_name}" LIMIT :limit OFFSET :offset')
            data = pd.read_sql_query(
                query,
                conn,
                params={"limit": safe_limit, "offset": safe_offset},
            )
        return data



