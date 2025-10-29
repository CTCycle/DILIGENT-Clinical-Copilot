from __future__ import annotations

import json
import os
from typing import Any

import lancedb
import pyarrow as pa
from lancedb import LanceDBConnection
from lancedb.table import LanceTable

from DILIGENT.app.logger import logger


VECTOR_TABLE_SCHEMA = pa.schema(
    [
        pa.field("document_id", pa.string()),
        pa.field("chunk_id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("embedding", pa.list_(pa.float32())),
        pa.field("source", pa.string()),
        pa.field("metadata", pa.string()),
    ]
)


###############################################################################
class LanceVectorDatabase:
    def __init__(
        self,
        database_path: str,
        collection_name: str,
        schema: pa.Schema | None = None,
        metric: str | None = None,
        index_type: str | None = None,
    ) -> None:
        self.database_path = database_path
        self.collection_name = collection_name
        self.schema = schema or VECTOR_TABLE_SCHEMA
        self.metric = metric
        self.index_type = index_type
        self.connection: LanceDBConnection | None = None
        self.table: LanceTable | None = None

    # -------------------------------------------------------------------------
    def connect(self) -> LanceDBConnection:
        if self.connection is None:
            os.makedirs(self.database_path, exist_ok=True)
            self.connection = lancedb.connect(self.database_path)
        return self.connection

    # -------------------------------------------------------------------------
    def initialize(self, reset_table: bool = False) -> None:
        connection = self.connect()
        if reset_table and self.collection_name in connection.table_names():
            logger.info(
                "Dropping existing LanceDB table '%s'", self.collection_name
            )
            connection.drop_table(self.collection_name)
        if reset_table:
            self.table = None

    # -------------------------------------------------------------------------
    def get_table(self) -> LanceTable:
        if self.table is not None:
            return self.table
        connection = self.connect()
        if self.collection_name in connection.table_names():
            self.table = connection.open_table(self.collection_name)
        else:
            logger.info(
                "Creating LanceDB table '%s' at %s",
                self.collection_name,
                self.database_path,
            )
            self.table = connection.create_table(
                self.collection_name,
                data=[],
                schema=self.schema,
                mode="create",
            )
        return self.table

    # -------------------------------------------------------------------------
    def upsert_embeddings(self, records: list[dict[str, Any]]) -> None:
        if not records:
            logger.info("No embedding records to persist into LanceDB")
            return
        table = self.get_table()
        table.add(records)
        self.ensure_vector_index(table)

    # -------------------------------------------------------------------------
    def ensure_vector_index(self, table: LanceTable | None = None) -> None:
        if not self.metric or not self.index_type:
            return
        table = table or self.get_table()
        try:
            indices = table.list_indices()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unable to inspect LanceDB indices: %s", exc)
            return
        for index in indices:
            column = index.get("column") or index.get("columns")
            if column == "embedding" or (
                isinstance(column, list) and "embedding" in column
            ):
                return
        try:
            table.create_index(
                column="embedding",
                metric=self.metric,
                index_type=self.index_type,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to create LanceDB vector index: %s", exc)

    # -------------------------------------------------------------------------
    def load_embeddings(self, limit: int | None = None) -> list[dict[str, Any]]:
        table = self.get_table()
        data = table.to_arrow()
        if limit is not None:
            data = data.slice(0, limit)
        return data.to_pylist()

    # -------------------------------------------------------------------------
    def drop(self) -> None:
        connection = self.connect()
        if self.collection_name in connection.table_names():
            logger.info("Dropping LanceDB table '%s'", self.collection_name)
            connection.drop_table(self.collection_name)
        self.table = None

    # -------------------------------------------------------------------------
    def to_json(self, limit: int | None = None) -> str:
        records = self.load_embeddings(limit=limit)
        return json.dumps(records, ensure_ascii=False, indent=2)


VectorDatabase = LanceVectorDatabase

