from __future__ import annotations

import json
import os
from typing import Any

import lancedb
import pyarrow as pa
from lancedb import LanceDBConnection
from lancedb.table import LanceTable

from DILIGENT.src.packages.logger import logger


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
        self.index_ready = False
        self.index_creation_attempted = False
        self.embedding_size: int | None = None

    # -------------------------------------------------------------------------
    def connect(self) -> LanceDBConnection:
        if self.connection is None:
            path = self.database_path
            suffix = os.path.splitext(path)[1]
            base_directory = os.path.dirname(path) if suffix else path
            if not base_directory:
                base_directory = "."
            os.makedirs(base_directory, exist_ok=True)
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
            self.index_ready = False
            self.index_creation_attempted = False
            self.embedding_size = None

    # -------------------------------------------------------------------------
    def get_table(self) -> LanceTable:
        if self.table is not None:
            return self.table
        connection = self.connect()
        if self.collection_name in connection.table_names():
            try:
                self.table = connection.open_table(self.collection_name)
            except ValueError:
                self.table = None
        if self.table is None:
            logger.info(
                "Creating LanceDB table '%s' at %s",
                self.collection_name,
                self.database_path,
            )
            empty_table = self.schema.empty_table()
            self.table = connection.create_table(
                self.collection_name,
                data=empty_table,
                schema=self.schema,
                mode="create",
            )
            self.index_ready = False
            self.index_creation_attempted = False
        return self.table

    # -------------------------------------------------------------------------
    def upsert_embeddings(self, records: list[dict[str, Any]]) -> None:
        if not records:
            logger.info("No embedding records to persist into LanceDB")
            return
        embedding_length = 0
        first_embedding = records[0].get("embedding")
        if isinstance(first_embedding, list):
            embedding_length = len(first_embedding)
        if embedding_length > 0:
            self.configure_embedding_size(embedding_length)
        table = self.get_table()
        table.add(records)
        self.ensure_vector_index(table)

    # -------------------------------------------------------------------------
    def configure_embedding_size(self, embedding_size: int) -> None:
        if embedding_size <= 0:
            return
        if self.embedding_size == embedding_size:
            return
        try:
            current_field = self.schema.field("embedding")
        except KeyError:
            return
        desired_type = pa.list_(pa.float32(), embedding_size)
        if isinstance(current_field.type, pa.FixedSizeListType):
            if current_field.type.list_size == embedding_size:
                self.embedding_size = embedding_size
                return
        fields: list[pa.Field] = []
        for field in self.schema:
            if field.name == "embedding":
                fields.append(pa.field("embedding", desired_type))
            else:
                fields.append(field)
        new_schema = pa.schema(fields)
        if new_schema == self.schema and self.embedding_size == embedding_size:
            return
        existing_records: list[dict[str, Any]] = []
        if self.table is not None:
            row_count = 0
            try:
                row_count = self.table.count_rows()
            except Exception:  # noqa: BLE001
                row_count = 0
            if row_count > 0:
                try:
                    existing_records = self.table.to_pylist()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Unable to cache existing embeddings while updating LanceDB schema: %s",
                        exc,
                    )
        connection = self.connect()
        if self.collection_name in connection.table_names():
            logger.info(
                "Recreating LanceDB table '%s' with embedding size %d",
                self.collection_name,
                embedding_size,
            )
            connection.drop_table(self.collection_name)
        self.schema = new_schema
        self.table = None
        self.index_ready = False
        self.index_creation_attempted = False
        self.embedding_size = embedding_size
        table = self.get_table()
        if existing_records:
            valid_records: list[dict[str, Any]] = []
            discarded = 0
            for record in existing_records:
                vector = record.get("embedding")
                if isinstance(vector, list) and len(vector) == embedding_size:
                    valid_records.append(record)
                else:
                    discarded += 1
            if discarded:
                logger.warning(
                    "Discarded %d embeddings that did not match new dimension %d during LanceDB schema update",
                    discarded,
                    embedding_size,
                )
            if valid_records:
                table.add(valid_records)

    # -------------------------------------------------------------------------
    def ensure_vector_index(self, table: LanceTable | None = None) -> None:
        if not self.metric or not self.index_type:
            return
        if self.index_ready:
            return
        table = table or self.get_table()
        try:
            indices = table.list_indices()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unable to inspect LanceDB indices: %s", exc)
            return
        for index in indices:
            column = (
                index.get("column")
                or index.get("columns")
                or index.get("vector_column")
                or index.get("vector_column_name")
            )
            if column == "embedding":
                self.index_ready = True
                return
            if isinstance(column, dict):
                name = (
                    column.get("name")
                    or column.get("column")
                    or column.get("vector_column")
                    or column.get("vector_column_name")
                )
                if name == "embedding":
                    self.index_ready = True
                    return
            if isinstance(column, (list, tuple)):
                for entry in column:
                    if entry == "embedding":
                        self.index_ready = True
                        return
                    if isinstance(entry, dict):
                        name = (
                            entry.get("name")
                            or entry.get("column")
                            or entry.get("vector_column")
                            or entry.get("vector_column_name")
                        )
                        if name == "embedding":
                            self.index_ready = True
                            return
        if self.index_creation_attempted:
            return
        self.index_creation_attempted = True
        try:
            schema = table.schema
            embedding_field = schema.field("embedding")
        except KeyError:
            logger.warning(
                "Skipping LanceDB index creation because 'embedding' field is missing"
            )
            return
        if not isinstance(embedding_field.type, pa.FixedSizeListType):
            logger.warning(
                "Skipping LanceDB vector index creation; expected FixedSizeList for 'embedding' column but received %s",
                embedding_field.type,
            )
            return
        try:
            table.create_index(
                vector_column_name="embedding",
                metric=self.metric,
                index_type=self.index_type,
            )
            self.index_ready = True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to create LanceDB vector index: %s", exc)
            self.index_ready = False

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
        self.index_ready = False
        self.index_creation_attempted = False

    # -------------------------------------------------------------------------
    def to_json(self, limit: int | None = None) -> str:
        records = self.load_embeddings(limit=limit)
        return json.dumps(records, ensure_ascii=False, indent=2)


VectorDatabase = LanceVectorDatabase

