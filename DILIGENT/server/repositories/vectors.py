from __future__ import annotations

import json
import os
from typing import Any, Iterator, Literal, cast

import lancedb
import pyarrow as pa
from lancedb.db import DBConnection
from lancedb.table import Table

from DILIGENT.server.common.utils.logger import logger

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

DistanceType = Literal["l2", "cosine", "dot"]
IndexType = Literal["IVF_FLAT", "IVF_PQ", "IVF_HNSW_SQ", "IVF_HNSW_PQ"]


###############################################################################
class LanceVectorDatabase:
    def __init__(
        self,
        database_path: str,
        collection_name: str,
        schema: pa.Schema | None = None,
        metric: str | None = None,
        index_type: str | None = None,
        stream_batch_size: int | None = None,
    ) -> None:
        self.database_path = database_path
        self.collection_name = collection_name
        self.schema = schema or VECTOR_TABLE_SCHEMA
        self.metric = metric
        self.index_type = index_type
        self.stream_batch_size = stream_batch_size
        self.connection: DBConnection | None = None
        self.table: Table | None = None
        self.index_ready = False
        self.index_creation_attempted = False
        self.embedding_size: int | None = None

    # -------------------------------------------------------------------------
    def connect(self) -> DBConnection:
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
    def initialize(self) -> None:
        self.connect()

    # -------------------------------------------------------------------------
    def get_table(self) -> Table:
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
        if self.embedding_size is not None and self.embedding_size > 0:
            records, discarded = self._filter_records_by_embedding_size(
                records,
                self.embedding_size,
            )
            if discarded:
                logger.warning(
                    "Discarded %d embeddings with unexpected dimension (expected %d)",
                    discarded,
                    self.embedding_size,
                )
            if not records:
                return
        table = self.get_table()
        table.add(records)
        self.ensure_vector_index(table)

    # -------------------------------------------------------------------------
    def configure_embedding_size(self, embedding_size: int) -> None:
        if embedding_size <= 0:
            return
        persisted_size = self._read_existing_embedding_size()
        if persisted_size is not None:
            self.embedding_size = persisted_size
            if embedding_size != persisted_size:
                logger.warning(
                    "Skipping embeddings with dimension %d because LanceDB table '%s' uses dimension %d",
                    embedding_size,
                    self.collection_name,
                    persisted_size,
                )
            return
        if self.embedding_size is None:
            self.embedding_size = embedding_size
        elif self.embedding_size != embedding_size:
            logger.warning(
                "Skipping embeddings with dimension %d because active dataset uses dimension %d",
                embedding_size,
                self.embedding_size,
            )
            return
        connection = self.connect()
        if self.table is None and self.collection_name not in connection.table_names():
            self.schema = self._schema_with_embedding_size(self.embedding_size)

    # -------------------------------------------------------------------------
    def ensure_vector_index(self, table: Table | None = None) -> None:
        if not self.metric or not self.index_type:
            return
        if self.index_ready:
            return
        table = table or self.get_table()
        try:
            indices = cast(list[Any], table.list_indices())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unable to inspect LanceDB indices: %s", exc)
            return
        for index in indices:
            if self._index_has_embedding_column(index):
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
                metric=cast(DistanceType, self.metric),
                index_type=cast(IndexType, self.index_type),
            )
            self.index_ready = True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to create LanceDB vector index: %s", exc)
            self.index_ready = False

    # -------------------------------------------------------------------------
    def _read_existing_embedding_size(self) -> int | None:
        table = self.table
        if table is None:
            connection = self.connect()
            if self.collection_name not in connection.table_names():
                return None
            try:
                table = connection.open_table(self.collection_name)
            except ValueError:
                return None
            self.table = table
        try:
            embedding_field = table.schema.field("embedding")
        except KeyError:
            return None
        if isinstance(embedding_field.type, pa.FixedSizeListType):
            return int(embedding_field.type.list_size)
        return None

    def _schema_with_embedding_size(self, embedding_size: int) -> pa.Schema:
        desired_type = pa.list_(pa.float32(), embedding_size)
        fields: list[pa.Field] = []
        for field in self.schema:
            if field.name == "embedding":
                fields.append(pa.field("embedding", desired_type))
            else:
                fields.append(field)
        return pa.schema(fields)

    def _filter_records_by_embedding_size(
        self, records: list[dict[str, Any]], embedding_size: int
    ) -> tuple[list[dict[str, Any]], int]:
        valid_records: list[dict[str, Any]] = []
        discarded = 0
        for record in records:
            vector = record.get("embedding")
            if isinstance(vector, list) and len(vector) == embedding_size:
                valid_records.append(record)
            else:
                discarded += 1
        return valid_records, discarded

    def _index_has_embedding_column(self, index: Any) -> bool:
        column = (
            self._index_value(index, "column")
            or self._index_value(index, "columns")
            or self._index_value(index, "vector_column")
            or self._index_value(index, "vector_column_name")
        )
        for entry in self._iter_column_entries(column):
            name = self._column_entry_name(entry)
            if name == "embedding":
                return True
        return False

    def _iter_column_entries(self, column: Any) -> Iterator[Any]:
        if column is None:
            return iter(())
        if isinstance(column, (list, tuple, set)):
            return iter(column)
        return iter((column,))

    def _column_entry_name(self, entry: Any) -> str | None:
        if isinstance(entry, str):
            return entry
        return (
            self._index_value(entry, "name")
            or self._index_value(entry, "column")
            or self._index_value(entry, "vector_column")
            or self._index_value(entry, "vector_column_name")
        )

    @staticmethod
    def _index_value(index: Any, key: str) -> Any:
        if isinstance(index, dict):
            return index.get(key)
        return getattr(index, key, None)

    # -------------------------------------------------------------------------
    def iter_embeddings(
        self,
        batch_size: int | None = None,
        limit: int | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        table = self.get_table()
        resolved_batch = batch_size or self.stream_batch_size or 1024
        remaining = limit
        batches = table.to_arrow().to_batches(resolved_batch)
        for batch in batches:
            records = batch.to_pylist()
            if remaining is not None:
                if remaining <= 0:
                    break
                if len(records) > remaining:
                    yield records[:remaining]
                    break
                remaining -= len(records)
            yield records

    # -------------------------------------------------------------------------
    def load_embeddings(self, limit: int | None = None) -> list[dict[str, Any]]:
        if self.stream_batch_size:
            records: list[dict[str, Any]] = []
            remaining = limit
            for batch in self.iter_embeddings(self.stream_batch_size, limit=remaining):
                records.extend(batch)
                if remaining is not None:
                    remaining -= len(batch)
                    if remaining <= 0:
                        break
            if limit is not None and len(records) > limit:
                return records[:limit]
            return records
        table = self.get_table()
        data = table.to_arrow()
        if limit is not None:
            data = data.slice(0, limit)
        return data.to_pylist()

    # -------------------------------------------------------------------------
    def to_json(self, limit: int | None = None) -> str:
        records = self.load_embeddings(limit=limit)
        return json.dumps(records, ensure_ascii=False, indent=2)


VectorDatabase = LanceVectorDatabase
