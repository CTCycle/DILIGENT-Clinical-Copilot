from __future__ import annotations

import os

import lancedb
import pyarrow as pa
from lancedb.table import LanceTable


###############################################################################
class LanceVectorRepository:
    def __init__(self, database_path: str) -> None:
        self.database_path = database_path
        os.makedirs(self.database_path, exist_ok=True)
        self.connection = lancedb.connect(self.database_path)

    ############################################################################
    def table_exists(self, collection_name: str) -> bool:
        return collection_name in set(self.connection.table_names())

    ############################################################################
    def create_table(
        self, collection_name: str, data: pa.Table, *, mode: str
    ) -> LanceTable:
        return self.connection.create_table(collection_name, data=data, mode=mode)

    ############################################################################
    def open_table(self, collection_name: str) -> LanceTable:
        return self.connection.open_table(collection_name)

    ############################################################################
    def add_to_table(self, table: LanceTable, data: pa.Table) -> None:
        table.add(data)

    ############################################################################
    def count_rows(self, table: LanceTable) -> int:
        return int(table.count_rows())

    ############################################################################
    def create_index(
        self,
        table: LanceTable,
        *,
        metric: str,
        index_type: str,
        vector_column_name: str,
        num_partitions: int,
    ) -> None:
        table.create_index(
            metric=metric,
            index_type=index_type,
            vector_column_name=vector_column_name,
            num_partitions=num_partitions,
        )


__all__ = ["LanceVectorRepository"]
