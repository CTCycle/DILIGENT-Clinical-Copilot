from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from DILIGENT.server.repositories.database.backend import DILIGENTDatabase, database


###############################################################################
class DataRepositoryQueries:
    def __init__(self, db: DILIGENTDatabase = database) -> None:
        self.database = db

    # -------------------------------------------------------------------------
    def load_table(self, table_name: str) -> pd.DataFrame:
        return self.database.load_from_database(table_name)

    # -------------------------------------------------------------------------
    def upsert_table(self, dataset: pd.DataFrame, table_name: str) -> None:
        self.database.upsert_into_database(dataset, table_name)

    # -------------------------------------------------------------------------
    def count_rows(self, table_name: str) -> int:
        return self.database.count_rows(table_name)

    # -------------------------------------------------------------------------
    def stream_table(
        self, table_name: str, page_size: int
    ) -> Iterator[pd.DataFrame]:
        return self.database.stream_rows(table_name, page_size)

    # -------------------------------------------------------------------------
    def load_table_paginated(
        self, table_name: str, offset: int, limit: int
    ) -> pd.DataFrame:
        return self.database.load_paginated(table_name, offset, limit)
