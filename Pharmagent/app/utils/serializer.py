from __future__ import annotations

import pandas as pd
from typing import Any

from Pharmagent.app.utils.database.sqlite import database


LIVERTOX_UPSERT_BATCH_SIZE = 500


# [DATA SERIALIZATION]
###############################################################################
class DataSerializer:
    def __init__(self) -> None:
        pass

    # -------------------------------------------------------------------------
    def save_patients_info(self, patients: dict[str, Any]) -> None:
        data = pd.DataFrame([patients])
        database.upsert_into_database(data, "PATIENTS")

    # -----------------------------------------------------------------------------
    def save_livertox_records(self, records: list[dict[str, Any]]) -> None:
        prepared: list[dict[str, Any]] = []
        batch_size = max(1, LIVERTOX_UPSERT_BATCH_SIZE)
        for record in records:
            nbk_id = record.get("nbk_id")
            drug_name = record.get("drug_name")
            if nbk_id is None or drug_name is None:
                continue
            excerpt = record.get("excerpt")
            prepared.append(
                {
                    "nbk_id": str(nbk_id),
                    "drug_name": str(drug_name),
                    "excerpt": str(excerpt) if excerpt is not None else None,
                }
            )
            if len(prepared) >= batch_size:
                data = pd.DataFrame(prepared)
                if not data.empty:
                    database.upsert_into_database(data, "LIVERTOX_MONOGRAPHS")
                prepared.clear()

        if prepared:
            data = pd.DataFrame(prepared)
            if not data.empty:
                database.upsert_into_database(data, "LIVERTOX_MONOGRAPHS")

    # -----------------------------------------------------------------------------
    def load_from_database(self, table_name: str) -> pd.DataFrame:
        query = f'SELECT * FROM "{table_name}"'
        with database.engine.connect() as connection:
            return pd.read_sql_query(query, connection)

    # -----------------------------------------------------------------------------
    def get_livertox_records(self) -> pd.DataFrame:
        return self.load_from_database("LIVERTOX_MONOGRAPHS")
