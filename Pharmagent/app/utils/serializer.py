from __future__ import annotations

import pandas as pd
from typing import Any

from Pharmagent.app.constants import LIVERTOX_TABLE_NAME
from Pharmagent.app.utils.database.sqlite import database


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
        columns = ["nbk_id", "drug_name", "excerpt"]
        if not records:
            empty = pd.DataFrame(columns=columns)
            database.save_into_database(empty, LIVERTOX_TABLE_NAME)
            return
        df = pd.DataFrame(records)
        for column in columns:
            if column not in df.columns:
                df[column] = ""
        ordered = df[columns]
        database.save_into_database(ordered, LIVERTOX_TABLE_NAME)
