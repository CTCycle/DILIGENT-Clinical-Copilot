from __future__ import annotations

import re
from typing import Any

import pandas as pd

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
        sanitized = self.sanitize(records)
        if sanitized.empty:
            return
        sanitized = sanitized.where(pd.notnull(sanitized), None)
        batch_size = max(1, LIVERTOX_UPSERT_BATCH_SIZE)
        total_rows = len(sanitized)
        for start in range(0, total_rows, batch_size):
            batch = sanitized.iloc[start : start + batch_size]
            if not batch.empty:
                database.upsert_into_database(batch, "LIVERTOX_MONOGRAPHS")

    # -----------------------------------------------------------------------------
    def sanitize(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        required_columns = [
            "nbk_id",
            "drug_name",
            "excerpt",
            "additional_names",
            "synonyms",
        ]
        if df.empty:
            return pd.DataFrame(columns=required_columns)
        for column in required_columns:
            if column not in df.columns:
                df[column] = None
        df = df[required_columns]
        allowed_missing = {"nbk_id", "additional_names", "synonyms"}
        drop_columns = [col for col in required_columns if col not in allowed_missing]
        df = df.dropna(subset=drop_columns)
        df["drug_name"] = df["drug_name"].astype(str).str.strip()
        df = df[df["drug_name"].apply(self._is_valid_drug_name)]
        df["excerpt"] = df["excerpt"].astype(str).str.strip()
        df = df[df["excerpt"] != ""]
        df["nbk_id"] = df["nbk_id"].apply(
            lambda value: str(value).strip() if pd.notna(value) else None
        )
        for column in ("additional_names", "synonyms"):
            df[column] = df[column].apply(
                lambda value: (
                    str(value).strip() if pd.notna(value) and str(value).strip() else None
                )
            )
        df = df.drop_duplicates(subset=["nbk_id", "drug_name"], keep="first")
        return df.reset_index(drop=True)

    # -----------------------------------------------------------------------------
    def _is_valid_drug_name(self, value: str) -> bool:
        normalized = value.strip()
        if len(normalized) < 3 or len(normalized) > 200:
            return False
        if len(normalized.split()) > 8:
            return False
        if not re.fullmatch(r"[A-Za-z0-9\s\-/(),'+\.]+", normalized):
            return False
        return True

    # -----------------------------------------------------------------------------
    def load_from_database(self, table_name: str) -> pd.DataFrame:
        query = f'SELECT * FROM "{table_name}"'
        with database.engine.connect() as connection:
            return pd.read_sql_query(query, connection)

    # -----------------------------------------------------------------------------
    def get_livertox_records(self) -> pd.DataFrame:
        return self.load_from_database("LIVERTOX_MONOGRAPHS")
