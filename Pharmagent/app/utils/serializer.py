from __future__ import annotations

import re
from typing import Any

import pandas as pd

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
        sanitized = self.sanitize_livertox_records(records)
        sanitized = sanitized.where(pd.notnull(sanitized), None)
        if sanitized.empty:
            database.save_into_database(sanitized, "LIVERTOX_MONOGRAPHS")
            return
        database.save_into_database(sanitized, "LIVERTOX_MONOGRAPHS")

    # -----------------------------------------------------------------------------
    def sanitize_livertox_records(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        required_columns = [
            "nbk_id",
            "drug_name",
            "excerpt",
            "synonyms",
        ]
        if df.empty:
            return pd.DataFrame(columns=required_columns)
        for column in required_columns:
            if column not in df.columns:
                df[column] = None
        df = df[required_columns]
        allowed_missing = {"nbk_id", "synonyms"}
        drop_columns = [col for col in required_columns if col not in allowed_missing]
        df = df.dropna(subset=drop_columns)
        df["drug_name"] = df["drug_name"].astype(str).str.strip()
        df = df[df["drug_name"].apply(self._is_valid_drug_name)]
        df["excerpt"] = df["excerpt"].astype(str).str.strip()
        df = df[df["excerpt"] != ""]
        df["nbk_id"] = df["nbk_id"].apply(
            lambda value: str(value).strip() if pd.notna(value) else None
        )
        df["synonyms"] = df["synonyms"].apply(
            lambda value: (
                str(value).strip()
                if pd.notna(value) and str(value).strip()
                else None
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
    def get_livertox_records(self) -> pd.DataFrame:
        return database.load_from_database("LIVERTOX_MONOGRAPHS")

    # -----------------------------------------------------------------------------
    def save_livertox_master_list(
        self, frame: pd.DataFrame, *, source_url: str, last_modified: str | None
    ) -> None:        
        frame["source_url"] = source_url
        frame["source_last_modified"] = last_modified
        frame = frame.copy()
        if "brand_name" not in frame.columns:
            return

        frame = frame[pd.notnull(frame["brand_name"])].copy()
        frame["brand_name"] = frame["brand_name"].astype(str).str.strip()
        frame = frame[frame["brand_name"] != ""]
        database.save_into_database(frame, "LIVERTOX_MASTER_LIST")

    