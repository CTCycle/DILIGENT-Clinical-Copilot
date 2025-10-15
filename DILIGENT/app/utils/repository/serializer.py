from __future__ import annotations

import re
from typing import Any

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from DILIGENT.app.utils.repository.sqlite import ClinicalSession, database


# [DATA SERIALIZATION]
###############################################################################
LIVERTOX_COLUMNS = [
    "drug_name",
    "ingredient",
    "brand_name",
    "nbk_id",
    "excerpt",
    "synonyms",
    "likelihood_score",
    "last_update",
    "reference_count",
    "year_approved",
    "agent_classification",
    "primary_classification",
    "secondary_classification",
    "include_in_livertox",
    "source_url",
    "source_last_modified",
]


###############################################################################
class DataSerializer:
    def __init__(self) -> None:
        pass

    # -------------------------------------------------------------------------
    def record_clinical_session(self, session_data: dict[str, Any]) -> None:
        session = database.Session()
        try:
            session.add(ClinicalSession(**session_data))
            session.commit()
        except SQLAlchemyError as exc:
            session.rollback()
            raise exc
        finally:
            session.close()

    # -----------------------------------------------------------------------------
    def save_livertox_records(self, records: pd.DataFrame) -> None:
        frame = records.copy()
        frame = frame.reindex(columns=LIVERTOX_COLUMNS)
        frame = frame.where(pd.notnull(frame), None)
        database.save_into_database(frame, "LIVERTOX_DATA")

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
        df = df[df["drug_name"].apply(self.is_valid_drug_name)]
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
    def is_valid_drug_name(self, value: str) -> bool:
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
        frame = database.load_from_database("LIVERTOX_DATA")
        if frame.empty:
            return pd.DataFrame(columns=LIVERTOX_COLUMNS)
        return frame.reindex(columns=LIVERTOX_COLUMNS)

    # -----------------------------------------------------------------------------
    def get_livertox_master_list(self) -> pd.DataFrame:
        frame = database.load_from_database("LIVERTOX_DATA")
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "drug_name",
                    "ingredient",
                    "brand_name",
                    "likelihood_score",
                    "last_update",
                    "reference_count",
                    "year_approved",
                    "agent_classification",
                    "primary_classification",
                    "secondary_classification",
                    "include_in_livertox",
                    "source_url",
                    "source_last_modified",
                ]
            )
        alias_columns = [
            "drug_name",
            "ingredient",
            "brand_name",
            "likelihood_score",
            "last_update",
            "reference_count",
            "year_approved",
            "agent_classification",
            "primary_classification",
            "secondary_classification",
            "include_in_livertox",
            "source_url",
            "source_last_modified",
        ]
        return frame.reindex(columns=alias_columns).dropna(subset=["drug_name"]).reset_index(
            drop=True
        )

    
