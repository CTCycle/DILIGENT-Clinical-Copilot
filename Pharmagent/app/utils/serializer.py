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
        sanitized = self.sanitize_livertox_master_list(frame)
        sanitized["source_url"] = source_url
        sanitized["source_last_modified"] = last_modified
        sanitized = sanitized.where(pd.notnull(sanitized), None)
        database.save_into_database(sanitized, "LIVERTOX_MASTER_LIST")

    # -----------------------------------------------------------------------------
    def sanitize_livertox_master_list(self, frame: pd.DataFrame) -> pd.DataFrame:
        required_columns = [
            "ingredient",
            "brand_name",
            "likelihood_score",
            "chapter_title",
            "last_update",
            "reference_count",
            "year_approved",
            "agent_classification",
            "include_in_livertox",
        ]
        if frame.empty:
            return pd.DataFrame(columns=required_columns)
        normalized_map = {
            self._normalize_master_list_column(name): name for name in frame.columns
        }
        column_aliases: dict[str, tuple[str, ...]] = {
            "ingredient": ("ingredient", "generic", "drug", "agent"),
            "brand_name": ("brand", "trade", "brandname"),
            "likelihood_score": ("likelihood", "score"),
            "chapter_title": ("chapter", "title"),
            "last_update": ("lastupdate", "lastupdated", "revision"),
            "reference_count": ("references", "referencecount"),
            "year_approved": ("yearapproved", "approvalyear"),
            "agent_classification": ("agenttype", "agentclass", "classification"),
            "include_in_livertox": ("include", "inclusion", "included"),
        }
        data: dict[str, Any] = {}
        for column, aliases in column_aliases.items():
            source = None
            for alias in aliases:
                normalized_alias = self._normalize_master_list_column(alias)
                if normalized_alias in normalized_map:
                    source = normalized_map[normalized_alias]
                    break
            if source is None:
                data[column] = [None] * len(frame.index)
                continue
            data[column] = frame[source]

        sanitized = pd.DataFrame(data)
        sanitized["ingredient"] = sanitized["ingredient"].astype(str).str.strip()
        sanitized = sanitized[sanitized["ingredient"] != ""]
        sanitized["brand_name"] = sanitized["brand_name"].apply(
            lambda value: str(value).strip() if pd.notna(value) else None
        )
        sanitized["likelihood_score"] = sanitized["likelihood_score"].apply(
            lambda value: str(value).strip() if pd.notna(value) else None
        )
        sanitized["chapter_title"] = sanitized["chapter_title"].apply(
            lambda value: str(value).strip() if pd.notna(value) else None
        )
        sanitized["last_update"] = pd.to_datetime(
            sanitized["last_update"], errors="coerce"
        ).dt.date
        sanitized["reference_count"] = pd.to_numeric(
            sanitized["reference_count"], errors="coerce"
        ).astype("Int64")
        sanitized["year_approved"] = pd.to_numeric(
            sanitized["year_approved"], errors="coerce"
        ).astype("Int64")
        sanitized["agent_classification"] = sanitized[
            "agent_classification"
        ].apply(lambda value: str(value).strip() if pd.notna(value) else None)
        sanitized["include_in_livertox"] = sanitized[
            "include_in_livertox"
        ].apply(lambda value: str(value).strip() if pd.notna(value) else None)
        sanitized = sanitized.drop_duplicates(subset=["ingredient"], keep="first")
        sanitized = sanitized.reset_index(drop=True)
        return sanitized[
            [
                "ingredient",
                "brand_name",
                "likelihood_score",
                "chapter_title",
                "last_update",
                "reference_count",
                "year_approved",
                "agent_classification",
                "include_in_livertox",
            ]
        ]

    # -----------------------------------------------------------------------------
    def _normalize_master_list_column(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())
