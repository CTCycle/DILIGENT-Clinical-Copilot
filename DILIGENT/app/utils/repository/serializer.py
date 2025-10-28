from __future__ import annotations

import json
import logging
import re
from typing import Any

import pandas as pd
import pyarrow as pa
from lancedb.table import LanceTable
from sqlalchemy.exc import SQLAlchemyError

from DILIGENT.app.utils.repository.database import ClinicalSession, database
from DILIGENT.app.logger import logger


# [DATA SERIALIZATION]
###############################################################################
LIVERTOX_COLUMNS = [
    "drug_name",
    "nbk_id",
    "excerpt",
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

FDA_COLUMNS = [
    "report_id",
    "case_version",
    "receipt_date",
    "occur_country",
    "patient_age",
    "patient_age_unit",
    "patient_sex",
    "reaction_terms",
    "all_reactions",
    "suspect_products",
    "suspect_product_count",
    "serious",
    "seriousness_death",
    "seriousness_lifethreatening",
    "seriousness_hospitalization",
    "seriousness_disabling",
    "seriousness_congenital_anom",
    "seriousness_other",
]

HEPATOTOXIC_MEDDRA_TERMS = {
    "hepatotoxicity",
    "drug induced liver injury",
    "drug-induced liver injury",
    "liver injury",
    "hepatic failure",
    "acute hepatic failure",
    "hepatitis cholestatic",
    "cholestasis",
    "liver disorder",
    "liver function test increased",
    "alanine aminotransferase increased",
    "aspartate aminotransferase increased",
    "alkaline phosphatase increased",
    "blood bilirubin increased",
}


###############################################################################
class VectorSerializer:
    def __init__(
        self,
        database_path: str,
        collection_name: str,
        *,
        index_metric: str,
        index_type: str,
        reset_collection: bool,
        vector_column_name: str = "embedding",
    ) -> None:
        self.collection_name = collection_name
        self.index_metric = index_metric
        self.index_type = index_type
        self.reset_collection = reset_collection
        self.vector_column_name = vector_column_name
        self.repository = LanceVectorRepository(database_path)

    ############################################################################
    def save_embeddings(self, data: pa.Table) -> None:
        try:
            if self.repository.table_exists(self.collection_name):
                if self.reset_collection:
                    table = self.repository.create_table(
                        self.collection_name,
                        data=data,
                        mode="overwrite",
                    )
                else:
                    table = self.repository.open_table(self.collection_name)
                    self.repository.add_to_table(table, data)
            else:
                table = self.repository.create_table(
                    self.collection_name,
                    data=data,
                    mode="create",
                )
        except Exception as exc:
            logger.error("Failed to persist vector embeddings: %s", exc)
            raise
        self.build_index(table)

    ############################################################################
    def build_index(self, table: LanceTable) -> None:
        try:
            row_count = self.repository.count_rows(table)
        except Exception as exc:
            logger.warning("Unable to determine table row count: %s", exc)
            return
        if row_count == 0:
            logger.warning("Skipping index creation because the table is empty")
            return
        partitions = self.calculate_partitions(row_count)
        try:
            self.repository.create_index(
                table,
                metric=self.index_metric,
                index_type=self.index_type,
                vector_column_name=self.vector_column_name,
                num_partitions=partitions,
            )
            logger.info(
                "Created %s index with %s partitions using %s metric",
                self.index_type,
                partitions,
                self.index_metric,
            )
        except RuntimeError as exc:
            if self.index_type != "IVF_FLAT":
                logger.warning(
                    "Falling back to IVF_FLAT index after failure: %s",
                    exc,
                )
                try:
                    self.repository.create_index(
                        table,
                        metric=self.index_metric,
                        index_type="IVF_FLAT",
                        vector_column_name=self.vector_column_name,
                        num_partitions=partitions,
                    )
                    logger.info(
                        "Created IVF_FLAT index with %s partitions after fallback",
                        partitions,
                    )
                except RuntimeError as nested_exc:
                    logger.error(
                        "Unable to create fallback IVF_FLAT index: %s",
                        nested_exc,
                    )
            else:
                logger.error("Unable to create vector index: %s", exc)

    ############################################################################
    def calculate_partitions(self, row_count: int) -> int:
        partitions = row_count // 75
        if partitions < 1:
            return 1
        return min(partitions, 256)


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
            logger.error("Failed to record clinical session: %s", exc)
            raise exc
        finally:
            session.close()

    # -----------------------------------------------------------------------------
    def save_livertox_records(self, records: pd.DataFrame) -> None:
        frame = records.copy()
        if "drug_name" in frame.columns:
            frame = frame.drop_duplicates(subset=["drug_name"], keep="first")
        frame = frame.reindex(columns=LIVERTOX_COLUMNS)
        frame = frame.where(pd.notnull(frame), None)
        database.save_into_database(frame, "LIVERTOX_DATA")
   
    # -----------------------------------------------------------------------------
    def upsert_drugs_catalog_records(
        self, records: pd.DataFrame | list[dict[str, Any]]
    ) -> None:
        if isinstance(records, pd.DataFrame):
            frame = records.copy()
        else:
            frame = pd.DataFrame(records)
        frame = frame.reindex(
            columns=[
                "rxcui",
                "raw_name",
                "term_type",
                "name",
                "brand_names",
                "synonyms",
            ]
        )
        if frame.empty:
            return
        frame = frame.where(pd.notnull(frame), None)
        frame["brand_names"] = frame["brand_names"].apply(self.serialize_brand_names)
        frame["synonyms"] = frame["synonyms"].apply(self.serialize_string_list)
        database.upsert_into_database(frame, "DRUGS_CATALOG")

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
                str(value).strip() if pd.notna(value) and str(value).strip() else None
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
    def serialize_string_list(self, value: Any) -> str:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return "[]"
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed_values = [stripped]
            else:
                if isinstance(parsed, list):
                    parsed_values = parsed
                else:
                    parsed_values = [stripped]
        elif isinstance(value, list):
            parsed_values = value
        elif pd.isna(value) or value is None:
            parsed_values = []
        else:
            parsed_values = [value]
        normalized: list[str] = []
        for item in parsed_values:
            normalized_item = self.normalize_list_item(item)
            if normalized_item is not None:
                normalized.append(normalized_item)
        return json.dumps(normalized, ensure_ascii=False)

    # -----------------------------------------------------------------------------
    def serialize_brand_names(self, value: Any) -> str | None:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return self.normalize_list_item(stripped)
            return self.serialize_brand_names(parsed)
        if isinstance(value, list):
            normalized: list[str] = []
            seen: set[str] = set()
            for item in value:
                normalized_item = self.normalize_list_item(item)
                if normalized_item is None:
                    continue
                key = normalized_item.casefold()
                if key in seen:
                    continue
                seen.add(key)
                normalized.append(normalized_item)
            if not normalized:
                return None
            if len(normalized) == 1:
                return normalized[0]
            return ", ".join(normalized)
        if pd.isna(value) or value is None:
            return None
        normalized_item = self.normalize_list_item(value)
        return normalized_item

    # -----------------------------------------------------------------------------
    def normalize_list_item(self, value: Any) -> str | None:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized if normalized else None
        if pd.isna(value) or value is None:
            return None
        normalized = str(value).strip()
        return normalized if normalized else None

    # -----------------------------------------------------------------------------
    def deserialize_string_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            parsed_values = value
        elif isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                normalized_item = self.normalize_list_item(stripped)
                return [normalized_item] if normalized_item is not None else []
            if isinstance(parsed, list):
                parsed_values = parsed
            else:
                normalized_item = self.normalize_list_item(parsed)
                return [normalized_item] if normalized_item is not None else []
        elif pd.isna(value) or value is None:
            return []
        else:
            parsed_values = [value]
        normalized: list[str] = []
        for item in parsed_values:
            normalized_item = self.normalize_list_item(item)
            if normalized_item is not None:
                normalized.append(normalized_item)
        return normalized

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
        available = [column for column in alias_columns if column in frame.columns]
        if not available:
            return pd.DataFrame(columns=["drug_name"])
        return frame.reindex(columns=available).dropna(subset=["drug_name"]).reset_index(
            drop=True
        )

    # -----------------------------------------------------------------------------
    def get_drugs_catalog(self) -> pd.DataFrame:
        frame = database.load_from_database("DRUGS_CATALOG")
        columns = [
            "rxcui",
            "raw_name",
            "term_type",
            "name",
            "brand_names",
            "synonyms",
        ]
        if frame.empty:
            return pd.DataFrame(columns=columns)
        return frame.reindex(columns=columns)

    # -----------------------------------------------------------------------------
    def normalize_string(self, value: Any) -> str | None:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized if normalized else None
        if pd.isna(value):
            return None
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized if normalized else None

    # -----------------------------------------------------------------------------
    def normalize_flag(self, value: Any) -> int | None:
        normalized = self.normalize_string(value)
        if normalized is None:
            return None
        lowered = normalized.lower()
        if lowered in {"1", "y", "yes", "true"}:
            return 1
        if lowered in {"0", "n", "no", "false"}:
            return 0
        if lowered == "2":
            return 0
        try:
            numeric = int(normalized)
        except (TypeError, ValueError):
            return None
        return 1 if numeric != 0 else 0

    # -----------------------------------------------------------------------------
    def normalize_date(self, value: Any) -> str | None:
        normalized = self.normalize_string(value)
        if not normalized:
            return None
        parsed = pd.to_datetime(normalized, errors="coerce", utc=True)
        if pd.isna(parsed):
            return normalized
        return parsed.date().isoformat()

    # -----------------------------------------------------------------------------
    def join_values(self, values: set[str]) -> str | None:
        if not values:
            return None
        return "; ".join(sorted(values))    

    

    
