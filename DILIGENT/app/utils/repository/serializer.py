from __future__ import annotations

import re
import logging
from typing import Any

import pandas as pd
import pyarrow as pa
from lancedb.table import LanceTable
from sqlalchemy.exc import SQLAlchemyError

from DILIGENT.app.utils.repository.database import ClinicalSession, database
from DILIGENT.app.utils.repository.vectors import LanceVectorRepository


logger = logging.getLogger(__name__)


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

FDA_COLUMNS = [
    "application_number",
    "product_number",
    "sponsor_name",
    "brand_name",
    "active_ingredients",
    "dosage_form",
    "route",
    "marketing_status",
    "application_type",
    "submission_number",
    "submission_type",
    "submission_status",
    "submission_status_date",
    "submission_action_date",
]


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
        frame = frame.reindex(columns=LIVERTOX_COLUMNS)
        frame = frame.where(pd.notnull(frame), None)
        database.save_into_database(frame, "LIVERTOX_DATA")

    # -----------------------------------------------------------------------------
    def save_fda_records(self, records: pd.DataFrame) -> None:
        frame = records.copy()
        frame = frame.reindex(columns=FDA_COLUMNS)
        frame = frame.where(pd.notnull(frame), None)
        database.save_into_database(frame, "FDA_APPROVALS")

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
    def sanitize_fda_records(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        normalized: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            application_number = self.normalize_string(record.get("application_number"))
            if not application_number:
                continue
            sponsor_name = self.normalize_string(record.get("sponsor_name"))
            application_type = self.normalize_string(record.get("application_type"))
            products = record.get("products")
            if not isinstance(products, list):
                continue
            submission = self.get_latest_submission(record.get("submissions"))
            for product in products:
                if not isinstance(product, dict):
                    continue
                product_number = self.normalize_string(product.get("product_number"))
                if not product_number:
                    continue
                brand_name = self.normalize_string(product.get("brand_name"))
                dosage_form = self.normalize_string(product.get("dosage_form"))
                route = self.join_string_values(product.get("route"))
                marketing_status = self.join_string_values(
                    product.get("marketing_status")
                )
                active_ingredients = product.get("active_ingredients")
                ingredients_value = None
                if isinstance(active_ingredients, list):
                    entries = []
                    for item in active_ingredients:
                        if not isinstance(item, dict):
                            continue
                        ingredient_name = self.normalize_string(item.get("name"))
                        strength = self.normalize_string(item.get("strength"))
                        if ingredient_name and strength:
                            entries.append(f"{ingredient_name} ({strength})")
                        elif ingredient_name:
                            entries.append(ingredient_name)
                    if entries:
                        ingredients_value = "; ".join(entries)
                else:
                    ingredients_value = self.normalize_string(active_ingredients)
                normalized.append(
                    {
                        "application_number": application_number,
                        "product_number": product_number,
                        "sponsor_name": sponsor_name,
                        "brand_name": brand_name,
                        "active_ingredients": ingredients_value,
                        "dosage_form": dosage_form,
                        "route": route,
                        "marketing_status": marketing_status,
                        "application_type": application_type,
                        "submission_number": self.normalize_string(
                            (submission or {}).get("submission_number")
                        ),
                        "submission_type": self.normalize_string(
                            (submission or {}).get("submission_type")
                        ),
                        "submission_status": self.normalize_string(
                            (submission or {}).get("submission_status")
                        ),
                        "submission_status_date": self.normalize_string(
                            (submission or {}).get("submission_status_date")
                        ),
                        "submission_action_date": self.normalize_string(
                            (submission or {}).get("submission_action_date")
                        ),
                    }
                )
        if not normalized:
            return pd.DataFrame(columns=FDA_COLUMNS)
        frame = pd.DataFrame(normalized)
        frame = frame.drop_duplicates(
            subset=["application_number", "product_number"], keep="last"
        )
        return frame.reindex(columns=FDA_COLUMNS).reset_index(drop=True)

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
    def join_string_values(self, value: Any) -> str | None:
        if isinstance(value, list):
            entries = [self.normalize_string(item) for item in value]
            entries = [item for item in entries if item]
            if entries:
                return "; ".join(entries)
            return None
        return self.normalize_string(value)

    # -----------------------------------------------------------------------------
    def get_latest_submission(self, submissions: Any) -> dict[str, Any] | None:
        if not isinstance(submissions, list):
            return None
        latest: dict[str, Any] | None = None
        latest_date = None
        for entry in submissions:
            if not isinstance(entry, dict):
                continue
            submission_date = self.parse_submission_date(entry)
            if submission_date is None:
                if latest is None:
                    latest = entry
                continue
            if latest_date is None or submission_date > latest_date:
                latest_date = submission_date
                latest = entry
        return latest

    # -----------------------------------------------------------------------------
    def parse_submission_date(self, entry: dict[str, Any]) -> pd.Timestamp | None:
        date_fields = [
            "submission_status_date",
            "submission_action_date",
            "submission_review_date",
        ]
        for field in date_fields:
            raw_value = entry.get(field)
            if not raw_value:
                continue
            parsed = pd.to_datetime(raw_value, errors="coerce", utc=True)
            if pd.notna(parsed):
                return parsed
        return None

    # -----------------------------------------------------------------------------
    def get_fda_records(self) -> pd.DataFrame:
        frame = database.load_from_database("FDA_APPROVALS")
        if frame.empty:
            return pd.DataFrame(columns=FDA_COLUMNS)
        return frame.reindex(columns=FDA_COLUMNS)

    
