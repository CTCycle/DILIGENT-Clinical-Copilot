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
        frame = frame.reindex(columns=LIVERTOX_COLUMNS)
        frame = frame.where(pd.notnull(frame), None)
        database.save_into_database(frame, "LIVERTOX_DATA")

    # -----------------------------------------------------------------------------
    def save_fda_records(self, records: pd.DataFrame) -> None:
        frame = records.copy()
        frame = frame.reindex(columns=FDA_COLUMNS)
        frame = frame.where(pd.notnull(frame), None)
        database.save_into_database(frame, "FDA_ADVERSE_EVENTS")

    # -----------------------------------------------------------------------------
    def upsert_fda_records(self, records: pd.DataFrame) -> None:
        frame = records.copy()
        frame = frame.reindex(columns=FDA_COLUMNS)
        frame = frame.where(pd.notnull(frame), None)
        database.upsert_into_database(frame, "FDA_ADVERSE_EVENTS")

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
            hepatic_terms = self.extract_hepatic_reactions(record)
            if not hepatic_terms:
                continue
            report_id = self.normalize_string(
                record.get("safetyreportid") or record.get("primaryid")
            )
            if not report_id:
                continue
            case_version = self.normalize_string(record.get("safetyreportversion"))
            receipt_date = self.normalize_date(
                record.get("receiptdate")
                or record.get("receivedate")
                or record.get("fulfillexpdate")
            )
            occur_country = self.normalize_string(
                record.get("occurcountry") or record.get("primarysourcecountry")
            )
            patient_data = record.get("patient")
            patient = patient_data if isinstance(patient_data, dict) else {}
            patient_age = None
            patient_age_unit = None
            patient_sex = None
            if patient:
                patient_age = self.normalize_string(patient.get("patientonsetage"))
                patient_age_unit = self.normalize_string(
                    patient.get("patientonsetageunit")
                )
                patient_sex = self.map_patient_sex(patient.get("patientsex"))
            all_reactions = self.collect_all_reactions(record)
            suspect_products = self.extract_suspect_products(record)
            normalized.append(
                {
                    "report_id": report_id,
                    "case_version": case_version or "1",
                    "receipt_date": receipt_date,
                    "occur_country": occur_country,
                    "patient_age": patient_age,
                    "patient_age_unit": patient_age_unit,
                    "patient_sex": patient_sex,
                    "reaction_terms": self.join_values(hepatic_terms),
                    "all_reactions": self.join_values(all_reactions),
                    "suspect_products": self.join_values(suspect_products),
                    "suspect_product_count": len(suspect_products) if suspect_products else None,
                    "serious": self.normalize_flag(record.get("serious")),
                    "seriousness_death": self.normalize_flag(
                        record.get("seriousnessdeath")
                    ),
                    "seriousness_lifethreatening": self.normalize_flag(
                        record.get("seriousnesslifethreatening")
                    ),
                    "seriousness_hospitalization": self.normalize_flag(
                        record.get("seriousnesshospitalization")
                    ),
                    "seriousness_disabling": self.normalize_flag(
                        record.get("seriousnessdisabling")
                    ),
                    "seriousness_congenital_anom": self.normalize_flag(
                        record.get("seriousnesscongenitalanomali")
                    ),
                    "seriousness_other": self.normalize_flag(
                        record.get("seriousnessother")
                    ),
                }
            )
        if not normalized:
            return pd.DataFrame(columns=FDA_COLUMNS)
        frame = pd.DataFrame(normalized)
        frame = frame.drop_duplicates(
            subset=["report_id", "case_version"], keep="last"
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

    # -----------------------------------------------------------------------------
    def collect_all_reactions(self, record: dict[str, Any]) -> set[str]:
        patient = record.get("patient")
        if not isinstance(patient, dict):
            return set()
        reactions = patient.get("reaction")
        if not isinstance(reactions, list):
            return set()
        collected: dict[str, str] = {}
        for entry in reactions:
            if not isinstance(entry, dict):
                continue
            term = self.normalize_string(entry.get("reactionmeddrapt"))
            if not term:
                continue
            key = term.lower()
            if key not in collected:
                collected[key] = term
        return set(collected.values())

    # -----------------------------------------------------------------------------
    def extract_hepatic_reactions(self, record: dict[str, Any]) -> set[str]:
        reactions = self.collect_all_reactions(record)
        hepatic: dict[str, str] = {}
        for term in reactions:
            key = term.lower()
            if key in HEPATOTOXIC_MEDDRA_TERMS and key not in hepatic:
                hepatic[key] = term
        return set(hepatic.values())

    # -----------------------------------------------------------------------------
    def extract_suspect_products(self, record: dict[str, Any]) -> set[str]:
        patient = record.get("patient")
        if not isinstance(patient, dict):
            return set()
        drugs = patient.get("drug")
        if not isinstance(drugs, list):
            return set()
        products: dict[str, str] = {}
        fallback: dict[str, str] = {}
        for entry in drugs:
            if not isinstance(entry, dict):
                continue
            characterization = self.normalize_string(entry.get("drugcharacterization"))
            is_suspect = False
            if characterization:
                if characterization == "1" or characterization.lower() == "suspect":
                    is_suspect = True
            target = products if is_suspect else fallback
            names: dict[str, str] = {}
            product_name = self.normalize_string(entry.get("medicinalproduct"))
            if product_name:
                names[product_name.lower()] = product_name
            openfda = entry.get("openfda")
            if isinstance(openfda, dict):
                for key in (
                    "substance_name",
                    "generic_name",
                    "brand_name",
                ):
                    value = openfda.get(key)
                    if isinstance(value, list):
                        for item in value:
                            candidate = self.normalize_string(item)
                            if candidate and candidate.lower() not in names:
                                names[candidate.lower()] = candidate
                    else:
                        candidate = self.normalize_string(value)
                        if candidate and candidate.lower() not in names:
                            names[candidate.lower()] = candidate
            if not names:
                continue
            for key, value in names.items():
                if key not in target:
                    target[key] = value
        if products:
            return set(products.values())
        return set(fallback.values())

    # -----------------------------------------------------------------------------
    def map_patient_sex(self, value: Any) -> str | None:
        normalized = self.normalize_string(value)
        if not normalized:
            return None
        mapping = {
            "0": "unknown",
            "1": "male",
            "2": "female",
            "3": "unknown",
            "4": "unknown",
            "male": "male",
            "female": "female",
            "unknown": "unknown",
        }
        lowered = normalized.lower()
        return mapping.get(lowered, normalized)

    # -----------------------------------------------------------------------------
    def get_fda_records(self) -> pd.DataFrame:
        frame = database.load_from_database("FDA_ADVERSE_EVENTS")
        if frame.empty:
            return pd.DataFrame(columns=FDA_COLUMNS)
        return frame.reindex(columns=FDA_COLUMNS)

    
