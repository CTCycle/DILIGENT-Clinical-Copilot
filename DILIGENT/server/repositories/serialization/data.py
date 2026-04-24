from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Any, Iterator, cast
from xml.etree import ElementTree

import pandas as pd
from pypdf import PdfReader
from sqlalchemy.engine import Engine
from sqlalchemy import and_, case, delete, exists, func, inspect, or_, select, update
from sqlalchemy.orm import Session, selectinload, sessionmaker

from DILIGENT.server.configurations.startup import server_settings
from DILIGENT.server.domain.documents import Document
from DILIGENT.server.common.constants import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DRUG_NAME_ALLOWED_PATTERN,
    DOCUMENT_SUPPORTED_EXTENSIONS,
    LIVERTOX_COLUMNS,
    LIVERTOX_MASTER_COLUMNS,
    LIVERTOX_OPTIONAL_COLUMNS,
    LIVERTOX_REQUIRED_COLUMNS,
    RXNORM_CATALOG_COLUMNS,
    TEXT_FILE_FALLBACK_ENCODINGS,
)
from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.repositories.database.session import (
    resolve_engine,
    resolve_session_factory,
)
from DILIGENT.server.repositories.queries.drugs import DrugRepositoryQueries
from DILIGENT.server.repositories.schemas.models import (
    ClinicalSession,
    ClinicalSessionDrug,
    ClinicalSessionLab,
    ClinicalSessionResult,
    ClinicalSessionSection,
    Drug,
    DrugAlias,
    DrugDiliAnnotation,
    DrugLabelDocument,
    DrugLabelSection,
    DrugRxnormCode,
    LiverToxMonograph,
    Patient,
)
from DILIGENT.server.repositories.serialization.text_normalization import (
    TextNormalizationVocabularySerializer,
)
from DILIGENT.server.repositories.vectors import LanceVectorDatabase
from DILIGENT.server.services.retrieval.embeddings import EmbeddingGenerator
from DILIGENT.server.services.text.normalization import coerce_text, normalize_drug_name
from DILIGENT.server.services.text.synonyms import parse_synonym_list, split_synonym_variants
from DILIGENT.server.services.text.vocabulary import (
    invalidate_text_normalization_snapshot,
)


class _RepositorySerializationService:
    def __init__(
        self,
        *,
        engine: Engine | None = None,
        session_factory: sessionmaker | None = None,
    ) -> None:
        self.engine = resolve_engine(engine)
        self.session_factory = resolve_session_factory(
            engine=self.engine,
            session_factory=session_factory,
        )

    # -------------------------------------------------------------------------
    def save_clinical_session(self, session_data: dict[str, Any]) -> None:
        if not session_data:
            logger.warning("Skipping clinical session save; payload is empty")
            return
        self.ensure_session_result_table()
        db_session = self.session_factory()
        try:
            persisted_patient = self.persist_patient(db_session, session_data)
            persisted_session = ClinicalSession(
                patient_id=int(persisted_patient.id),
                session_timestamp=self.parse_datetime(
                    session_data.get("session_timestamp")
                ),
                hepatic_pattern=self.normalize_string(session_data.get("hepatic_pattern")),
                text_extraction_model=self.normalize_string(session_data.get("text_extraction_model")),
                clinical_model=self.normalize_string(session_data.get("clinical_model")),
                total_duration=self.to_float(session_data.get("total_duration")),
                session_status=self.normalize_session_status(
                    session_data.get("session_status")
                ),
            )
            db_session.add(persisted_session)
            db_session.flush()
            session_id = int(persisted_session.id)
            self.persist_session_sections(db_session, session_id, session_data)
            self.persist_session_labs(db_session, session_id, session_data)
            self.persist_session_drugs(db_session, session_id, session_data)
            self.persist_session_result_payload(db_session, session_id, session_data)
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    # -----------------------------------------------------------------------------
    def ensure_session_result_table(self) -> None:
        inspector = inspect(self.engine)
        required_tables = (
            Patient.__tablename__,
            ClinicalSession.__tablename__,
            ClinicalSessionResult.__tablename__,
            Drug.__tablename__,
            DrugDiliAnnotation.__tablename__,
            DrugLabelDocument.__tablename__,
            DrugLabelSection.__tablename__,
        )
        missing_tables = [
            table_name for table_name in required_tables if not inspector.has_table(table_name)
        ]
        if missing_tables:
            joined = ", ".join(missing_tables)
            raise RuntimeError(
                f"Database schema mismatch: missing required table(s): {joined}"
            )

        required_columns = {
            Patient.__tablename__: {"image_blob"},
            ClinicalSession.__tablename__: {"patient_id", "session_status"},
            Drug.__tablename__: {"rxnav_last_update"},
        }
        for table_name, columns in required_columns.items():
            existing = {str(item.get("name")) for item in inspector.get_columns(table_name)}
            missing = sorted(columns - existing)
            if missing:
                joined = ", ".join(missing)
                raise RuntimeError(
                    "Database schema mismatch: "
                    f"missing required column(s) in {table_name}: {joined}"
                )

    # -----------------------------------------------------------------------------
    def normalize_session_status(self, value: Any) -> str:
        normalized = self.normalize_string(value)
        if normalized is None:
            return "successful"
        lowered = normalized.casefold()
        if lowered == "failed":
            return "failed"
        return "successful"

    # -----------------------------------------------------------------------------
    def persist_patient(self, db_session: Session, session_data: dict[str, Any]) -> Patient:
        patient = Patient(
            name=self.normalize_string(session_data.get("patient_name")),
            visit_date=self.normalize_date_value(session_data.get("patient_visit_date")),
            anamnesis=self.normalize_string(session_data.get("anamnesis")),
            drugs=self.normalize_string(session_data.get("drugs")),
            laboratory_analysis=self.normalize_string(session_data.get("laboratory_analysis")),
            image_blob=self.decode_patient_image(session_data.get("patient_image_base64")),
        )
        db_session.add(patient)
        db_session.flush()
        return patient

    # -----------------------------------------------------------------------------
    def decode_patient_image(self, value: Any) -> bytes | None:
        normalized = self.normalize_string(value)
        if normalized is None:
            return None
        payload = normalized
        if payload.startswith("data:") and "," in payload:
            payload = payload.split(",", maxsplit=1)[1].strip()
        try:
            return base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            logger.warning("Skipping invalid patient image payload during session save")
            return None

    # -----------------------------------------------------------------------------
    def save_livertox_records(self, records: pd.DataFrame) -> None:
        self.ensure_session_result_table()
        prepared_rows = self.prepare_livertox_rows(records)
        if not prepared_rows:
            return
        db_session = self.session_factory()
        try:
            for row in prepared_rows:
                drug_name = cast(str, row["_drug_name"])
                normalized_name = cast(str, row["_canonical_name_norm"])
                safe_nbk_id = self.normalize_string(row.get("nbk_id"))
                drug = self.ensure_drug(
                    db_session,
                    canonical_name=drug_name,
                    canonical_name_norm=normalized_name,
                    rxnorm_rxcui=None,
                    livertox_nbk_id=None,
                    use_livertox_nbk_lookup=False,
                )
                if safe_nbk_id is not None:
                    self.try_assign_livertox_nbk_id(
                        db_session,
                        drug=drug,
                        livertox_nbk_id=safe_nbk_id,
                    )
                self.upsert_drug_alias(
                    db_session,
                    drug_id=int(drug.id),
                    alias=drug_name,
                    alias_kind="canonical",
                    source="livertox",
                    term_type=None,
                )
                self.persist_livertox_aliases(db_session, int(drug.id), row)
                self.upsert_livertox_monograph(
                    db_session=db_session,
                    drug_id=int(drug.id),
                    row=row,
                )
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    # -----------------------------------------------------------------------------
    def prepare_livertox_rows(self, records: pd.DataFrame) -> list[dict[str, Any]]:
        frame = records.copy()
        if frame.empty:
            return []
        frame = frame.where(pd.notnull(frame), cast(Any, None))
        prepared_rows: list[dict[str, Any]] = []
        for row in frame.to_dict(orient="records"):
            drug_name = self.normalize_string(row.get("drug_name"))
            if drug_name is None:
                continue
            canonical_name_norm = normalize_drug_name(drug_name)
            if not canonical_name_norm:
                continue
            prepared_rows.append(
                {
                    **row,
                    "_drug_name": drug_name,
                    "_canonical_name_norm": canonical_name_norm,
                    "_source_last_modified": self.normalize_string(
                        row.get("source_last_modified")
                    )
                    or "",
                    "_source_url": self.normalize_string(row.get("source_url")) or "",
                    "_last_update": self.normalize_date(row.get("last_update")) or "",
                }
            )
        prepared_rows.sort(key=self.livertox_row_sort_key)
        return prepared_rows

    # -----------------------------------------------------------------------------
    def livertox_row_sort_key(self, row: dict[str, Any]) -> tuple[str, ...]:
        return (
            self.to_sortable_text(row.get("_canonical_name_norm")),
            self.to_sortable_text(row.get("_source_last_modified")),
            self.to_sortable_text(row.get("_source_url")),
            self.to_sortable_text(row.get("_last_update")),
            self.to_sortable_text(row.get("_drug_name")),
        )

    # -----------------------------------------------------------------------------
    def to_sortable_text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).casefold()

    # -----------------------------------------------------------------------------
    def upsert_livertox_monograph(
        self,
        *,
        db_session: Session,
        drug_id: int,
        row: dict[str, Any],
    ) -> None:
        monograph = self.get_monograph_by_drug_id(db_session, drug_id)
        if monograph is None:
            monograph = LiverToxMonograph(drug_id=drug_id)
            db_session.add(monograph)
        monograph.excerpt = self.normalize_string(row.get("excerpt"))
        monograph.likelihood_score = self.normalize_string(row.get("likelihood_score"))
        monograph.last_update = self.normalize_date(row.get("last_update"))
        monograph.reference_count = self.to_int(row.get("reference_count"))
        monograph.year_approved = self.to_int(row.get("year_approved"))
        monograph.agent_classification = self.normalize_string(
            row.get("agent_classification")
        )
        monograph.primary_classification = self.normalize_string(
            row.get("primary_classification")
        )
        monograph.secondary_classification = self.normalize_string(
            row.get("secondary_classification")
        )
        include_flag = self.normalize_flag(row.get("include_in_livertox"))
        monograph.include_in_livertox = None if include_flag is None else include_flag == 1
        monograph.source_url = self.normalize_string(row.get("source_url"))
        monograph.source_last_modified = self.normalize_string(
            row.get("source_last_modified")
        )

    # -----------------------------------------------------------------------------
    def try_assign_livertox_nbk_id(
        self,
        db_session: Session,
        *,
        drug: Drug,
        livertox_nbk_id: str,
    ) -> None:
        normalized = self.normalize_string(livertox_nbk_id)
        if normalized is None:
            return
        existing = (
            db_session.execute(DrugRepositoryQueries.by_livertox_nbk_id(normalized))
            .scalars()
            .first()
        )
        if existing is not None and int(existing.id) != int(drug.id):
            logger.warning(
                "Skipping livertox_nbk_id '%s' for drug_id=%d because it belongs to drug_id=%d",
                normalized,
                int(drug.id),
                int(existing.id),
            )
            return
        current = self.normalize_string(drug.livertox_nbk_id)
        if current is None:
            drug.livertox_nbk_id = normalized
            return
        if current != normalized:
            logger.warning(
                "Skipping livertox_nbk_id update for drug_id=%d (existing='%s', incoming='%s')",
                int(drug.id),
                current,
                normalized,
            )
   
    # -----------------------------------------------------------------------------
    def upsert_drugs_catalog_records(
        self,
        records: pd.DataFrame | list[dict[str, Any]],
        *,
        commit_interval: int | None = None,
        curated_aliases_by_canonical: dict[str, list[tuple[str, str]]] | None = None,
    ) -> None:
        self.ensure_session_result_table()
        prepared_rows = self.prepare_rxnav_rows(records)
        if not prepared_rows:
            return
        effective_commit_interval = self.resolve_commit_interval(commit_interval)
        today_marker = date.today().isoformat()
        db_session = self.session_factory()
        try:
            pending = 0
            for row in prepared_rows:
                rxcui = cast(str, row["_rxcui"])
                raw_name = cast(str | None, row.get("_raw_name"))
                standard_name = cast(str | None, row.get("_standard_name"))
                canonical_name = cast(str, row["_canonical_name"])
                canonical_name_norm = cast(str, row["_canonical_name_norm"])
                term_type = cast(str | None, row.get("_term_type"))
                drug = self.ensure_drug(
                    db_session,
                    canonical_name=canonical_name,
                    canonical_name_norm=canonical_name_norm,
                    rxnorm_rxcui=rxcui,
                    livertox_nbk_id=None,
                    rxnav_last_update=today_marker,
                )
                drug_id = int(drug.id)
                self.upsert_drug_alias(
                    db_session,
                    drug_id=drug_id,
                    alias=canonical_name,
                    alias_kind="canonical",
                    source="derived",
                    term_type=term_type,
                )
                if raw_name is not None:
                    self.upsert_drug_alias(
                        db_session,
                        drug_id=drug_id,
                        alias=raw_name,
                        alias_kind="raw_name",
                        source="rxnorm",
                        term_type=term_type,
                    )
                if standard_name is not None:
                    self.upsert_drug_alias(
                        db_session,
                        drug_id=drug_id,
                        alias=standard_name,
                        alias_kind="standard_name",
                        source="rxnorm",
                        term_type=term_type,
                    )
                for brand in self.extract_text_candidates(row.get("brand_names")):
                    self.upsert_drug_alias(
                        db_session,
                        drug_id=drug_id,
                        alias=brand,
                        alias_kind="brand",
                        source="rxnorm",
                        term_type=term_type,
                    )
                for synonym in self.extract_synonym_candidates(row.get("synonyms")):
                    self.upsert_drug_alias(
                        db_session,
                        drug_id=drug_id,
                        alias=synonym,
                        alias_kind="synonym",
                        source="rxnorm",
                        term_type=term_type,
                    )
                if curated_aliases_by_canonical:
                    curated_entries = curated_aliases_by_canonical.get(
                        canonical_name_norm,
                        [],
                    )
                    for curated_alias, curated_kind in curated_entries:
                        self.upsert_drug_alias(
                            db_session,
                            drug_id=drug_id,
                            alias=curated_alias,
                            alias_kind=curated_kind,
                            source="curated",
                            term_type=term_type,
                        )
                pending += 1
                if pending >= effective_commit_interval:
                    db_session.commit()
                    pending = 0
            if pending:
                db_session.commit()
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    # -----------------------------------------------------------------------------
    def resolve_commit_interval(self, override: int | None) -> int:
        if override is not None:
            return max(int(override), 1)
        return max(int(server_settings.database.insert_commit_interval), 1)

    # -----------------------------------------------------------------------------
    def prepare_rxnav_rows(
        self,
        records: pd.DataFrame | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if isinstance(records, pd.DataFrame):
            frame = records.copy()
        else:
            frame = pd.DataFrame(records)
        frame = frame.reindex(columns=RXNORM_CATALOG_COLUMNS)
        if frame.empty:
            return []
        frame = frame.where(pd.notnull(frame), cast(Any, None))
        prepared_rows: list[dict[str, Any]] = []
        rxcui_to_name_norm: dict[str, str] = {}
        for row in frame.to_dict(orient="records"):
            prepared = self.prepare_rxnav_row(row)
            if prepared is None:
                continue
            rxcui = cast(str, prepared["_rxcui"])
            canonical_name_norm = cast(str, prepared["_canonical_name_norm"])
            mapped = rxcui_to_name_norm.get(rxcui)
            if mapped is not None and mapped != canonical_name_norm:
                raise RuntimeError(
                    f"Conflicting canonical_name_norm values for rxcui '{rxcui}'"
                )
            rxcui_to_name_norm[rxcui] = canonical_name_norm
            prepared_rows.append(prepared)
        prepared_rows.sort(key=self.rxnav_row_sort_key)
        return prepared_rows

    # -----------------------------------------------------------------------------
    def prepare_rxnav_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        rxcui = self.normalize_string(row.get("rxcui"))
        if rxcui is None:
            return None
        raw_name = self.normalize_string(row.get("raw_name"))
        standard_name = self.normalize_string(row.get("name"))
        canonical_name = standard_name or raw_name
        if canonical_name is None:
            return None
        canonical_name_norm = normalize_drug_name(canonical_name)
        if not canonical_name_norm:
            return None
        return {
            **row,
            "_rxcui": rxcui,
            "_raw_name": raw_name,
            "_standard_name": standard_name,
            "_canonical_name": canonical_name,
            "_canonical_name_norm": canonical_name_norm,
            "_term_type": self.normalize_string(row.get("term_type")),
        }

    # -----------------------------------------------------------------------------
    def rxnav_row_sort_key(self, row: dict[str, Any]) -> tuple[str, ...]:
        return (
            self.to_sortable_text(row.get("_rxcui")),
            self.to_sortable_text(row.get("_canonical_name_norm")),
            self.to_sortable_text(row.get("_canonical_name")),
            self.to_sortable_text(row.get("_raw_name")),
            self.to_sortable_text(row.get("_standard_name")),
            self.to_sortable_text(row.get("_term_type")),
        )

    # -----------------------------------------------------------------------------
    def sanitize_livertox_records(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        if df.empty:
            return pd.DataFrame(columns=LIVERTOX_REQUIRED_COLUMNS)
        for column in LIVERTOX_REQUIRED_COLUMNS:
            if column not in df.columns:
                df[column] = None
        df = df[LIVERTOX_REQUIRED_COLUMNS]
        drop_columns = [
            column
            for column in LIVERTOX_REQUIRED_COLUMNS
            if column not in LIVERTOX_OPTIONAL_COLUMNS
        ]
        df = df.dropna(subset=drop_columns)
        df["drug_name"] = df["drug_name"].apply(coerce_text)
        df = df[df["drug_name"].notna()]
        df = df[df["drug_name"].apply(self.is_valid_drug_name)]
        df["excerpt"] = df["excerpt"].apply(coerce_text)
        df = df[df["excerpt"].notna()]
        df["nbk_id"] = df["nbk_id"].apply(coerce_text)
        df["synonyms"] = df["synonyms"].apply(coerce_text)
        df = df.drop_duplicates(subset=["nbk_id", "drug_name"], keep="first")
        return df.reset_index(drop=True)

    # -----------------------------------------------------------------------------
    def is_valid_drug_name(self, value: str) -> bool:
        normalized = value.strip()
        min_length = server_settings.ingestion.drug_name_min_length
        max_length = server_settings.ingestion.drug_name_max_length
        max_tokens = server_settings.ingestion.drug_name_max_tokens
        if len(normalized) < min_length or len(normalized) > max_length:
            return False
        if len(normalized.split()) > max_tokens:
            return False
        if not re.fullmatch(DRUG_NAME_ALLOWED_PATTERN, normalized):
            return False
        return True

    # -----------------------------------------------------------------------------
    def get_livertox_records(self) -> pd.DataFrame:
        self.ensure_session_result_table()
        db_session = self.session_factory()
        try:
            drugs = (
                db_session.execute(
                    select(Drug)
                    .join(Drug.monograph)
                    .options(
                        selectinload(Drug.monograph),
                        selectinload(Drug.aliases),
                    )
                    .order_by(Drug.id.asc())
                )
                .scalars()
                .unique()
                .all()
            )
        finally:
            db_session.close()
        if not drugs:
            return pd.DataFrame(columns=LIVERTOX_COLUMNS)
        records: list[dict[str, Any]] = []
        for drug in drugs:
            monograph = drug.monograph
            if monograph is None:
                continue
            grouped_aliases = self.group_aliases_by_kind(list(drug.aliases))
            records.append(
                {
                    "drug_name": self.normalize_string(drug.canonical_name),
                    "nbk_id": self.normalize_string(drug.livertox_nbk_id),
                    "ingredient": self.join_values(grouped_aliases.get("ingredient", set())),
                    "brand_name": self.join_values(grouped_aliases.get("brand", set())),
                    "synonyms": self.join_values(grouped_aliases.get("synonym", set())),
                    "excerpt": self.normalize_string(monograph.excerpt),
                    "likelihood_score": self.normalize_string(monograph.likelihood_score),
                    "last_update": self.normalize_string(monograph.last_update),
                    "reference_count": monograph.reference_count,
                    "year_approved": monograph.year_approved,
                    "agent_classification": self.normalize_string(
                        monograph.agent_classification
                    ),
                    "primary_classification": self.normalize_string(
                        monograph.primary_classification
                    ),
                    "secondary_classification": self.normalize_string(
                        monograph.secondary_classification
                    ),
                    "include_in_livertox": monograph.include_in_livertox,
                    "source_url": self.normalize_string(monograph.source_url),
                    "source_last_modified": self.normalize_string(
                        monograph.source_last_modified
                    ),
                }
            )
        frame = pd.DataFrame(records)
        if frame.empty:
            return pd.DataFrame(columns=LIVERTOX_COLUMNS)
        frame = frame.where(pd.notnull(frame), cast(Any, None))
        return frame.reindex(columns=LIVERTOX_COLUMNS)

    # -----------------------------------------------------------------------------
    def get_livertox_master_list(self) -> pd.DataFrame:
        frame = self.get_livertox_records()
        if frame.empty:
            return pd.DataFrame(columns=LIVERTOX_MASTER_COLUMNS)
        available = [column for column in LIVERTOX_MASTER_COLUMNS if column in frame.columns]
        if not available:
            return pd.DataFrame(columns=["drug_name"])
        return frame.reindex(columns=available).dropna(subset=["drug_name"]).reset_index(
            drop=True
        )

    # -----------------------------------------------------------------------------
    def get_drugs_catalog(self) -> pd.DataFrame:
        self.ensure_session_result_table()
        db_session = self.session_factory()
        try:
            drugs = (
                db_session.execute(
                    select(Drug)
                    .options(
                        selectinload(Drug.rxnorm_codes),
                        selectinload(Drug.aliases),
                    )
                    .order_by(Drug.id.asc())
                )
                .scalars()
                .unique()
                .all()
            )
        finally:
            db_session.close()
        if not drugs:
            return pd.DataFrame(columns=RXNORM_CATALOG_COLUMNS)
        records: list[dict[str, Any]] = []
        for drug in drugs:
            rxnorm_aliases = [
                alias
                for alias in drug.aliases
                if (self.normalize_string(alias.source) or "").casefold() == "rxnorm"
            ]
            if not rxnorm_aliases:
                continue
            rxcui_values = {
                normalized_rxcui
                for normalized_rxcui in (
                    self.normalize_string(mapping.rxcui) for mapping in drug.rxnorm_codes
                )
                if normalized_rxcui is not None
            }
            primary_rxcui = self.normalize_string(drug.rxnorm_rxcui)
            if primary_rxcui is not None:
                rxcui_values.add(primary_rxcui)
            if not rxcui_values:
                continue
            raw_name = self.first_alias_model_value(rxnorm_aliases, "raw_name")
            standard_name = self.first_alias_model_value(rxnorm_aliases, "standard_name")
            term_type = self.first_alias_model_term_type(rxnorm_aliases)
            brand_names = self.join_values(
                self.alias_model_values_for_kind(rxnorm_aliases, "brand")
            )
            synonyms = sorted(self.alias_model_values_for_kind(rxnorm_aliases, "synonym"))
            for rxcui in sorted(rxcui_values):
                records.append(
                    {
                        "rxcui": rxcui,
                        "raw_name": raw_name or self.normalize_string(drug.canonical_name),
                        "term_type": term_type,
                        "name": standard_name or self.normalize_string(drug.canonical_name),
                        "brand_names": brand_names,
                        "synonyms": json.dumps(synonyms, ensure_ascii=False),
                    }
                )
        if not records:
            return pd.DataFrame(columns=RXNORM_CATALOG_COLUMNS)
        frame = pd.DataFrame(records)
        return frame.reindex(columns=RXNORM_CATALOG_COLUMNS)

    # -----------------------------------------------------------------------------
    def stream_drugs_catalog(
        self, page_size: int | None = None
    ) -> Iterator[pd.DataFrame]:
        chunk_size = (
            server_settings.database.select_page_size
            if page_size is None
            else max(int(page_size), 1)
        )
        frame = self.get_drugs_catalog()
        if frame.empty:
            return
        for start in range(0, len(frame), chunk_size):
            chunk = frame.iloc[start : start + chunk_size]
            if not chunk.empty:
                yield chunk.reset_index(drop=True)

    # -----------------------------------------------------------------------------
    def build_search_pattern(self, search: str | None) -> str | None:
        normalized = self.normalize_string(search)
        if normalized is None:
            return None
        escaped = (
            normalized.casefold()
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        return f"%{escaped}%"

    # -----------------------------------------------------------------------------
    def list_sessions(
        self,
        *,
        search: str | None,
        status_filter: str | None,
        date_mode: str | None,
        filter_date: date | None,
        offset: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        self.ensure_session_result_table()
        safe_offset = max(int(offset), 0)
        safe_limit = max(int(limit), 1)
        conditions: list[Any] = []
        search_pattern = self.build_search_pattern(search)
        if search_pattern is not None:
            section_match = exists(
                select(1).where(
                    ClinicalSessionSection.session_id == ClinicalSession.id,
                    func.lower(func.coalesce(ClinicalSessionSection.content, "")).like(
                        search_pattern,
                        escape="\\",
                    ),
                )
            )
            result_payload_match = exists(
                select(1).where(
                    ClinicalSessionResult.session_id == ClinicalSession.id,
                    func.lower(
                        func.coalesce(ClinicalSessionResult.payload_json, "")
                    ).like(search_pattern, escape="\\"),
                )
            )
            conditions.append(
                or_(
                    func.lower(func.coalesce(Patient.name, "")).like(
                        search_pattern,
                        escape="\\",
                    ),
                    section_match,
                    result_payload_match,
                )
            )
        normalized_status_filter = (
            status_filter.casefold() if isinstance(status_filter, str) else None
        )
        if normalized_status_filter in {"successful", "failed"}:
            conditions.append(
                func.lower(
                    func.coalesce(ClinicalSession.session_status, "successful")
                )
                == normalized_status_filter
            )
        if filter_date is not None and date_mode in {"before", "after", "exact"}:
            day_start = datetime.combine(filter_date, datetime.min.time())
            next_day = day_start + timedelta(days=1)
            if date_mode == "before":
                conditions.append(ClinicalSession.session_timestamp < day_start)
            elif date_mode == "after":
                conditions.append(ClinicalSession.session_timestamp >= next_day)
            elif date_mode == "exact":
                conditions.append(ClinicalSession.session_timestamp >= day_start)
                conditions.append(ClinicalSession.session_timestamp < next_day)

        db_session = self.session_factory()
        try:
            sessions_stmt = select(ClinicalSession, Patient).join(
                Patient,
                ClinicalSession.patient_id == Patient.id,
            )
            count_stmt = (
                select(func.count())
                .select_from(ClinicalSession)
                .join(Patient, ClinicalSession.patient_id == Patient.id)
            )
            if conditions:
                combined = and_(*conditions)
                sessions_stmt = sessions_stmt.where(combined)
                count_stmt = count_stmt.where(combined)
            total_rows = int(db_session.execute(count_stmt).scalar_one())
            rows = (
                db_session.execute(
                    sessions_stmt.order_by(
                        ClinicalSession.session_timestamp.desc(),
                        ClinicalSession.id.desc(),
                    )
                    .offset(safe_offset)
                    .limit(safe_limit)
                )
                .all()
            )
            session_ids = [int(session_row.id) for session_row, _ in rows]
            report_session_ids: set[int] = set()
            timeline_session_ids: set[int] = set()
            if session_ids:
                result_rows = db_session.execute(
                    select(
                        ClinicalSessionResult.session_id,
                        ClinicalSessionResult.payload_json,
                    ).where(ClinicalSessionResult.session_id.in_(session_ids))
                ).all()
                for result_session_id, payload_json in result_rows:
                    parsed_payload = self.parse_session_result_payload(payload_json)
                    if not isinstance(parsed_payload, dict):
                        continue
                    parsed_report = self.normalize_string(parsed_payload.get("report"))
                    if parsed_report is not None:
                        report_session_ids.add(int(result_session_id))
                    parsed_timeline = parsed_payload.get("patient_timeline")
                    if isinstance(parsed_timeline, dict):
                        timeline_session_ids.add(int(result_session_id))
                section_report_rows = db_session.execute(
                    select(ClinicalSessionSection.session_id).where(
                        ClinicalSessionSection.session_id.in_(session_ids),
                        ClinicalSessionSection.section_kind == "final_report",
                    )
                ).all()
                for (section_session_id,) in section_report_rows:
                    report_session_ids.add(int(section_session_id))
            items = [
                {
                    "session_id": int(session_row.id),
                    "patient_name": self.normalize_string(patient_row.name),
                    "session_timestamp": session_row.session_timestamp,
                    "status": self.normalize_session_status(session_row.session_status),
                    "total_duration": self.to_float(session_row.total_duration),
                    "has_report": int(session_row.id) in report_session_ids,
                    "has_timeline": int(session_row.id) in timeline_session_ids,
                    "can_generate_timeline": bool(
                        self.normalize_string(patient_row.anamnesis)
                        or self.normalize_string(patient_row.drugs)
                        or self.normalize_string(patient_row.laboratory_analysis)
                    ),
                }
                for session_row, patient_row in rows
            ]
            return items, total_rows
        finally:
            db_session.close()

    # -----------------------------------------------------------------------------
    def get_session_report(self, session_id: int) -> str | None:
        self.ensure_session_result_table()
        safe_session_id = int(session_id)
        db_session = self.session_factory()
        try:
            payload_json = db_session.execute(
                select(ClinicalSessionResult.payload_json).where(
                    ClinicalSessionResult.session_id == safe_session_id
                )
            ).scalar_one_or_none()
            if payload_json is not None:
                normalized_payload = self.normalize_string(payload_json)
                if normalized_payload is not None:
                    try:
                        parsed = json.loads(normalized_payload)
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict):
                        report = self.normalize_string(parsed.get("report"))
                        if report is not None:
                            return report
                    elif isinstance(parsed, str):
                        report = self.normalize_string(parsed)
                        if report is not None:
                            return report
            section_report = db_session.execute(
                select(ClinicalSessionSection.content).where(
                    ClinicalSessionSection.session_id == safe_session_id,
                    ClinicalSessionSection.section_kind == "final_report",
                )
            ).scalar_one_or_none()
            return self.normalize_string(section_report)
        finally:
            db_session.close()

    # -----------------------------------------------------------------------------
    def parse_session_result_payload(self, payload_json: str | None) -> dict[str, Any] | None:
        normalized_payload = self.normalize_string(payload_json)
        if normalized_payload is None:
            return None
        try:
            parsed = json.loads(normalized_payload)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    # -----------------------------------------------------------------------------
    def get_session_result_payload(self, session_id: int) -> dict[str, Any] | None:
        self.ensure_session_result_table()
        safe_session_id = int(session_id)
        db_session = self.session_factory()
        try:
            payload_json = db_session.execute(
                select(ClinicalSessionResult.payload_json).where(
                    ClinicalSessionResult.session_id == safe_session_id
                )
            ).scalar_one_or_none()
            return self.parse_session_result_payload(payload_json)
        finally:
            db_session.close()

    # -----------------------------------------------------------------------------
    def upsert_session_result_payload(self, session_id: int, payload: dict[str, Any]) -> bool:
        self.ensure_session_result_table()
        safe_session_id = int(session_id)
        serialized_payload = self.serialize_json_payload(payload)
        if serialized_payload is None:
            return False
        db_session = self.session_factory()
        try:
            existing_session = db_session.get(ClinicalSession, safe_session_id)
            if existing_session is None:
                return False
            existing_result = db_session.execute(
                select(ClinicalSessionResult).where(
                    ClinicalSessionResult.session_id == safe_session_id
                )
            ).scalar_one_or_none()
            if existing_result is None:
                db_session.add(
                    ClinicalSessionResult(
                        session_id=safe_session_id,
                        payload_json=serialized_payload,
                    )
                )
            else:
                existing_result.payload_json = serialized_payload
            db_session.commit()
            return True
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    # -----------------------------------------------------------------------------
    def get_session_timeline_source(self, session_id: int) -> dict[str, Any] | None:
        self.ensure_session_result_table()
        safe_session_id = int(session_id)
        db_session = self.session_factory()
        try:
            row = db_session.execute(
                select(ClinicalSession, Patient)
                .join(Patient, ClinicalSession.patient_id == Patient.id)
                .where(ClinicalSession.id == safe_session_id)
            ).first()
            if row is None:
                return None
            session_row, patient_row = row
            payload_json = db_session.execute(
                select(ClinicalSessionResult.payload_json).where(
                    ClinicalSessionResult.session_id == safe_session_id
                )
            ).scalar_one_or_none()
            session_payload = self.parse_session_result_payload(payload_json) or {}
            section_rows = db_session.execute(
                select(ClinicalSessionSection.section_kind, ClinicalSessionSection.content).where(
                    ClinicalSessionSection.session_id == safe_session_id
                )
            ).all()
            sections = {
                str(kind): self.normalize_string(content)
                for kind, content in section_rows
                if self.normalize_string(kind) is not None
            }
            return {
                "session_id": safe_session_id,
                "patient_name": self.normalize_string(patient_row.name),
                "visit_date": patient_row.visit_date.isoformat() if patient_row.visit_date else None,
                "session_timestamp": (
                    session_row.session_timestamp.isoformat()
                    if session_row.session_timestamp
                    else None
                ),
                "anamnesis": self.normalize_string(patient_row.anamnesis),
                "drugs": self.normalize_string(patient_row.drugs),
                "laboratory_analysis": self.normalize_string(patient_row.laboratory_analysis),
                "text_extraction_model": self.normalize_string(session_row.text_extraction_model),
                "clinical_model": self.normalize_string(session_row.clinical_model),
                "sections": sections,
                "session_result_payload": session_payload,
            }
        finally:
            db_session.close()

    # -----------------------------------------------------------------------------
    def delete_session(self, session_id: int) -> bool:
        self.ensure_session_result_table()
        safe_session_id = int(session_id)
        db_session = self.session_factory()
        try:
            existing = db_session.get(ClinicalSession, safe_session_id)
            if existing is None:
                return False
            patient_id = int(existing.patient_id)
            db_session.execute(
                delete(ClinicalSessionResult).where(
                    ClinicalSessionResult.session_id == safe_session_id
                )
            )
            db_session.execute(
                delete(ClinicalSessionSection).where(
                    ClinicalSessionSection.session_id == safe_session_id
                )
            )
            db_session.execute(
                delete(ClinicalSessionLab).where(
                    ClinicalSessionLab.session_id == safe_session_id
                )
            )
            db_session.execute(
                delete(ClinicalSessionDrug).where(
                    ClinicalSessionDrug.session_id == safe_session_id
                )
            )
            db_session.execute(
                delete(ClinicalSession).where(ClinicalSession.id == safe_session_id)
            )
            remaining_patient_sessions = db_session.execute(
                select(func.count())
                .select_from(ClinicalSession)
                .where(ClinicalSession.patient_id == patient_id)
            ).scalar_one()
            if int(remaining_patient_sessions) == 0:
                db_session.execute(delete(Patient).where(Patient.id == patient_id))
            db_session.commit()
            return True
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    # -----------------------------------------------------------------------------
    def list_rxnav_catalog(
        self,
        *,
        search: str | None,
        offset: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        self.ensure_session_result_table()
        safe_offset = max(int(offset), 0)
        safe_limit = max(int(limit), 1)
        search_pattern = self.build_search_pattern(search)
        has_rxnav_data = or_(
            Drug.rxnorm_rxcui.is_not(None),
            exists(
                select(1).where(
                    DrugRxnormCode.drug_id == Drug.id,
                )
            ),
            exists(
                select(1).where(
                    DrugAlias.drug_id == Drug.id,
                    func.lower(func.coalesce(DrugAlias.source, "")) == "rxnorm",
                )
            ),
        )
        conditions: list[Any] = [has_rxnav_data]
        if search_pattern is not None:
            alias_match = exists(
                select(1).where(
                    DrugAlias.drug_id == Drug.id,
                    func.lower(func.coalesce(DrugAlias.alias, "")).like(
                        search_pattern,
                        escape="\\",
                    ),
                )
            )
            conditions.append(
                or_(
                    func.lower(func.coalesce(Drug.canonical_name, "")).like(
                        search_pattern,
                        escape="\\",
                    ),
                    alias_match,
                )
            )
        db_session = self.session_factory()
        try:
            filtered = and_(*conditions)
            count_stmt = select(func.count()).select_from(Drug).where(filtered)
            total_rows = int(db_session.execute(count_stmt).scalar_one())
            rows = db_session.execute(
                select(Drug.id, Drug.canonical_name, Drug.rxnav_last_update)
                .where(filtered)
                .order_by(
                    func.lower(func.coalesce(Drug.canonical_name, "")),
                    Drug.id.asc(),
                )
                .offset(safe_offset)
                .limit(safe_limit)
            ).all()
            items = [
                {
                    "drug_id": int(row.id),
                    "drug_name": row.canonical_name,
                    "last_update": self.normalize_date(row.rxnav_last_update),
                }
                for row in rows
            ]
            return items, total_rows
        finally:
            db_session.close()

    # -----------------------------------------------------------------------------
    def get_rxnav_alias_groups(self, drug_id: int) -> dict[str, Any] | None:
        self.ensure_session_result_table()
        safe_drug_id = int(drug_id)
        db_session = self.session_factory()
        try:
            drug = db_session.get(Drug, safe_drug_id)
            if drug is None:
                return None
            alias_rows = db_session.execute(
                select(DrugAlias.source, DrugAlias.alias, DrugAlias.alias_kind).where(
                    DrugAlias.drug_id == safe_drug_id
                )
            ).all()
            grouped: dict[str, list[dict[str, str]]] = {}
            seen: dict[str, set[str]] = {}
            for source_value, alias_value, alias_kind_value in alias_rows:
                source = self.normalize_string(source_value) or "unknown"
                alias = self.normalize_string(alias_value)
                alias_kind = self.normalize_string(alias_kind_value) or "unknown"
                if alias is None:
                    continue
                dedupe_key = f"{alias.casefold()}::{alias_kind.casefold()}"
                source_seen = seen.setdefault(source, set())
                if dedupe_key in source_seen:
                    continue
                source_seen.add(dedupe_key)
                grouped.setdefault(source, []).append(
                    {"alias": alias, "alias_kind": alias_kind}
                )
            groups = [
                {"source": source, "aliases": aliases}
                for source, aliases in sorted(grouped.items(), key=lambda item: item[0])
            ]
            return {
                "drug_id": safe_drug_id,
                "drug_name": drug.canonical_name,
                "groups": groups,
            }
        finally:
            db_session.close()

    # -----------------------------------------------------------------------------
    def list_livertox_catalog(
        self,
        *,
        search: str | None,
        offset: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        self.ensure_session_result_table()
        safe_offset = max(int(offset), 0)
        safe_limit = max(int(limit), 1)
        search_pattern = self.build_search_pattern(search)
        join_condition = Drug.id == LiverToxMonograph.drug_id
        conditions: list[Any] = []
        if search_pattern is not None:
            alias_match = exists(
                select(1).where(
                    DrugAlias.drug_id == Drug.id,
                    func.lower(func.coalesce(DrugAlias.alias, "")).like(
                        search_pattern,
                        escape="\\",
                    ),
                )
            )
            conditions.append(
                or_(
                    func.lower(func.coalesce(Drug.canonical_name, "")).like(
                        search_pattern,
                        escape="\\",
                    ),
                    func.lower(func.coalesce(LiverToxMonograph.excerpt, "")).like(
                        search_pattern,
                        escape="\\",
                    ),
                    alias_match,
                )
            )
        db_session = self.session_factory()
        try:
            records_stmt = select(
                Drug.id,
                Drug.canonical_name,
                LiverToxMonograph.last_update,
            ).join(LiverToxMonograph, join_condition)
            count_stmt = (
                select(func.count())
                .select_from(Drug)
                .join(LiverToxMonograph, join_condition)
            )
            if conditions:
                combined = and_(*conditions)
                records_stmt = records_stmt.where(combined)
                count_stmt = count_stmt.where(combined)
            total_rows = int(db_session.execute(count_stmt).scalar_one())
            rows = db_session.execute(
                records_stmt.order_by(
                    func.lower(func.coalesce(Drug.canonical_name, "")),
                    Drug.id.asc(),
                )
                .offset(safe_offset)
                .limit(safe_limit)
            ).all()
            items = [
                {
                    "drug_id": int(row.id),
                    "drug_name": row.canonical_name,
                    "last_update": self.normalize_date(row.last_update),
                }
                for row in rows
            ]
            return items, total_rows
        finally:
            db_session.close()

    # -----------------------------------------------------------------------------
    def get_livertox_excerpt(self, drug_id: int) -> dict[str, Any] | None:
        self.ensure_session_result_table()
        safe_drug_id = int(drug_id)
        db_session = self.session_factory()
        try:
            row = db_session.execute(
                select(
                    Drug.id,
                    Drug.canonical_name,
                    LiverToxMonograph.excerpt,
                    LiverToxMonograph.last_update,
                )
                .join(LiverToxMonograph, Drug.id == LiverToxMonograph.drug_id)
                .where(Drug.id == safe_drug_id)
            ).one_or_none()
            if row is None:
                return None
            excerpt = self.normalize_string(row.excerpt)
            if excerpt is None:
                return None
            return {
                "drug_id": int(row.id),
                "drug_name": row.canonical_name,
                "excerpt": excerpt,
                "last_update": self.normalize_date(row.last_update),
            }
        finally:
            db_session.close()

    # -----------------------------------------------------------------------------
    def save_dili_annotations(self, records: pd.DataFrame) -> None:
        self.ensure_session_result_table()
        if records.empty:
            return
        frame = records.copy().where(pd.notnull(records), cast(Any, None))
        db_session = self.session_factory()
        try:
            for row in frame.to_dict(orient="records"):
                source_dataset = self.normalize_string(row.get("source_dataset"))
                source_record_id = self.normalize_string(row.get("source_record_id"))
                if source_dataset is None or source_record_id is None:
                    continue
                existing = db_session.execute(
                    select(DrugDiliAnnotation).where(
                        DrugDiliAnnotation.source_dataset == source_dataset,
                        DrugDiliAnnotation.source_record_id == source_record_id,
                    )
                ).scalar_one_or_none()
                if existing is None:
                    existing = DrugDiliAnnotation(
                        source_dataset=source_dataset,
                        source_record_id=source_record_id,
                    )
                    db_session.add(existing)
                drug_id = self.to_int(row.get("drug_id"))
                existing.drug_id = drug_id
                existing.source_name = self.normalize_string(row.get("source_name"))
                existing.source_name_norm = self.normalize_string(row.get("source_name_norm"))
                existing.classification = self.normalize_string(row.get("classification"))
                existing.severity_class = self.normalize_string(row.get("severity_class"))
                existing.concern_class = self.normalize_string(row.get("concern_class"))
                existing.label_section = self.normalize_string(row.get("label_section"))
                existing.routes = self.normalize_string(row.get("routes"))
                existing.comment = self.normalize_string(row.get("comment"))
                existing.source_url = self.normalize_string(row.get("source_url"))
                existing.source_last_modified = self.normalize_string(
                    row.get("source_last_modified")
                )
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    # -----------------------------------------------------------------------------
    def replace_drug_label_documents(self, records: list[dict[str, Any]]) -> None:
        self.ensure_session_result_table()
        db_session = self.session_factory()
        try:
            db_session.execute(delete(DrugLabelSection))
            db_session.execute(delete(DrugLabelDocument))
            db_session.flush()
            deduped: dict[tuple[str, str, int], dict[str, Any]] = {}
            for record in records:
                drug_id = self.to_int(record.get("drug_id"))
                source = self.normalize_string(record.get("source"))
                set_id = self.normalize_string(record.get("set_id"))
                spl_version = self.to_int(record.get("spl_version"))
                if drug_id is None or source is None or set_id is None or spl_version is None:
                    continue
                key = (source, set_id, spl_version)
                if key in deduped:
                    continue
                deduped[key] = {
                    "drug_id": drug_id,
                    "source": source,
                    "set_id": set_id,
                    "spl_version": spl_version,
                    "title": self.normalize_string(record.get("title")),
                    "labeler": self.normalize_string(record.get("labeler")),
                    "effective_date": self.normalize_date(record.get("effective_date")),
                    "source_url": self.normalize_string(record.get("source_url")),
                    "source_last_modified": self.normalize_string(
                        record.get("source_last_modified")
                    ),
                    "sections": record.get("sections"),
                }
            for record in deduped.values():
                document = DrugLabelDocument(
                    drug_id=int(record["drug_id"]),
                    source=str(record["source"]),
                    set_id=str(record["set_id"]),
                    spl_version=int(record["spl_version"]),
                    title=record["title"],
                    labeler=record["labeler"],
                    effective_date=record["effective_date"],
                    source_url=record["source_url"],
                    source_last_modified=record["source_last_modified"],
                )
                db_session.add(document)
                db_session.flush()
                sections = record.get("sections")
                if not isinstance(sections, list):
                    continue
                for index, section in enumerate(sections):
                    if not isinstance(section, dict):
                        continue
                    section_key = self.normalize_string(section.get("section_key"))
                    text_value = self.normalize_string(section.get("text"))
                    if section_key is None or text_value is None:
                        continue
                    db_session.add(
                        DrugLabelSection(
                            document_id=int(document.id),
                            section_key=section_key,
                            section_title=self.normalize_string(section.get("section_title")),
                            text=text_value,
                            contains_hepatic_keywords=bool(
                                section.get("contains_hepatic_keywords", False)
                            ),
                            display_order=self.to_int(section.get("display_order")) or index,
                        )
                    )
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    # -----------------------------------------------------------------------------
    def list_dili_annotations_catalog(
        self,
        *,
        search: str | None,
        offset: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        self.ensure_session_result_table()
        safe_offset = max(int(offset), 0)
        safe_limit = max(int(limit), 1)
        search_pattern = self.build_search_pattern(search)
        db_session = self.session_factory()
        try:
            dilirank_class = func.max(
                case(
                    (
                        func.lower(DrugDiliAnnotation.source_dataset) == "dilirank",
                        DrugDiliAnnotation.classification,
                    ),
                    else_=None,
                )
            ).label("dilirank_class")
            dilist_class = func.max(
                case(
                    (
                        func.lower(DrugDiliAnnotation.source_dataset) == "dilist",
                        DrugDiliAnnotation.classification,
                    ),
                    else_=None,
                )
            ).label("dilist_class")
            linked_source_count = func.count(DrugDiliAnnotation.id).label("linked_source_count")
            base = (
                select(
                    Drug.id.label("drug_id"),
                    Drug.canonical_name.label("drug_name"),
                    dilirank_class,
                    dilist_class,
                    linked_source_count,
                )
                .join(DrugDiliAnnotation, Drug.id == DrugDiliAnnotation.drug_id)
                .group_by(Drug.id, Drug.canonical_name)
            )
            if search_pattern is not None:
                base = base.where(
                    or_(
                        func.lower(func.coalesce(Drug.canonical_name, "")).like(
                            search_pattern,
                            escape="\\",
                        ),
                        exists(
                            select(1).where(
                                DrugAlias.drug_id == Drug.id,
                                func.lower(func.coalesce(DrugAlias.alias, "")).like(
                                    search_pattern,
                                    escape="\\",
                                ),
                            )
                        ),
                    )
                )
            wrapped = base.subquery()
            total_rows = int(
                db_session.execute(
                    select(func.count()).select_from(wrapped)
                ).scalar_one()
            )
            rows = db_session.execute(
                select(wrapped)
                .order_by(
                    func.lower(func.coalesce(wrapped.c.drug_name, "")),
                    wrapped.c.drug_id.asc(),
                )
                .offset(safe_offset)
                .limit(safe_limit)
            ).all()
            items = [
                {
                    "drug_id": int(row.drug_id),
                    "drug_name": row.drug_name,
                    "dilirank_class": self.normalize_string(row.dilirank_class),
                    "dilist_class": self.normalize_string(row.dilist_class),
                    "linked_source_count": int(row.linked_source_count or 0),
                }
                for row in rows
            ]
            return items, total_rows
        finally:
            db_session.close()

    # -----------------------------------------------------------------------------
    def get_dili_annotation_details(self, drug_id: int) -> dict[str, Any] | None:
        self.ensure_session_result_table()
        safe_drug_id = int(drug_id)
        db_session = self.session_factory()
        try:
            drug = db_session.get(Drug, safe_drug_id)
            if drug is None:
                return None
            rows = db_session.execute(
                select(DrugDiliAnnotation).where(DrugDiliAnnotation.drug_id == safe_drug_id)
            ).scalars().all()
            annotations: list[dict[str, Any]] = []
            for item in sorted(
                rows,
                key=lambda value: (
                    self.to_sortable_text(value.source_dataset),
                    self.to_sortable_text(value.source_record_id),
                ),
            ):
                annotations.append(
                    {
                        "source_dataset": item.source_dataset,
                        "source_record_id": item.source_record_id,
                        "source_name": item.source_name,
                        "source_name_norm": item.source_name_norm,
                        "classification": item.classification,
                        "severity_class": item.severity_class,
                        "concern_class": item.concern_class,
                        "label_section": item.label_section,
                        "routes": item.routes,
                        "comment": item.comment,
                        "source_url": item.source_url,
                        "source_last_modified": item.source_last_modified,
                    }
                )
            return {
                "drug_id": int(drug.id),
                "drug_name": drug.canonical_name,
                "annotations": annotations,
            }
        finally:
            db_session.close()

    # -----------------------------------------------------------------------------
    def list_drug_label_catalog(
        self,
        *,
        search: str | None,
        offset: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        self.ensure_session_result_table()
        safe_offset = max(int(offset), 0)
        safe_limit = max(int(limit), 1)
        search_pattern = self.build_search_pattern(search)
        db_session = self.session_factory()
        try:
            section_count = (
                select(
                    DrugLabelSection.document_id,
                    func.count(DrugLabelSection.id).label("retained_section_count"),
                )
                .group_by(DrugLabelSection.document_id)
                .subquery()
            )
            stmt = (
                select(
                    Drug.id.label("drug_id"),
                    Drug.canonical_name.label("drug_name"),
                    DrugLabelDocument.source,
                    DrugLabelDocument.effective_date,
                    func.coalesce(section_count.c.retained_section_count, 0).label(
                        "retained_section_count"
                    ),
                )
                .join(DrugLabelDocument, DrugLabelDocument.drug_id == Drug.id)
                .outerjoin(section_count, section_count.c.document_id == DrugLabelDocument.id)
            )
            if search_pattern is not None:
                stmt = stmt.where(
                    or_(
                        func.lower(func.coalesce(Drug.canonical_name, "")).like(
                            search_pattern,
                            escape="\\",
                        ),
                        exists(
                            select(1).where(
                                DrugAlias.drug_id == Drug.id,
                                func.lower(func.coalesce(DrugAlias.alias, "")).like(
                                    search_pattern,
                                    escape="\\",
                                ),
                            )
                        ),
                    )
                )
            count_stmt = select(func.count()).select_from(stmt.subquery())
            total_rows = int(db_session.execute(count_stmt).scalar_one())
            rows = db_session.execute(
                stmt.order_by(
                    func.lower(func.coalesce(Drug.canonical_name, "")),
                    Drug.id.asc(),
                )
                .offset(safe_offset)
                .limit(safe_limit)
            ).all()
            items = [
                {
                    "drug_id": int(row.drug_id),
                    "drug_name": row.drug_name,
                    "source": row.source,
                    "effective_date": self.normalize_date(row.effective_date),
                    "retained_section_count": int(row.retained_section_count or 0),
                }
                for row in rows
            ]
            return items, total_rows
        finally:
            db_session.close()

    # -----------------------------------------------------------------------------
    def get_drug_label_sections(self, drug_id: int) -> dict[str, Any] | None:
        self.ensure_session_result_table()
        safe_drug_id = int(drug_id)
        db_session = self.session_factory()
        try:
            drug = db_session.get(Drug, safe_drug_id)
            if drug is None:
                return None
            document = db_session.execute(
                select(DrugLabelDocument)
                .where(DrugLabelDocument.drug_id == safe_drug_id)
                .order_by(
                    func.lower(func.coalesce(DrugLabelDocument.source, "")),
                    DrugLabelDocument.effective_date.desc(),
                    DrugLabelDocument.spl_version.desc(),
                    DrugLabelDocument.set_id.asc(),
                )
            ).scalars().first()
            if document is None:
                return None
            sections = db_session.execute(
                select(DrugLabelSection).where(DrugLabelSection.document_id == int(document.id))
            ).scalars().all()
            section_rows = [
                {
                    "section_key": item.section_key,
                    "section_title": item.section_title,
                    "text": item.text,
                    "contains_hepatic_keywords": bool(item.contains_hepatic_keywords),
                    "display_order": int(item.display_order),
                }
                for item in sorted(sections, key=lambda value: int(value.display_order))
            ]
            return {
                "drug_id": int(drug.id),
                "drug_name": drug.canonical_name,
                "source": document.source,
                "set_id": document.set_id,
                "spl_version": int(document.spl_version),
                "effective_date": self.normalize_date(document.effective_date),
                "sections": section_rows,
            }
        finally:
            db_session.close()

    # -----------------------------------------------------------------------------
    def get_drug_knowledge_bundle(self, drug_id: int) -> dict[str, Any]:
        self.ensure_session_result_table()
        safe_drug_id = int(drug_id)
        db_session = self.session_factory()
        try:
            drug = db_session.get(Drug, safe_drug_id)
            if drug is None:
                return {
                    "drug_id": safe_drug_id,
                    "drug_name": None,
                    "livertox_excerpt": None,
                    "dili_annotations": [],
                    "label_sections": [],
                }
            livertox_excerpt = db_session.execute(
                select(LiverToxMonograph.excerpt).where(LiverToxMonograph.drug_id == safe_drug_id)
            ).scalar_one_or_none()
            annotations = db_session.execute(
                select(DrugDiliAnnotation).where(DrugDiliAnnotation.drug_id == safe_drug_id)
            ).scalars().all()
            documents = db_session.execute(
                select(DrugLabelDocument).where(DrugLabelDocument.drug_id == safe_drug_id)
            ).scalars().all()
            document_ids = [int(item.id) for item in documents]
            section_rows: list[DrugLabelSection] = []
            if document_ids:
                section_rows = db_session.execute(
                    select(DrugLabelSection).where(DrugLabelSection.document_id.in_(document_ids))
                ).scalars().all()
            document_by_id = {int(item.id): item for item in documents}
            return {
                "drug_id": int(drug.id),
                "drug_name": drug.canonical_name,
                "livertox_excerpt": self.normalize_string(livertox_excerpt),
                "dili_annotations": [
                    {
                        "source_dataset": item.source_dataset,
                        "classification": item.classification,
                        "severity_class": item.severity_class,
                        "concern_class": item.concern_class,
                        "comment": item.comment,
                    }
                    for item in sorted(
                        annotations,
                        key=lambda value: (
                            self.to_sortable_text(value.source_dataset),
                            self.to_sortable_text(value.source_record_id),
                        ),
                    )
                ],
                "label_sections": [
                    {
                        "source": document_by_id[int(item.document_id)].source
                        if int(item.document_id) in document_by_id
                        else "dailymed",
                        "section_key": item.section_key,
                        "section_title": item.section_title,
                        "text": item.text,
                        "contains_hepatic_keywords": bool(item.contains_hepatic_keywords),
                        "display_order": int(item.display_order),
                    }
                    for item in sorted(
                        section_rows,
                        key=lambda value: (
                            self.to_sortable_text(
                                document_by_id[int(value.document_id)].source
                                if int(value.document_id) in document_by_id
                                else "dailymed"
                            ),
                            int(value.display_order),
                        ),
                    )
                ],
            }
        finally:
            db_session.close()

    # -----------------------------------------------------------------------------
    def delete_drug_with_cleanup(self, drug_id: int) -> bool:
        self.ensure_session_result_table()
        safe_drug_id = int(drug_id)
        db_session = self.session_factory()
        try:
            existing = db_session.get(Drug, safe_drug_id)
            if existing is None:
                return False
            db_session.execute(
                update(ClinicalSessionDrug)
                .where(ClinicalSessionDrug.drug_id == safe_drug_id)
                .values(drug_id=None)
            )
            db_session.execute(delete(DrugAlias).where(DrugAlias.drug_id == safe_drug_id))
            db_session.execute(
                delete(DrugRxnormCode).where(DrugRxnormCode.drug_id == safe_drug_id)
            )
            db_session.execute(
                delete(LiverToxMonograph).where(LiverToxMonograph.drug_id == safe_drug_id)
            )
            db_session.execute(
                delete(DrugDiliAnnotation).where(DrugDiliAnnotation.drug_id == safe_drug_id)
            )
            label_document_ids = db_session.execute(
                select(DrugLabelDocument.id).where(DrugLabelDocument.drug_id == safe_drug_id)
            ).scalars().all()
            if label_document_ids:
                db_session.execute(
                    delete(DrugLabelSection).where(
                        DrugLabelSection.document_id.in_(list(label_document_ids))
                    )
                )
            db_session.execute(
                delete(DrugLabelDocument).where(DrugLabelDocument.drug_id == safe_drug_id)
            )
            db_session.execute(delete(Drug).where(Drug.id == safe_drug_id))
            db_session.commit()
            return True
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    # -----------------------------------------------------------------------------
    def normalize_string(self, value: Any) -> str | None:
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return None
            if normalized.lower() in {"not available", "nan", "none", "<na>", "nat"}:
                return None
            return normalized
        if pd.isna(value):
            return None
        if value is None:
            return None
        normalized = str(value).strip()
        if not normalized:
            return None
        if normalized.lower() in {"not available", "nan", "none", "<na>", "nat"}:
            return None
        return normalized

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
        normalized_date = self.normalize_date_value(value)
        if normalized_date is None:
            normalized = self.normalize_string(value)
            return normalized or None
        return normalized_date.isoformat()

    # -----------------------------------------------------------------------------
    def normalize_date_value(self, value: Any) -> date | None:
        normalized = self.normalize_string(value)
        if not normalized:
            return None
        parsed: Any
        if re.fullmatch(r"[+-]?\d+", normalized):
            digits = normalized[1:] if normalized.startswith(("+", "-")) else normalized
            if len(digits) == 8:
                parsed = pd.to_datetime(
                    normalized, errors="coerce", format="%Y%m%d", utc=True
                )
            else:
                inferred_unit = {
                    10: "s",
                    13: "ms",
                    16: "us",
                    19: "ns",
                }.get(len(digits))
                if inferred_unit is None:
                    return normalized
                parsed = pd.to_datetime(
                    int(normalized),
                    errors="coerce",
                    utc=True,
                    unit=inferred_unit,
                )
        else:
            parsed = pd.to_datetime(normalized, errors="coerce", utc=True)
        if pd.isna(parsed):
            return None
        return parsed.date()

    # -----------------------------------------------------------------------------
    def join_values(self, values: set[str]) -> str | None:
        if not values:
            return None
        return "; ".join(sorted(values))

    # -----------------------------------------------------------------------------
    def to_int(self, value: Any) -> int | None:
        normalized = self.normalize_string(value)
        if normalized is None:
            return None
        try:
            return int(float(normalized))
        except (TypeError, ValueError):
            return None

    # -----------------------------------------------------------------------------
    def to_float(self, value: Any) -> float | None:
        normalized = self.normalize_string(value)
        if normalized is None:
            return None
        try:
            return float(normalized)
        except (TypeError, ValueError):
            return None

    # -----------------------------------------------------------------------------
    def parse_datetime(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        if isinstance(parsed, pd.Timestamp):
            return parsed.to_pydatetime()
        return parsed

    # -----------------------------------------------------------------------------
    def persist_session_sections(
        self, db_session: Session, session_id: int, session_data: dict[str, Any]
    ) -> None:
        issues_content: str | None = None
        issues_raw = session_data.get("issues")
        if isinstance(issues_raw, (list, dict)):
            issues_content = json.dumps(issues_raw, ensure_ascii=False)
        elif isinstance(issues_raw, str):
            issues_content = self.normalize_string(issues_raw)
        payload = {
            "anamnesis": session_data.get("anamnesis"),
            "drugs": session_data.get("drugs"),
            "laboratory_analysis": session_data.get("laboratory_analysis"),
            "final_report": session_data.get("final_report"),
            "issues": issues_content,
        }
        for section_kind, value in payload.items():
            content = self.normalize_string(value)
            if content is None:
                continue
            db_session.add(
                ClinicalSessionSection(
                    session_id=session_id,
                    section_kind=section_kind,
                    content=content,
                )
            )

    # -----------------------------------------------------------------------------
    def persist_session_labs(
        self, db_session: Session, session_id: int, session_data: dict[str, Any]
    ) -> None:
        result_payload = session_data.get("session_result_payload")
        if not isinstance(result_payload, dict):
            return
        timeline_raw = result_payload.get("lab_timeline")
        if not isinstance(timeline_raw, list):
            return
        persisted_codes = {
            "ALT": "alt",
            "AST": "ast",
            "ALP": "alp",
            "TBIL": "tbil",
            "DBIL": "dbil",
            "GGT": "ggt",
            "INR": "inr",
            "ALB": "albumin",
        }
        # DB schema enforces one lab row per (session_id, lab_code), so collapse
        # repeated timeline points of the same marker into a single persisted row.
        rows_by_lab_code: dict[str, tuple[str | None, str | None]] = {}
        for item in timeline_raw:
            if not isinstance(item, dict):
                continue
            marker_name = self.normalize_string(item.get("marker_name"))
            if marker_name is None:
                continue
            lab_code = persisted_codes.get(marker_name.upper())
            if lab_code is None:
                continue
            value_raw = self.normalize_string(item.get("value")) or self.normalize_string(item.get("value_text"))
            upper_limit_raw = self.normalize_string(item.get("upper_limit_normal")) or self.normalize_string(item.get("upper_limit_text"))
            if value_raw is None and upper_limit_raw is None:
                continue
            existing_value_raw, existing_upper_limit_raw = rows_by_lab_code.get(
                lab_code, (None, None)
            )
            merged_value_raw = existing_value_raw or value_raw
            merged_upper_limit_raw = existing_upper_limit_raw or upper_limit_raw
            rows_by_lab_code[lab_code] = (merged_value_raw, merged_upper_limit_raw)
        for lab_code, (value_raw, upper_limit_raw) in rows_by_lab_code.items():
            db_session.add(
                ClinicalSessionLab(
                    session_id=session_id,
                    lab_code=lab_code,
                    value_raw=value_raw,
                    upper_limit_raw=upper_limit_raw,
                )
            )

    # -----------------------------------------------------------------------------
    def persist_session_drugs(
        self, db_session: Session, session_id: int, session_data: dict[str, Any]
    ) -> None:
        payload = session_data.get("matched_drugs")
        records: list[dict[str, Any]] = []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    records.append(item)
                elif isinstance(item, str):
                    records.append({"raw_drug_name": item})
        if not records:
            detected_drugs = session_data.get("detected_drugs")
            if isinstance(detected_drugs, list):
                for item in detected_drugs:
                    if isinstance(item, str):
                        records.append({"raw_drug_name": item})
        seen: set[str] = set()
        vocabulary_changed = False
        vocabulary_serializer = TextNormalizationVocabularySerializer(
            engine=self.engine,
            session_factory=self.session_factory,
        )
        for item in records:
            raw_drug_name = self.normalize_string(
                item.get("raw_drug_name") or item.get("name")
            )
            if raw_drug_name is None:
                continue
            raw_drug_name_norm = normalize_drug_name(raw_drug_name)
            if not raw_drug_name_norm or raw_drug_name_norm in seen:
                continue
            seen.add(raw_drug_name_norm)
            matched_drug_name = self.normalize_string(item.get("matched_drug_name"))
            rxcui = self.normalize_string(item.get("rxcui"))
            nbk_id = self.normalize_string(item.get("nbk_id"))
            resolved_drug_id = self.resolve_drug_id(
                db_session,
                matched_drug_name=matched_drug_name,
                rxcui=rxcui,
                nbk_id=nbk_id,
            )
            match_reason = self.normalize_string(item.get("match_reason"))
            match_confidence = self.to_float(item.get("match_confidence"))
            should_promote_observed_alias = (
                resolved_drug_id is not None
                and match_reason == "exact_canonical"
                and match_confidence == 1.0
            )
            if should_promote_observed_alias:
                self.upsert_drug_alias(
                    db_session,
                    drug_id=resolved_drug_id,
                    alias=raw_drug_name,
                    alias_kind="observed_query",
                    source="session",
                    term_type=None,
                )
            else:
                observation_category = (
                    "observed_unresolved_query"
                    if resolved_drug_id is None
                    else "observed_unpromoted_query"
                )
                vocabulary_serializer.upsert_term(
                    db_session,
                    category=observation_category,
                    term=raw_drug_name,
                    replacement=None,
                    source="session",
                    increment=True,
                )
                vocabulary_changed = True
            notes = item.get("match_notes")
            if isinstance(notes, (list, dict)):
                notes_value = json.dumps(notes, ensure_ascii=False)
            else:
                notes_value = self.normalize_string(notes)
            db_session.add(
                ClinicalSessionDrug(
                    session_id=session_id,
                    raw_drug_name=raw_drug_name,
                    raw_drug_name_norm=raw_drug_name_norm,
                    drug_id=resolved_drug_id,
                    match_confidence=match_confidence,
                    match_reason=match_reason,
                    match_notes=notes_value,
                )
            )
        if vocabulary_changed:
            invalidate_text_normalization_snapshot()

    # -----------------------------------------------------------------------------
    def persist_session_result_payload(
        self, db_session: Session, session_id: int, session_data: dict[str, Any]
    ) -> None:
        payload = session_data.get("session_result_payload")
        serialized_payload = self.serialize_json_payload(payload)
        if serialized_payload is None:
            return
        db_session.add(
            ClinicalSessionResult(
                session_id=session_id,
                payload_json=serialized_payload,
            )
        )

    # -----------------------------------------------------------------------------
    def serialize_json_payload(self, payload: Any) -> str | None:
        if payload is None:
            return None
        if isinstance(payload, str):
            return self.normalize_string(payload)
        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return self.normalize_string(payload)

    # -----------------------------------------------------------------------------
    def resolve_drug_id(
        self,
        db_session: Session,
        *,
        matched_drug_name: str | None,
        rxcui: str | None,
        nbk_id: str | None,
    ) -> int | None:
        drug = self.get_drug_by_rxcui(db_session, rxcui)
        if drug is not None:
            return int(drug.id)
        drug = self.get_drug_by_livertox_nbk_id(db_session, nbk_id)
        if drug is not None:
            return int(drug.id)
        if matched_drug_name is None:
            return None
        normalized_name = normalize_drug_name(matched_drug_name)
        if not normalized_name:
            return None
        drug = self.get_drug_by_canonical_name_norm(db_session, normalized_name)
        if drug is not None:
            return int(drug.id)
        alias = self.get_drug_alias_by_norm(db_session, normalized_name)
        if alias is None:
            return None
        return int(alias.drug_id)

    # -----------------------------------------------------------------------------
    def ensure_drug(
        self,
        db_session: Session,
        *,
        canonical_name: str,
        canonical_name_norm: str,
        rxnorm_rxcui: str | None,
        livertox_nbk_id: str | None,
        rxnav_last_update: str | None = None,
        use_livertox_nbk_lookup: bool = True,
    ) -> Drug:
        candidate_by_rxcui = self.get_drug_by_rxcui(db_session, rxnorm_rxcui)
        candidate_by_nbk = (
            self.get_drug_by_livertox_nbk_id(db_session, livertox_nbk_id)
            if use_livertox_nbk_lookup
            else None
        )
        candidate_by_name = self.get_drug_by_canonical_name_norm(
            db_session,
            canonical_name_norm,
        )
        resolved_ids: set[int] = set()
        for candidate in (candidate_by_rxcui, candidate_by_nbk, candidate_by_name):
            if candidate is not None:
                resolved_ids.add(int(candidate.id))
        if len(resolved_ids) > 1:
            raise RuntimeError(
                "Conflicting drug selectors resolved to different rows "
                f"(canonical_name_norm='{canonical_name_norm}', "
                f"rxnorm_rxcui='{rxnorm_rxcui}', livertox_nbk_id='{livertox_nbk_id}')"
            )
        candidate = candidate_by_rxcui or candidate_by_nbk or candidate_by_name
        if candidate is None:
            candidate = Drug(
                canonical_name=canonical_name,
                canonical_name_norm=canonical_name_norm,
                rxnorm_rxcui=rxnorm_rxcui,
                livertox_nbk_id=livertox_nbk_id if use_livertox_nbk_lookup else None,
                rxnav_last_update=self.normalize_date(rxnav_last_update),
            )
            db_session.add(candidate)
            db_session.flush()
            self.upsert_drug_rxcui(
                db_session,
                drug_id=int(candidate.id),
                rxcui=rxnorm_rxcui,
            )
            return candidate
        self.assign_primary_rxcui_if_missing(
            drug=candidate,
            incoming_rxcui=rxnorm_rxcui,
        )
        self.upsert_drug_rxcui(
            db_session,
            drug_id=int(candidate.id),
            rxcui=rxnorm_rxcui,
        )
        if use_livertox_nbk_lookup:
            self.assign_identifier_if_consistent(
                drug=candidate,
                field_name="livertox_nbk_id",
                incoming_value=livertox_nbk_id,
            )
        normalized_rxnav_last_update = self.normalize_date(rxnav_last_update)
        if normalized_rxnav_last_update is not None:
            candidate.rxnav_last_update = normalized_rxnav_last_update
        return candidate

    # -----------------------------------------------------------------------------
    def assign_primary_rxcui_if_missing(
        self,
        *,
        drug: Drug,
        incoming_rxcui: str | None,
    ) -> None:
        if incoming_rxcui is None:
            return
        current_rxcui = self.normalize_string(drug.rxnorm_rxcui)
        if current_rxcui is None:
            drug.rxnorm_rxcui = incoming_rxcui

    # -----------------------------------------------------------------------------
    def assign_identifier_if_consistent(
        self,
        *,
        drug: Drug,
        field_name: str,
        incoming_value: str | None,
    ) -> None:
        if incoming_value is None:
            return
        current_value = self.normalize_string(getattr(drug, field_name))
        if current_value is not None and current_value != incoming_value:
            raise RuntimeError(
                f"Conflicting {field_name} for existing drug row "
                f"(drug_id={int(drug.id)}, existing='{current_value}', incoming='{incoming_value}')"
            )
        if current_value is None:
            setattr(drug, field_name, incoming_value)

    # -----------------------------------------------------------------------------
    def upsert_drug_rxcui(
        self,
        db_session: Session,
        *,
        drug_id: int,
        rxcui: str | None,
    ) -> None:
        normalized_rxcui = self.normalize_string(rxcui)
        if normalized_rxcui is None:
            return
        existing = (
            db_session.execute(DrugRepositoryQueries.drug_rxcui_mapping(normalized_rxcui))
            .scalars()
            .first()
        )
        if existing is None:
            db_session.add(DrugRxnormCode(drug_id=drug_id, rxcui=normalized_rxcui))
            return
        if int(existing.drug_id) != int(drug_id):
            raise RuntimeError(
                "Conflicting rxcui mapping for existing drug row "
                f"(rxcui='{normalized_rxcui}', existing_drug_id={int(existing.drug_id)}, incoming_drug_id={drug_id})"
            )

    # -----------------------------------------------------------------------------
    def get_drug_by_rxcui(
        self,
        db_session: Session,
        rxcui: str | None,
    ) -> Drug | None:
        normalized_rxcui = self.normalize_string(rxcui)
        if normalized_rxcui is None:
            return None
        mapped = (
            db_session.execute(DrugRepositoryQueries.drug_by_joined_rxcui(normalized_rxcui))
            .scalars()
            .first()
        )
        if mapped is not None:
            return mapped
        return (
            db_session.execute(DrugRepositoryQueries.drug_by_rxnorm_rxcui(normalized_rxcui))
            .scalars()
            .first()
        )

    # -----------------------------------------------------------------------------
    def get_drug_by_livertox_nbk_id(
        self,
        db_session: Session,
        livertox_nbk_id: str | None,
    ) -> Drug | None:
        if livertox_nbk_id is None:
            return None
        return (
            db_session.execute(DrugRepositoryQueries.by_livertox_nbk_id(livertox_nbk_id))
            .scalars()
            .first()
        )

    # -----------------------------------------------------------------------------
    def get_drug_by_canonical_name_norm(
        self,
        db_session: Session,
        canonical_name_norm: str | None,
    ) -> Drug | None:
        if canonical_name_norm is None:
            return None
        return (
            db_session.execute(
                DrugRepositoryQueries.drug_by_canonical_name_norm(canonical_name_norm)
            )
            .scalars()
            .first()
        )

    # -----------------------------------------------------------------------------
    def get_drug_alias_by_norm(
        self,
        db_session: Session,
        alias_norm: str | None,
    ) -> DrugAlias | None:
        if alias_norm is None:
            return None
        return (
            db_session.execute(DrugRepositoryQueries.alias_by_norm(alias_norm))
            .scalars()
            .first()
        )

    # -----------------------------------------------------------------------------
    def get_monograph_by_drug_id(
        self,
        db_session: Session,
        drug_id: int,
    ) -> LiverToxMonograph | None:
        return (
            db_session.execute(DrugRepositoryQueries.monograph_by_drug_id(drug_id))
            .scalars()
            .first()
        )

    # -----------------------------------------------------------------------------
    def upsert_drug_alias(
        self,
        db_session: Session,
        *,
        drug_id: int,
        alias: str,
        alias_kind: str,
        source: str,
        term_type: str | None,
    ) -> None:
        clean_alias = self.normalize_string(alias)
        if clean_alias is None:
            return
        alias_norm = normalize_drug_name(clean_alias)
        if not alias_norm:
            return
        existing = (
            db_session.execute(
                DrugRepositoryQueries.alias_for_drug(
                    drug_id=drug_id,
                    alias_norm=alias_norm,
                    alias_kind=alias_kind,
                    source=source,
                )
            )
            .scalars()
            .first()
        )
        if existing is None:
            db_session.add(
                DrugAlias(
                    drug_id=drug_id,
                    alias=clean_alias,
                    alias_norm=alias_norm,
                    alias_kind=alias_kind,
                    source=source,
                    term_type=term_type,
                )
            )
            return
        if existing.term_type is None and term_type is not None:
            existing.term_type = term_type

    # -----------------------------------------------------------------------------
    def persist_livertox_aliases(
        self, db_session: Session, drug_id: int, row: dict[str, Any]
    ) -> None:
        for alias in self.extract_text_candidates(row.get("ingredient")):
            self.upsert_drug_alias(
                db_session,
                drug_id=drug_id,
                alias=alias,
                alias_kind="ingredient",
                source="livertox",
                term_type=None,
            )
        for alias in self.extract_text_candidates(row.get("brand_name")):
            self.upsert_drug_alias(
                db_session,
                drug_id=drug_id,
                alias=alias,
                alias_kind="brand",
                source="livertox",
                term_type=None,
            )
        for alias in self.extract_synonym_candidates(row.get("synonyms")):
            self.upsert_drug_alias(
                db_session,
                drug_id=drug_id,
                alias=alias,
                alias_kind="synonym",
                source="livertox",
                term_type=None,
            )

    # -----------------------------------------------------------------------------
    def extract_text_candidates(self, value: Any) -> list[str]:
        if value is None:
            return []
        collected: list[str] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    collected.extend(split_synonym_variants(item))
            return self.unique_text(collected)
        text_value = self.normalize_string(value)
        if text_value is None:
            return []
        collected.extend(split_synonym_variants(text_value))
        return self.unique_text(collected)

    # -----------------------------------------------------------------------------
    def extract_synonym_candidates(self, value: Any) -> list[str]:
        collected: list[str] = []
        for item in parse_synonym_list(value):
            collected.extend(split_synonym_variants(item))
        return self.unique_text(collected)

    # -----------------------------------------------------------------------------
    def unique_text(self, values: list[str]) -> list[str]:
        unique: dict[str, str] = {}
        for value in values:
            normalized = self.normalize_string(value)
            if normalized is None:
                continue
            key = normalized.casefold()
            if key not in unique:
                unique[key] = normalized
        return list(unique.values())

    # -----------------------------------------------------------------------------
    def build_alias_lookup_by_kind(
        self, aliases_frame: pd.DataFrame
    ) -> dict[int, dict[str, set[str]]]:
        lookup: dict[int, dict[str, set[str]]] = {}
        if aliases_frame.empty:
            return lookup
        for row in aliases_frame.to_dict(orient="records"):
            drug_id = row.get("drug_id")
            alias_kind = self.normalize_string(row.get("alias_kind"))
            alias = self.normalize_string(row.get("alias"))
            if drug_id is None or alias_kind is None or alias is None:
                continue
            by_kind = lookup.setdefault(int(drug_id), {})
            values = by_kind.setdefault(alias_kind, set())
            values.add(alias)
        return lookup

    # -----------------------------------------------------------------------------
    def group_aliases_by_kind(self, aliases: list[DrugAlias]) -> dict[str, set[str]]:
        grouped: dict[str, set[str]] = {}
        for alias in aliases:
            alias_value = self.normalize_string(alias.alias)
            alias_kind = self.normalize_string(alias.alias_kind)
            if alias_value is None or alias_kind is None:
                continue
            grouped.setdefault(alias_kind.casefold(), set()).add(alias_value)
        return grouped

    # -----------------------------------------------------------------------------
    def alias_values_for_kind(self, aliases: pd.DataFrame, alias_kind: str) -> set[str]:
        if aliases.empty:
            return set()
        selected = aliases[
            aliases["alias_kind"].astype(str).str.casefold() == alias_kind.casefold()
        ]
        values: set[str] = set()
        for item in selected["alias"].tolist():
            normalized = self.normalize_string(item)
            if normalized is not None:
                values.add(normalized)
        return values

    # -----------------------------------------------------------------------------
    def alias_model_values_for_kind(
        self,
        aliases: list[DrugAlias],
        alias_kind: str,
    ) -> set[str]:
        values: set[str] = set()
        for alias in aliases:
            if (self.normalize_string(alias.alias_kind) or "").casefold() != alias_kind.casefold():
                continue
            normalized = self.normalize_string(alias.alias)
            if normalized is not None:
                values.add(normalized)
        return values

    # -----------------------------------------------------------------------------
    def first_alias_value(self, aliases: pd.DataFrame, alias_kind: str) -> str | None:
        values = sorted(self.alias_values_for_kind(aliases, alias_kind), key=str.casefold)
        return values[0] if values else None

    # -----------------------------------------------------------------------------
    def first_alias_term_type(self, aliases: pd.DataFrame) -> str | None:
        if aliases.empty or "term_type" not in aliases.columns:
            return None
        for value in aliases["term_type"].tolist():
            normalized = self.normalize_string(value)
            if normalized is not None:
                return normalized
        return None

    # -----------------------------------------------------------------------------
    def first_alias_model_value(
        self,
        aliases: list[DrugAlias],
        alias_kind: str,
    ) -> str | None:
        values = sorted(self.alias_model_values_for_kind(aliases, alias_kind), key=str.casefold)
        return values[0] if values else None

    # -----------------------------------------------------------------------------
    def first_alias_model_term_type(self, aliases: list[DrugAlias]) -> str | None:
        for alias in aliases:
            normalized = self.normalize_string(alias.term_type)
            if normalized is not None:
                return normalized
        return None


###############################################################################
class DataSerializer:
    def __init__(
        self,
        *,
        engine: Engine | None = None,
        session_factory: sessionmaker | None = None,
    ) -> None:
        self.service = _RepositorySerializationService(
            engine=engine,
            session_factory=session_factory,
        )

    # -------------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        return getattr(self.service, name)



###############################################################################
class DocumentSerializer:
    SUPPORTED_EXTENSIONS = DOCUMENT_SUPPORTED_EXTENSIONS

    def __init__(self, documents_path: str) -> None:
        self.documents_path = documents_path

    # -------------------------------------------------------------------------
    def collect_document_paths(self) -> list[str]:
        collected: list[str] = []
        for root, _, files in os.walk(self.documents_path):
            for name in files:
                extension = os.path.splitext(name)[1].lower()
                if extension in self.SUPPORTED_EXTENSIONS:
                    collected.append(os.path.join(root, name))
                else:
                    logger.debug("Skipping unsupported document '%s'", name)
        collected.sort()
        return collected

    # -------------------------------------------------------------------------
    def load_documents(self) -> list[Document]:
        documents: list[Document] = []
        for file_path in self.collect_document_paths():
            extension = os.path.splitext(file_path)[1].lower()
            if extension == ".pdf":
                documents.extend(self.load_pdf(file_path))
            elif extension == ".docx":
                documents.extend(self.load_docx(file_path))
            elif extension == ".doc":
                logger.warning(
                    "Legacy Word document '%s' is not supported; skipping", file_path
                )
            elif extension in {".txt", ".xml"}:
                documents.extend(self.load_textual_file(file_path, extension))
        return documents

    # -------------------------------------------------------------------------
    def load_pdf(self, file_path: str) -> list[Document]:
        try:
            reader = PdfReader(file_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load PDF '%s': %s", file_path, exc)
            return []

        metadata = self.build_metadata(file_path)
        pages: list[Document] = []
        for index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to extract text from '%s' page %d: %s",
                    file_path,
                    index,
                    exc,
                )
                continue
            content = text.strip()
            if not content:
                continue
            page_metadata = dict(metadata)
            page_metadata["page_number"] = index
            pages.append(Document(page_content=content, metadata=page_metadata))
        return pages

    # -------------------------------------------------------------------------
    def load_docx(self, file_path: str) -> list[Document]:
        try:
            with zipfile.ZipFile(file_path) as archive:
                xml_content = archive.read("word/document.xml")
        except (KeyError, zipfile.BadZipFile, OSError) as exc:
            logger.error("Unable to read DOCX '%s': %s", file_path, exc)
            return []
        try:
            tree = ElementTree.fromstring(xml_content)
        except ElementTree.ParseError as exc:
            logger.error("Failed to parse DOCX '%s': %s", file_path, exc)
            return []
        namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        paragraphs: list[str] = []
        for paragraph in tree.iter(f"{namespace}p"):
            texts = [
                node.text
                for node in paragraph.iter(f"{namespace}t")
                if node.text and node.text.strip()
            ]
            if texts:
                paragraphs.append("".join(texts))
        content = "\n".join(paragraphs).strip()
        if not content:
            return []
        document = Document(page_content=content, metadata=self.build_metadata(file_path))
        return [document]

    # -------------------------------------------------------------------------
    def load_textual_file(self, file_path: str, extension: str) -> list[Document]:
        text = self.read_text_content(file_path, extension)
        if not text:
            return []
        document = Document(page_content=text, metadata=self.build_metadata(file_path))
        return [document]

    # -------------------------------------------------------------------------
    def read_text_content(self, file_path: str, extension: str) -> str:
        if extension == ".xml":
            return self.read_xml_content(file_path)
        for encoding in TEXT_FILE_FALLBACK_ENCODINGS:
            try:
                with open(file_path, "r", encoding=encoding) as handle:
                    text = handle.read()
            except (OSError, UnicodeDecodeError):
                continue
            return text.strip()
        logger.error("Failed to read text file '%s'", file_path)
        return ""

    # -------------------------------------------------------------------------
    def read_xml_content(self, file_path: str) -> str:
        try:
            tree = ElementTree.parse(file_path)
            root = tree.getroot()
            text = " ".join(segment.strip() for segment in root.itertext())
            return text.strip()
        except (OSError, ElementTree.ParseError) as exc:
            logger.error("Failed to parse XML '%s': %s", file_path, exc)
        return ""

    # -------------------------------------------------------------------------
    def build_metadata(self, file_path: str) -> dict[str, Any]:
        document_id = self.compute_document_id(file_path)
        return {
            "document_id": document_id,
            "source": file_path,
            "file_name": os.path.basename(file_path),
        }

    # -------------------------------------------------------------------------
    def compute_document_id(self, file_path: str) -> str:
        relative_path = os.path.relpath(file_path, self.documents_path)
        return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()


###############################################################################
class DocumentChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = max(chunk_size, 1)
        self.chunk_overlap = max(chunk_overlap, 0)

    # -------------------------------------------------------------------------
    def split_text(self, content: str) -> list[tuple[str, int]]:
        text = content.strip()
        if not text:
            return []
        step = max(self.chunk_size - self.chunk_overlap, 1)
        chunks: list[tuple[str, int]] = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append((chunk_text, start))
            if end >= text_length:
                break
            start += step
        return chunks

    # -------------------------------------------------------------------------
    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        chunks: list[Document] = []
        for document in documents:
            metadata = dict(document.metadata)
            for chunk_text, start_index in self.split_text(document.page_content):
                chunk_metadata = dict(metadata)
                chunk_metadata["start_index"] = start_index
                chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
        return chunks


###############################################################################
###############################################################################
class VectorSerializer:
    def __init__(
        self,
        documents_path: str,
        vector_database: LanceVectorDatabase,
        chunk_size: int,
        chunk_overlap: int,
        embedding_backend: str,
        ollama_base_url: str | None = None,
        ollama_model: str | None = None,
        hf_model: str | None = None,
        use_cloud_embeddings: bool = False,
        cloud_provider: str | None = None,
        cloud_embedding_model: str | None = None,
        embedding_batch_size: int | None = None,
        embedding_workers: int | None = None,
    ) -> None:
        if not isinstance(vector_database, LanceVectorDatabase):
            raise TypeError("vector_database must be a LanceVectorDatabase instance")
        self.vector_database = vector_database
        self.documents_path = documents_path
        self.document_serializer = DocumentSerializer(documents_path)
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.embedding_generator = EmbeddingGenerator(
            backend=embedding_backend,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,            
            use_cloud_embeddings=use_cloud_embeddings,
            cloud_provider=cloud_provider,
            cloud_embedding_model=cloud_embedding_model,
        )
        resolved_batch_size = (
            DEFAULT_EMBEDDING_BATCH_SIZE if embedding_batch_size is None else embedding_batch_size
        )
        self.embedding_batch_size = max(int(resolved_batch_size), 1)
        resolved_workers = (
            server_settings.rag.embedding_max_workers
            if embedding_workers is None
            else embedding_workers
        )
        self.embedding_workers = max(int(resolved_workers), 1)

    # -------------------------------------------------------------------------
    def serialize(self) -> dict[str, int]:
        self.vector_database.initialize()
        self.vector_database.get_table()
        documents = self.document_serializer.load_documents()
        if not documents:
            logger.warning("No documents available for embedding serialization")
            return {"documents": 0, "chunks": 0}
        chunks = self.chunker.chunk_documents(documents)
        if not chunks:
            logger.warning("Document chunking resulted in zero chunks")
            return {"documents": 0, "chunks": 0}
        batch_size = self.embedding_batch_size
        total_records = 0
        document_ids: set[str] = set()
        batch_iter: list[list[Document]] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            if batch:
                batch_iter.append(batch)
        with ThreadPoolExecutor(max_workers=self.embedding_workers) as executor:
            futures = [
                executor.submit(self._embed_chunk_batch, batch_chunks)
                for batch_chunks in batch_iter
            ]
            for future in as_completed(futures):
                records, batch_ids = future.result()
                document_ids.update(batch_ids)
                if not records:
                    continue
                self.vector_database.upsert_embeddings(records)
                total_records += len(records)
        logger.info(
            "Serialized %d documents into %d vector chunks",
            len(document_ids),
            total_records,
        )
        return {"documents": len(document_ids), "chunks": total_records}

    # -------------------------------------------------------------------------
    def _embed_chunk_batch(
        self, batch_chunks: list[Document]
    ) -> tuple[list[dict[str, Any]], set[str]]:
        if not batch_chunks:
            return [], set()
        texts = [chunk.page_content for chunk in batch_chunks]
        embeddings = self.embedding_generator.embed_texts(texts)
        if len(embeddings) != len(batch_chunks):
            raise RuntimeError("Embedding count does not match chunk count")
        document_ids = {
            str(chunk.metadata.get("document_id"))
            for chunk in batch_chunks
            if chunk.metadata.get("document_id")
        }
        records = self.build_records(batch_chunks, embeddings)
        return records, document_ids

    # -------------------------------------------------------------------------
    def build_records(
        self,
        chunks: list[Document],
        embeddings: list[list[float]],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for chunk, embedding in zip(chunks, embeddings):
            document_id = str(chunk.metadata.get("document_id", ""))
            chunk_index = chunk.metadata.get("chunk_index")
            chunk_id = (
                f"{document_id}:{chunk_index}" if chunk_index is not None else document_id
            )
            metadata = self.serialize_metadata(chunk.metadata)
            records.append(
                {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "text": chunk.page_content,
                    "embedding": embedding,
                    "source": metadata.get("source", ""),
                    "metadata": json.dumps(metadata, ensure_ascii=False),
                }
            )
        return records

    # -------------------------------------------------------------------------
    def serialize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        serialized: dict[str, Any] = {}
        for key, value in metadata.items():
            if key == "chunk_index":
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                serialized[key] = value
            else:
                serialized[key] = str(value)
        return serialized


    




