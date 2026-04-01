from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterator, cast
from xml.etree import ElementTree

import pandas as pd
from pypdf import PdfReader
from sqlalchemy import and_, delete, exists, func, inspect, or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from DILIGENT.server.configurations import server_settings
from DILIGENT.server.common.constants import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DRUG_NAME_ALLOWED_PATTERN,
    DOCUMENT_SUPPORTED_EXTENSIONS,
    LIVERTOX_COLUMNS,
    LIVERTOX_MASTER_COLUMNS,
    LIVERTOX_OPTIONAL_COLUMNS,
    LIVERTOX_REQUIRED_COLUMNS,
    RXNORM_CATALOG_COLUMNS,
    TABLE_DRUGS,
    TABLE_DRUG_RXNORM_CODES,
    TABLE_DRUG_ALIASES,
    TABLE_LIVERTOX_MONOGRAPHS,
    TEXT_FILE_FALLBACK_ENCODINGS,
)
from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.repositories.queries.data import DataRepositoryQueries
from DILIGENT.server.repositories.queries.drugs import DrugRepositoryQueries
from DILIGENT.server.repositories.schemas.models import (
    ClinicalSession,
    ClinicalSessionDrug,
    ClinicalSessionLab,
    ClinicalSessionResult,
    ClinicalSessionSection,
    Drug,
    DrugAlias,
    DrugRxnormCode,
    LiverToxMonograph,
)
from DILIGENT.server.repositories.vectors import LanceVectorDatabase
from DILIGENT.server.services.retrieval.embeddings import EmbeddingGenerator
from DILIGENT.server.services.text.normalization import coerce_text, normalize_drug_name
from DILIGENT.server.services.text.synonyms import parse_synonym_list, split_synonym_variants


###############################################################################
@dataclass
class Document:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


###############################################################################
class _RepositorySerializationService:
    def __init__(self, queries: DataRepositoryQueries | None = None) -> None:
        self.queries = queries or DataRepositoryQueries()
        self.engine = self.queries.database.backend.engine  # type: ignore[attr-defined]
        self.session_factory = sessionmaker(
            bind=self.engine,
            future=True,
        )
        self.ensure_session_result_table()

    # -------------------------------------------------------------------------
    def save_clinical_session(self, session_data: dict[str, Any]) -> None:
        if not session_data:
            logger.warning("Skipping clinical session save; payload is empty")
            return
        self.ensure_session_result_table()
        db_session = self.session_factory()
        try:
            persisted_session = ClinicalSession(
                patient_name=self.normalize_string(session_data.get("patient_name")),
                session_timestamp=self.parse_datetime(
                    session_data.get("session_timestamp")
                ),
                hepatic_pattern=self.normalize_string(session_data.get("hepatic_pattern")),
                parsing_model=self.normalize_string(session_data.get("parsing_model")),
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
            ClinicalSession.__tablename__,
            ClinicalSessionResult.__tablename__,
            Drug.__tablename__,
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
            ClinicalSession.__tablename__: {"session_status"},
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
        drugs_frame = self.queries.load_table(TABLE_DRUGS)
        monographs_frame = self.queries.load_table(TABLE_LIVERTOX_MONOGRAPHS)
        aliases_frame = self.queries.load_table(TABLE_DRUG_ALIASES)
        if drugs_frame.empty or monographs_frame.empty:
            return pd.DataFrame(columns=LIVERTOX_COLUMNS)
        merged = monographs_frame.merge(
            drugs_frame,
            left_on="drug_id",
            right_on="id",
            how="left",
            suffixes=("_monograph", "_drug"),
        )
        aliases_by_drug = self.build_alias_lookup_by_kind(aliases_frame)
        records: list[dict[str, Any]] = []
        for row in merged.to_dict(orient="records"):
            drug_id = row.get("drug_id")
            grouped_aliases = (
                aliases_by_drug.get(int(drug_id), {}) if drug_id is not None else {}
            )
            records.append(
                {
                    "drug_name": self.normalize_string(row.get("canonical_name")),
                    "nbk_id": self.normalize_string(row.get("livertox_nbk_id")),
                    "ingredient": self.join_values(grouped_aliases.get("ingredient", set())),
                    "brand_name": self.join_values(grouped_aliases.get("brand", set())),
                    "synonyms": self.join_values(grouped_aliases.get("synonym", set())),
                    "excerpt": self.normalize_string(row.get("excerpt")),
                    "likelihood_score": self.normalize_string(row.get("likelihood_score")),
                    "last_update": self.normalize_string(row.get("last_update")),
                    "reference_count": row.get("reference_count"),
                    "year_approved": row.get("year_approved"),
                    "agent_classification": self.normalize_string(
                        row.get("agent_classification")
                    ),
                    "primary_classification": self.normalize_string(
                        row.get("primary_classification")
                    ),
                    "secondary_classification": self.normalize_string(
                        row.get("secondary_classification")
                    ),
                    "include_in_livertox": row.get("include_in_livertox"),
                    "source_url": self.normalize_string(row.get("source_url")),
                    "source_last_modified": self.normalize_string(
                        row.get("source_last_modified")
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
        drugs_frame = self.queries.load_table(TABLE_DRUGS)
        rxcui_frame = self.queries.load_table(TABLE_DRUG_RXNORM_CODES)
        aliases_frame = self.queries.load_table(TABLE_DRUG_ALIASES)
        if drugs_frame.empty:
            return pd.DataFrame(columns=RXNORM_CATALOG_COLUMNS)
        if aliases_frame.empty:
            return pd.DataFrame(columns=RXNORM_CATALOG_COLUMNS)
        source_series = aliases_frame.get("source", pd.Series(dtype=str))
        aliases_frame = aliases_frame[
            source_series.astype(str).str.casefold() == "rxnorm"
        ].copy()
        if aliases_frame.empty:
            return pd.DataFrame(columns=RXNORM_CATALOG_COLUMNS)
        rxcui_by_drug: dict[int, set[str]] = {}
        if not rxcui_frame.empty:
            for mapping in rxcui_frame.to_dict(orient="records"):
                drug_id = mapping.get("drug_id")
                normalized_rxcui = self.normalize_string(mapping.get("rxcui"))
                if drug_id is None or normalized_rxcui is None:
                    continue
                values = rxcui_by_drug.setdefault(int(drug_id), set())
                values.add(normalized_rxcui)
        records: list[dict[str, Any]] = []
        for row in drugs_frame.to_dict(orient="records"):
            drug_id = int(row["id"])
            rxcui_values = set(rxcui_by_drug.get(drug_id, set()))
            primary_rxcui = self.normalize_string(row.get("rxnorm_rxcui"))
            if primary_rxcui is not None:
                rxcui_values.add(primary_rxcui)
            if not rxcui_values:
                continue
            aliases = aliases_frame[aliases_frame["drug_id"] == drug_id]
            raw_name = self.first_alias_value(aliases, "raw_name")
            standard_name = self.first_alias_value(aliases, "standard_name")
            term_type = self.first_alias_term_type(aliases)
            brand_names = self.join_values(self.alias_values_for_kind(aliases, "brand"))
            synonyms = sorted(self.alias_values_for_kind(aliases, "synonym"))
            for rxcui in sorted(rxcui_values):
                records.append(
                    {
                        "rxcui": rxcui,
                        "raw_name": raw_name
                        or self.normalize_string(row.get("canonical_name")),
                        "term_type": term_type,
                        "name": standard_name
                        or self.normalize_string(row.get("canonical_name")),
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
                    func.lower(func.coalesce(ClinicalSession.patient_name, "")).like(
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
            sessions_stmt = select(ClinicalSession)
            count_stmt = select(func.count()).select_from(ClinicalSession)
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
                .scalars()
                .all()
            )
            items = [
                {
                    "session_id": int(row.id),
                    "patient_name": self.normalize_string(row.patient_name),
                    "session_timestamp": row.session_timestamp,
                    "status": self.normalize_session_status(row.session_status),
                    "total_duration": self.to_float(row.total_duration),
                }
                for row in rows
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
    def delete_session(self, session_id: int) -> bool:
        self.ensure_session_result_table()
        safe_session_id = int(session_id)
        db_session = self.session_factory()
        try:
            existing = db_session.get(ClinicalSession, safe_session_id)
            if existing is None:
                return False
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
            return normalized
        return parsed.date().isoformat()

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
        mapping = [
            ("alt", "alt_value", "alt_upper_limit"),
            ("alp", "alp_value", "alp_upper_limit"),
        ]
        for lab_code, value_key, upper_limit_key in mapping:
            value_raw = self.normalize_string(session_data.get(value_key))
            upper_limit_raw = self.normalize_string(session_data.get(upper_limit_key))
            if value_raw is None and upper_limit_raw is None:
                continue
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
                    match_confidence=self.to_float(item.get("match_confidence")),
                    match_reason=self.normalize_string(item.get("match_reason")),
                    match_notes=notes_value,
                )
            )

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


###############################################################################
class DataSerializer:
    def __init__(self, queries: DataRepositoryQueries | None = None) -> None:
        self.service = _RepositorySerializationService(queries=queries)

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


    


