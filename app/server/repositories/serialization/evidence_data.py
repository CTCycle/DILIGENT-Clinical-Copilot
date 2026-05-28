from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Iterator, cast

import pandas as pd
from sqlalchemy import and_, delete, exists, func, or_, select, update
from sqlalchemy.orm import Session, selectinload

from common.constants import (
    LIVERTOX_COLUMNS,
    LIVERTOX_MASTER_COLUMNS,
    RXNORM_CATALOG_COLUMNS,
)
from common.utils.logger import logger
from configurations.startup import get_server_settings
from repositories.schemas.models import (
    ClinicalSessionDrug,
    Drug,
    DrugAlias,
    DrugRxnormCode,
    KbMatchCache,
    LiverToxMonograph,
)
from services.text.normalization import normalize_drug_name

# Extracted from the facade module; functions intentionally accept the facade instance.


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


def livertox_row_sort_key(self, row: dict[str, Any]) -> tuple[str, ...]:
    return (
        self.to_sortable_text(row.get("_canonical_name_norm")),
        self.to_sortable_text(row.get("_source_last_modified")),
        self.to_sortable_text(row.get("_source_url")),
        self.to_sortable_text(row.get("_last_update")),
        self.to_sortable_text(row.get("_drug_name")),
    )


def to_sortable_text(self, value: Any) -> str:
    if value is None:
        return ""
    return str(value).casefold()


def upsert_livertox_monograph(
    self,
    *,
    db_session: Session,
    drug_id: int,
    row: dict[str, Any],
) -> None:
    monograph_key = self.build_livertox_monograph_key(row)
    monograph = self.get_monograph_by_key(db_session, monograph_key)
    if monograph is None:
        monograph = LiverToxMonograph(
            drug_id=drug_id,
            monograph_key=monograph_key,
            drug_name_norm=cast(str, row["_canonical_name_norm"]),
        )
        db_session.add(monograph)
    monograph.monograph_key = monograph_key
    monograph.drug_name_norm = cast(str, row["_canonical_name_norm"])
    monograph.nbk_id = self.normalize_string(row.get("nbk_id"))
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


def build_livertox_monograph_key(self, row: dict[str, Any]) -> str:
    identity_payload = {
        "drug_name_norm": self.normalize_string(row.get("_canonical_name_norm")) or "",
        "nbk_id": self.normalize_string(row.get("nbk_id")) or "",
        "source_url": self.normalize_string(row.get("source_url")) or "",
        "source_last_modified": self.normalize_string(row.get("source_last_modified"))
        or "",
    }
    serialized = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def get_livertox_records(self) -> pd.DataFrame:
    self.ensure_session_result_table()
    db_session = self.session_factory()
    try:
        drugs = (
            db_session.execute(
                select(Drug)
                .join(Drug.monographs)
                .options(
                    selectinload(Drug.monographs),
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
        monographs = sorted(
            list(drug.monographs),
            key=lambda item: (
                self.to_sortable_text(item.drug_name_norm),
                self.to_sortable_text(item.source_last_modified),
                self.to_sortable_text(item.source_url),
                self.to_sortable_text(item.nbk_id),
                int(item.id),
            ),
        )
        grouped_aliases = self.group_aliases_by_kind(list(drug.aliases))
        for monograph in monographs:
            records.append(
                {
                    "drug_name": self.normalize_string(drug.canonical_name),
                    "nbk_id": self.normalize_string(monograph.nbk_id),
                    "ingredient": self.join_values(
                        grouped_aliases.get("ingredient", set())
                    ),
                    "brand_name": self.join_values(grouped_aliases.get("brand", set())),
                    "synonyms": self.join_values(grouped_aliases.get("synonym", set())),
                    "excerpt": self.normalize_string(monograph.excerpt),
                    "likelihood_score": self.normalize_string(
                        monograph.likelihood_score
                    ),
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


def get_livertox_master_list(self) -> pd.DataFrame:
    frame = self.get_livertox_records()
    if frame.empty:
        return pd.DataFrame(columns=LIVERTOX_MASTER_COLUMNS)
    available = [
        column for column in LIVERTOX_MASTER_COLUMNS if column in frame.columns
    ]
    if not available:
        return pd.DataFrame(columns=["drug_name"])
    return (
        frame.reindex(columns=available)
        .dropna(subset=["drug_name"])
        .reset_index(drop=True)
    )


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


def stream_drugs_catalog(self, page_size: int | None = None) -> Iterator[pd.DataFrame]:
    chunk_size = (
        get_server_settings().database.select_page_size
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
            }
        monographs = (
            db_session.execute(
                select(LiverToxMonograph)
                .where(LiverToxMonograph.drug_id == safe_drug_id)
                .order_by(
                    LiverToxMonograph.last_update.desc(),
                    LiverToxMonograph.source_last_modified.desc(),
                    LiverToxMonograph.id.asc(),
                )
            )
            .scalars()
            .all()
        )
        livertox_excerpt = next(
            (
                self.normalize_string(monograph.excerpt)
                for monograph in monographs
                if self.normalize_string(monograph.excerpt) is not None
            ),
            None,
        )
        return {
            "drug_id": int(drug.id),
            "drug_name": drug.canonical_name,
            "livertox_excerpt": livertox_excerpt,
            "livertox_monographs": [
                {
                    "monograph_key": item.monograph_key,
                    "nbk_id": item.nbk_id,
                    "likelihood_score": item.likelihood_score,
                    "last_update": item.last_update,
                    "source_url": item.source_url,
                    "source_last_modified": item.source_last_modified,
                }
                for item in monographs
            ],
        }
    finally:
        db_session.close()


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
            delete(KbMatchCache).where(KbMatchCache.drug_id == safe_drug_id)
        )
        db_session.execute(delete(Drug).where(Drug.id == safe_drug_id))
        db_session.commit()
        return True
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def resolve_drug_id_from_match_cache(
    self,
    db_session: Session,
    *,
    normalized_drug_key: str,
) -> int | None:
    if not normalized_drug_key:
        return None
    cache = db_session.scalar(
        select(KbMatchCache)
        .where(
            KbMatchCache.normalized_drug_key == normalized_drug_key,
            KbMatchCache.invalidated_at.is_(None),
            KbMatchCache.confidence
            >= get_server_settings().drugs_matcher.min_confidence,
        )
        .order_by(KbMatchCache.updated_at.desc(), KbMatchCache.id.desc())
        .limit(1)
    )
    if cache is None or cache.drug_id is None:
        return None
    drug = db_session.get(Drug, int(cache.drug_id))
    if drug is None:
        cache.invalidated_at = datetime.utcnow()
        cache.invalidation_reason = "matched_drug_deleted"
        return None
    if (
        cache.rxnorm_rxcui
        and self.get_drug_by_rxcui(db_session, cache.rxnorm_rxcui) is None
    ):
        cache.invalidated_at = datetime.utcnow()
        cache.invalidation_reason = "rxnorm_code_no_longer_resolves"
        return None
    if cache.livertox_monograph_key:
        monograph = db_session.scalar(
            select(LiverToxMonograph).where(
                LiverToxMonograph.monograph_key == cache.livertox_monograph_key,
                LiverToxMonograph.drug_id == cache.drug_id,
            )
        )
        if monograph is None:
            cache.invalidated_at = datetime.utcnow()
            cache.invalidation_reason = "livertox_monograph_identity_changed"
            return None
    return int(cache.drug_id)


def upsert_high_confidence_kb_match_cache(
    self,
    db_session: Session,
    *,
    raw_drug_name: str,
    raw_drug_name_norm: str,
    normalized_drug_key: str,
    drug_id: int | None,
    rxnorm_rxcui: str | None,
    livertox_nbk_id: str | None,
    source: str,
    confidence: float | None,
    evidence: dict[str, Any],
    ambiguous: bool,
) -> None:
    if (
        drug_id is None
        or confidence is None
        or confidence < get_server_settings().drugs_matcher.min_confidence
        or ambiguous
        or source not in {"rxnav", "livertox", "rag"}
    ):
        return
    monograph = db_session.scalar(
        select(LiverToxMonograph)
        .where(LiverToxMonograph.drug_id == drug_id)
        .order_by(LiverToxMonograph.id.desc())
        .limit(1)
    )
    if livertox_nbk_id:
        matching_nbk_count = db_session.scalar(
            select(func.count())
            .select_from(LiverToxMonograph)
            .where(LiverToxMonograph.nbk_id == livertox_nbk_id)
        )
        if matching_nbk_count and int(matching_nbk_count) > 1 and monograph is None:
            return
    evidence_json = json.dumps(evidence, ensure_ascii=False, default=str)
    existing = db_session.scalar(
        select(KbMatchCache).where(
            KbMatchCache.normalized_drug_key == normalized_drug_key,
            KbMatchCache.source == source,
        )
    )
    now = datetime.utcnow()
    deterministic_evidence_version = None
    if rxnorm_rxcui:
        deterministic_evidence_version = f"rxnorm:{rxnorm_rxcui}"
    if monograph is not None:
        deterministic_evidence_version = f"livertox:{monograph.monograph_key}"
    if existing is None:
        db_session.add(
            KbMatchCache(
                raw_drug_name=raw_drug_name,
                raw_drug_name_norm=raw_drug_name_norm,
                normalized_drug_key=normalized_drug_key,
                drug_id=drug_id,
                rxnorm_rxcui=rxnorm_rxcui,
                livertox_monograph_key=monograph.monograph_key if monograph else None,
                livertox_nbk_id=livertox_nbk_id,
                source=source,
                confidence=confidence,
                evidence_json=evidence_json,
                deterministic_evidence_version=deterministic_evidence_version,
                updated_at=now,
            )
        )
        return
    existing.raw_drug_name = raw_drug_name
    existing.raw_drug_name_norm = raw_drug_name_norm
    existing.drug_id = drug_id
    existing.rxnorm_rxcui = rxnorm_rxcui
    existing.livertox_monograph_key = monograph.monograph_key if monograph else None
    existing.livertox_nbk_id = livertox_nbk_id
    existing.confidence = confidence
    existing.evidence_json = evidence_json
    existing.deterministic_evidence_version = deterministic_evidence_version
    existing.invalidated_at = None
    existing.invalidation_reason = None
    existing.updated_at = now
