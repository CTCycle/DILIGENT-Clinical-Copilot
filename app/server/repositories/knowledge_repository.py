# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, cast

import pandas as pd
from sqlalchemy import and_, delete, exists, func, or_, select, update
from sqlalchemy.orm import Session, selectinload

from common.constants import (
    DRUG_NAME_ALLOWED_PATTERN,
    LIVERTOX_COLUMNS,
    LIVERTOX_MASTER_COLUMNS,
    LIVERTOX_REQUIRED_COLUMNS,
)
from common.utils.text_utils import (
    coerce_text,
    normalize_drug_name,
    parse_synonym_list,
    split_synonym_variants,
)
from configurations.startup import get_server_settings
from repositories import values as repository_values
from repositories.context import RepositoryContext
from repositories.database.upsert import (
    dialect_insert,
    upsert_drug_aliases,
    upsert_livertox_monographs,
)
from repositories.queries.drugs import DrugRepositoryQueries
from repositories.schemas.knowledge import (
    Drug,
    DrugAlias,
    LiverToxMonograph,
)

###############################################################################
class KnowledgeRepository:

    # -------------------------------------------------------------------------
    def __init__(self, context: RepositoryContext) -> None:
        self.context = context
        self.engine = context.engine
        self.session_factory = context.session_factory

    # -------------------------------------------------------------------------
    def save_livertox_records(self, records: pd.DataFrame) -> None:
        self._persist_livertox_records(records, replace_source=False)

    # -------------------------------------------------------------------------
    def replace_livertox_records(self, records: pd.DataFrame) -> None:
        self._persist_livertox_records(records, replace_source=True)

    # -------------------------------------------------------------------------
    def _persist_livertox_records(
        self,
        records: pd.DataFrame,
        *,
        replace_source: bool,
    ) -> None:
        rows = self.prepare_livertox_rows(records)
        if not rows:
            if replace_source:
                raise ValueError("LiverTox refresh produced no usable records")
            return
        drug_values: dict[str, dict[str, Any]] = {}
        for row in rows:
            drug_values.setdefault(
                row["_canonical_name_norm"],
                {
                    "canonical_name": row["_drug_name"],
                    "canonical_name_norm": row["_canonical_name_norm"],
                    "livertox_nbk_id": repository_values.normalize_string(
                        row.get("nbk_id")
                    ),
                },
            )
        with self.session_factory() as db_session:
            try:
                if replace_source:
                    db_session.execute(delete(LiverToxMonograph))
                    db_session.execute(
                        delete(DrugAlias).where(DrugAlias.source == "livertox")
                    )
                    db_session.execute(
                        update(Drug)
                        .where(Drug.livertox_nbk_id.is_not(None))
                        .values(livertox_nbk_id=None)
                    )
                drug_insert = dialect_insert(db_session, Drug).values(
                    list(drug_values.values())
                )
                drug_insert = drug_insert.on_conflict_do_update(
                    index_elements=[Drug.canonical_name_norm],
                    set_={
                        "canonical_name": drug_insert.excluded.canonical_name,
                        "livertox_nbk_id": func.coalesce(
                            Drug.livertox_nbk_id, drug_insert.excluded.livertox_nbk_id
                        ),
                    },
                )
                db_session.execute(drug_insert)
                db_session.flush()
                names = list(drug_values)
                drug_ids = {
                    str(name): int(drug_id)
                    for name, drug_id in db_session.execute(
                        select(Drug.canonical_name_norm, Drug.id).where(
                            Drug.canonical_name_norm.in_(names)
                        )
                    ).all()
                }
                aliases: dict[tuple[int, str, str, str], dict[str, Any]] = {}
                monographs: dict[str, dict[str, Any]] = {}
                for row in rows:
                    drug_id = drug_ids[row["_canonical_name_norm"]]
                    for alias, alias_kind in [
                        (row["_drug_name"], "canonical"),
                        *[
                            (item, "ingredient")
                            for item in self.extract_text_candidates(
                                row.get("ingredient")
                            )
                        ],
                        *[
                            (item, "brand")
                            for item in self.extract_text_candidates(
                                row.get("brand_name")
                            )
                        ],
                        *[
                            (item, "synonym")
                            for item in self.extract_synonym_candidates(
                                row.get("synonyms")
                            )
                        ],
                    ]:
                        clean_alias = repository_values.normalize_string(alias)
                        alias_norm = (
                            normalize_drug_name(clean_alias) if clean_alias else None
                        )
                        if clean_alias is None or not alias_norm:
                            continue
                        key = (drug_id, alias_norm, alias_kind, "livertox")
                        aliases[key] = {
                            "drug_id": drug_id,
                            "alias": clean_alias,
                            "alias_norm": alias_norm,
                            "alias_kind": alias_kind,
                            "source": "livertox",
                            "term_type": None,
                        }
                    monograph_key = self.build_livertox_monograph_key(row)
                    flag = repository_values.normalize_flag(
                        row.get("include_in_livertox")
                    )
                    monographs[monograph_key] = {
                        "drug_id": drug_id,
                        "monograph_key": monograph_key,
                        "drug_name_norm": row["_canonical_name_norm"],
                        "nbk_id": repository_values.normalize_string(row.get("nbk_id")),
                        "excerpt": repository_values.normalize_string(
                            row.get("excerpt")
                        ),
                        "likelihood_score": repository_values.normalize_string(
                            row.get("likelihood_score")
                        ),
                        "last_update": repository_values.normalize_date(
                            row.get("last_update")
                        ),
                        "reference_count": repository_values.to_int(
                            row.get("reference_count")
                        ),
                        "year_approved": repository_values.to_int(
                            row.get("year_approved")
                        ),
                        "agent_classification": repository_values.normalize_string(
                            row.get("agent_classification")
                        ),
                        "primary_classification": repository_values.normalize_string(
                            row.get("primary_classification")
                        ),
                        "secondary_classification": repository_values.normalize_string(
                            row.get("secondary_classification")
                        ),
                        "include_in_livertox": None if flag is None else flag == 1,
                        "source_url": repository_values.normalize_string(
                            row.get("source_url")
                        ),
                        "source_last_modified": repository_values.normalize_string(
                            row.get("source_last_modified")
                        ),
                    }
                upsert_drug_aliases(db_session, list(aliases.values()))
                upsert_livertox_monographs(db_session, list(monographs.values()))
                db_session.commit()
            except Exception:
                db_session.rollback()
                raise

    # -------------------------------------------------------------------------
    def get_livertox_records(self) -> pd.DataFrame:
        with self.session_factory() as db_session:
            drugs = (
                db_session.execute(
                    select(Drug)
                    .join(Drug.monographs)
                    .options(selectinload(Drug.monographs), selectinload(Drug.aliases))
                    .order_by(Drug.id.asc())
                )
                .scalars()
                .unique()
                .all()
            )
        records: list[dict[str, Any]] = []
        for drug in drugs:
            grouped = self.group_aliases_by_kind(list(drug.aliases))
            monographs = sorted(
                drug.monographs,
                key=lambda item: (
                    self.to_sortable_text(item.drug_name_norm),
                    self.to_sortable_text(item.source_last_modified),
                    self.to_sortable_text(item.source_url),
                    self.to_sortable_text(item.nbk_id),
                    int(item.id),
                ),
            )
            for monograph in monographs:
                records.append(
                    {
                        "drug_name": repository_values.normalize_string(
                            drug.canonical_name
                        ),
                        "nbk_id": repository_values.normalize_string(monograph.nbk_id),
                        "ingredient": repository_values.join_values(
                            grouped.get("ingredient", set())
                        ),
                        "brand_name": repository_values.join_values(
                            grouped.get("brand", set())
                        ),
                        "synonyms": repository_values.join_values(
                            grouped.get("synonym", set())
                        ),
                        "excerpt": repository_values.normalize_string(
                            monograph.excerpt
                        ),
                        "likelihood_score": repository_values.normalize_string(
                            monograph.likelihood_score
                        ),
                        "last_update": repository_values.normalize_string(
                            monograph.last_update
                        ),
                        "reference_count": monograph.reference_count,
                        "year_approved": monograph.year_approved,
                        "agent_classification": repository_values.normalize_string(
                            monograph.agent_classification
                        ),
                        "primary_classification": repository_values.normalize_string(
                            monograph.primary_classification
                        ),
                        "secondary_classification": repository_values.normalize_string(
                            monograph.secondary_classification
                        ),
                        "include_in_livertox": monograph.include_in_livertox,
                        "source_url": repository_values.normalize_string(
                            monograph.source_url
                        ),
                        "source_last_modified": repository_values.normalize_string(
                            monograph.source_last_modified
                        ),
                    }
                )
        frame = pd.DataFrame(records)
        if frame.empty:
            return pd.DataFrame(columns=LIVERTOX_COLUMNS)
        return frame.where(pd.notnull(frame), cast(Any, None)).reindex(
            columns=LIVERTOX_COLUMNS
        )

    # -------------------------------------------------------------------------
    def get_livertox_master_list(self) -> pd.DataFrame:
        frame = self.get_livertox_records()
        if frame.empty:
            return pd.DataFrame(columns=LIVERTOX_MASTER_COLUMNS)
        available = [
            column for column in LIVERTOX_MASTER_COLUMNS if column in frame.columns
        ]
        return (
            frame.reindex(columns=available or ["drug_name"])
            .dropna(subset=["drug_name"])
            .reset_index(drop=True)
        )

    # -------------------------------------------------------------------------
    def list_livertox_catalog(
        self, *, search: str | None, offset: int, limit: int
    ) -> tuple[list[dict[str, Any]], int]:
        safe_offset = max(int(offset), 0)
        safe_limit = max(int(limit), 1)
        conditions: list[Any] = []
        search_pattern = self._build_search_pattern(search)
        if search_pattern is not None:
            alias_match = exists(
                select(1).where(
                    DrugAlias.drug_id == Drug.id,
                    func.lower(func.coalesce(DrugAlias.alias, "")).like(
                        search_pattern, escape="\\"
                    ),
                )
            )
            conditions.append(
                or_(
                    func.lower(func.coalesce(Drug.canonical_name, "")).like(
                        search_pattern, escape="\\"
                    ),
                    func.lower(func.coalesce(LiverToxMonograph.excerpt, "")).like(
                        search_pattern, escape="\\"
                    ),
                    alias_match,
                )
            )
        with self.session_factory() as db_session:
            records_stmt = select(
                Drug.id,
                Drug.canonical_name,
                LiverToxMonograph.last_update,
                (
                    func.length(
                        func.trim(func.coalesce(LiverToxMonograph.excerpt, ""))
                    )
                    > 0
                ).label("has_excerpt"),
            ).join(LiverToxMonograph, Drug.id == LiverToxMonograph.drug_id)
            count_stmt = (
                select(func.count())
                .select_from(Drug)
                .join(LiverToxMonograph, Drug.id == LiverToxMonograph.drug_id)
            )
            if conditions:
                combined = and_(*conditions)
                records_stmt = records_stmt.where(combined)
                count_stmt = count_stmt.where(combined)
            total = int(db_session.execute(count_stmt).scalar_one())
            rows = db_session.execute(
                records_stmt.order_by(
                    func.lower(func.coalesce(Drug.canonical_name, "")), Drug.id.asc()
                )
                .offset(safe_offset)
                .limit(safe_limit)
            ).all()
            return [
                {
                    "drug_id": int(row.id),
                    "drug_name": row.canonical_name,
                    "last_update": repository_values.normalize_date(row.last_update),
                    "has_excerpt": bool(row.has_excerpt),
                }
                for row in rows
            ], total

    # -------------------------------------------------------------------------
    def get_livertox_excerpt(self, drug_id: int) -> dict[str, Any] | None:
        with self.session_factory() as db_session:
            row = db_session.execute(
                select(
                    Drug.id,
                    Drug.canonical_name,
                    LiverToxMonograph.excerpt,
                    LiverToxMonograph.last_update,
                )
                .join(LiverToxMonograph, Drug.id == LiverToxMonograph.drug_id)
                .where(Drug.id == int(drug_id))
            ).one_or_none()
            if row is None:
                return None
            excerpt = repository_values.normalize_string(row.excerpt)
            if excerpt is None:
                return None
            return {
                "drug_id": int(row.id),
                "drug_name": row.canonical_name,
                "excerpt": excerpt,
                "last_update": repository_values.normalize_date(row.last_update),
            }

    # -------------------------------------------------------------------------
    def get_monograph_rows(self, monograph_ids: list[int]) -> dict[int, dict[str, Any]]:
        """LiverTox monograph evidence rows keyed by monograph id."""
        ids = sorted({int(item) for item in monograph_ids})
        if not ids:
            return {}
        with self.session_factory() as db_session:
            rows = db_session.execute(
                select(LiverToxMonograph, Drug.canonical_name)
                .join(Drug, Drug.id == LiverToxMonograph.drug_id)
                .where(LiverToxMonograph.id.in_(ids))
            ).all()
        return {
            int(monograph.id): {
                "drug_id": int(monograph.drug_id),
                "drug_name": repository_values.normalize_string(drug_name),
                "drug_name_norm": monograph.drug_name_norm,
                "monograph_key": monograph.monograph_key,
                "nbk_id": repository_values.normalize_string(monograph.nbk_id),
                "excerpt": repository_values.normalize_string(monograph.excerpt),
                "likelihood_score": repository_values.normalize_string(monograph.likelihood_score),
                "last_update": repository_values.normalize_string(monograph.last_update),
                "reference_count": monograph.reference_count,
                "year_approved": monograph.year_approved,
                "agent_classification": repository_values.normalize_string(monograph.agent_classification),
                "primary_classification": repository_values.normalize_string(monograph.primary_classification),
                "secondary_classification": repository_values.normalize_string(
                    monograph.secondary_classification
                ),
                "include_in_livertox": monograph.include_in_livertox,
                "source_url": repository_values.normalize_string(monograph.source_url),
                "source_last_modified": repository_values.normalize_string(monograph.source_last_modified),
            }
            for monograph, drug_name in rows
        }

    # -------------------------------------------------------------------------
    def count_livertox_monographs(self) -> int:
        with self.session_factory() as db_session:
            return int(db_session.scalar(select(func.count(LiverToxMonograph.id))) or 0)

    # -------------------------------------------------------------------------
    def get_drug_knowledge_bundle(self, drug_id: int) -> dict[str, Any]:
        safe_drug_id = int(drug_id)
        with self.session_factory() as db_session:
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
                        LiverToxMonograph.last_update.desc(), LiverToxMonograph.id.asc()
                    )
                )
                .scalars()
                .all()
            )
            excerpt = next(
                (
                    item.excerpt
                    for item in monographs
                    if repository_values.normalize_string(item.excerpt)
                ),
                None,
            )
            return {
                "drug_id": int(drug.id),
                "drug_name": drug.canonical_name,
                "livertox_excerpt": excerpt,
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

    # -------------------------------------------------------------------------
    def prepare_livertox_rows(self, records: pd.DataFrame) -> list[dict[str, Any]]:
        if records.empty:
            return []
        prepared: list[dict[str, Any]] = []
        for row in records.where(pd.notnull(records), cast(Any, None)).to_dict(
            orient="records"
        ):
            drug_name = repository_values.normalize_string(row.get("drug_name"))
            canonical_name_norm = normalize_drug_name(drug_name) if drug_name else None
            if drug_name is None or not canonical_name_norm:
                continue
            prepared.append(
                {
                    **row,
                    "_drug_name": drug_name,
                    "_canonical_name_norm": canonical_name_norm,
                    "_source_last_modified": repository_values.normalize_string(
                        row.get("source_last_modified")
                    )
                    or "",
                    "_source_url": repository_values.normalize_string(
                        row.get("source_url")
                    )
                    or "",
                    "_last_update": repository_values.normalize_date(
                        row.get("last_update")
                    )
                    or "",
                }
            )
        prepared.sort(key=self.livertox_row_sort_key)
        return prepared

    # -------------------------------------------------------------------------
    def sanitize_livertox_records(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(records)
        if frame.empty:
            return pd.DataFrame(columns=LIVERTOX_REQUIRED_COLUMNS)
        for column in LIVERTOX_REQUIRED_COLUMNS:
            if column not in frame.columns:
                frame[column] = None
        frame = cast(pd.DataFrame, frame[LIVERTOX_REQUIRED_COLUMNS])
        required = [
            column for column in LIVERTOX_REQUIRED_COLUMNS if column not in {"synonyms"}
        ]
        frame = frame.dropna(subset=required)
        drug_names = cast(pd.Series, frame["drug_name"]).map(coerce_text)
        frame["drug_name"] = drug_names
        frame = frame[drug_names.notna()]
        drug_name_series = cast(pd.Series, frame["drug_name"])
        frame = frame[drug_name_series.map(self._is_valid_drug_name_text)]
        excerpts = cast(pd.Series, frame["excerpt"]).map(coerce_text)
        frame["excerpt"] = excerpts
        frame = frame[excerpts.notna()]
        frame["nbk_id"] = cast(pd.Series, frame["nbk_id"]).map(coerce_text)
        frame["synonyms"] = cast(pd.Series, frame["synonyms"]).map(coerce_text)
        filtered = cast(pd.DataFrame, frame)
        return filtered.drop_duplicates(
            subset=["nbk_id", "drug_name"], keep="first"
        ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    def upsert_livertox_monograph(
        self, *, db_session: Session, drug_id: int, row: dict[str, Any]
    ) -> None:
        monograph_key = self.build_livertox_monograph_key(row)
        flag = repository_values.normalize_flag(row.get("include_in_livertox"))
        monograph = self.get_monograph_by_key(db_session, monograph_key)
        if monograph is None:
            monograph = LiverToxMonograph(
                drug_id=drug_id,
                monograph_key=monograph_key,
                drug_name_norm=cast(str, row["_canonical_name_norm"]),
            )
            db_session.add(monograph)
        monograph.nbk_id = repository_values.normalize_string(row.get("nbk_id"))
        monograph.excerpt = repository_values.normalize_string(row.get("excerpt"))
        monograph.likelihood_score = repository_values.normalize_string(
            row.get("likelihood_score")
        )
        monograph.last_update = repository_values.normalize_date(row.get("last_update"))
        monograph.reference_count = repository_values.to_int(row.get("reference_count"))
        monograph.year_approved = repository_values.to_int(row.get("year_approved"))
        monograph.agent_classification = repository_values.normalize_string(
            row.get("agent_classification")
        )
        monograph.primary_classification = repository_values.normalize_string(
            row.get("primary_classification")
        )
        monograph.secondary_classification = repository_values.normalize_string(
            row.get("secondary_classification")
        )
        monograph.include_in_livertox = None if flag is None else flag == 1
        monograph.source_url = repository_values.normalize_string(row.get("source_url"))
        monograph.source_last_modified = repository_values.normalize_string(
            row.get("source_last_modified")
        )

    # -------------------------------------------------------------------------
    def get_monograph_by_key(
        self, db_session: Session, monograph_key: str
    ) -> LiverToxMonograph | None:
        return (
            db_session.execute(DrugRepositoryQueries.monograph_by_key(monograph_key))
            .scalars()
            .first()
        )

    # -------------------------------------------------------------------------
    def get_monograph_by_drug_id(
        self, db_session: Session, drug_id: int
    ) -> LiverToxMonograph | None:
        return (
            db_session.execute(DrugRepositoryQueries.monograph_by_drug_id(drug_id))
            .scalars()
            .first()
        )

    # -------------------------------------------------------------------------
    def extract_text_candidates(self, value: Any) -> list[str]:
        collected: list[str] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    collected.extend(split_synonym_variants(item))
        else:
            text_value = repository_values.normalize_string(value)
            if text_value is not None:
                collected.extend(split_synonym_variants(text_value))
        return self.unique_text(collected)

    # -------------------------------------------------------------------------
    def extract_synonym_candidates(self, value: Any) -> list[str]:
        collected: list[str] = []
        for item in parse_synonym_list(value):
            collected.extend(split_synonym_variants(item))
        return self.unique_text(collected)

    # -------------------------------------------------------------------------
    @staticmethod
    def unique_text(values: list[str]) -> list[str]:
        unique: dict[str, str] = {}
        for value in values:
            normalized = repository_values.normalize_string(value)
            if normalized is not None:
                unique.setdefault(normalized.casefold(), normalized)
        return list(unique.values())

    # -------------------------------------------------------------------------
    def group_aliases_by_kind(self, aliases: list[DrugAlias]) -> dict[str, set[str]]:
        grouped: dict[str, set[str]] = {}
        for alias in aliases:
            alias_value = repository_values.normalize_string(alias.alias)
            alias_kind = repository_values.normalize_string(alias.alias_kind)
            if alias_value and alias_kind:
                grouped.setdefault(alias_kind.casefold(), set()).add(alias_value)
        return grouped

    # -------------------------------------------------------------------------
    def alias_model_values_for_kind(
        self, aliases: list[DrugAlias], alias_kind: str
    ) -> set[str]:
        return {
            normalized
            for alias in aliases
            if (repository_values.normalize_string(alias.alias_kind) or "").casefold()
            == alias_kind.casefold()
            for normalized in [repository_values.normalize_string(alias.alias)]
            if normalized is not None
        }

    # -------------------------------------------------------------------------
    def first_alias_model_value(
        self, aliases: list[DrugAlias], alias_kind: str
    ) -> str | None:
        values = sorted(
            self.alias_model_values_for_kind(aliases, alias_kind), key=str.casefold
        )
        return values[0] if values else None

    # -------------------------------------------------------------------------
    def first_alias_model_term_type(self, aliases: list[DrugAlias]) -> str | None:
        for alias in aliases:
            value = repository_values.normalize_string(alias.term_type)
            if value is not None:
                return value
        return None

    # -------------------------------------------------------------------------
    def livertox_row_sort_key(self, row: dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            self.to_sortable_text(row.get(key))
            for key in (
                "_canonical_name_norm",
                "_source_last_modified",
                "_source_url",
                "_last_update",
                "_drug_name",
            )
        )

    # -------------------------------------------------------------------------
    def to_sortable_text(self, value: Any) -> str:
        return "" if value is None else str(value).casefold()

    # -------------------------------------------------------------------------
    def build_livertox_monograph_key(self, row: dict[str, Any]) -> str:
        payload = {
            "drug_name_norm": repository_values.normalize_string(
                row.get("_canonical_name_norm")
            )
            or "",
            "nbk_id": repository_values.normalize_string(row.get("nbk_id")) or "",
            "source_url": repository_values.normalize_string(row.get("source_url"))
            or "",
            "source_last_modified": repository_values.normalize_string(
                row.get("source_last_modified")
            )
            or "",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    # -------------------------------------------------------------------------
    def _build_search_pattern(self, search: str | None) -> str | None:
        normalized = repository_values.normalize_string(search)
        if normalized is None:
            return None
        escaped = (
            normalized.casefold()
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        return f"%{escaped}%"

    # -------------------------------------------------------------------------
    def _is_valid_drug_name_text(self, value: str) -> bool:
        ingestion = get_server_settings().ingestion
        return (
            ingestion.drug_name_min_length
            <= len(value)
            <= ingestion.drug_name_max_length
            and len(value.split()) <= ingestion.drug_name_max_tokens
            and re.fullmatch(DRUG_NAME_ALLOWED_PATTERN, value.strip()) is not None
        )
