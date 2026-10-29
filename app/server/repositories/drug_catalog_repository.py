# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import and_, delete, exists, func, or_, select, update
from sqlalchemy.orm import Session

from common.constants import DRUG_NAME_ALLOWED_PATTERN
from common.utils.text_utils import normalize_drug_name
from configurations.startup import get_server_settings
from repositories import values as repository_values
from repositories.context import RepositoryContext
from repositories.database.upsert import upsert_drug_alias as atomic_upsert_drug_alias
from repositories.schemas.clinical import ClinicalDrugMention
from repositories.schemas.knowledge import (
    Drug,
    DrugAlias,
    DrugRxnormCode,
    KbMatchCache,
    LiverToxMonograph,
)

###############################################################################
def _build_search_pattern(search: str | None) -> str | None:
    normalized = repository_values.normalize_string(search)
    if normalized is None:
        return None
    escaped = re.sub(r"([%_\\])", r"\\\1", normalized.casefold())
    return f"%{escaped}%"

###############################################################################
class DrugCatalogRepository:
    """Knowledge-view access to RxNorm-coded drug concepts (browse, rename, delete)."""

    # -------------------------------------------------------------------------
    def __init__(self, context: RepositoryContext) -> None:
        self.context = context
        self.engine = context.engine
        self.session_factory = context.session_factory

    # -------------------------------------------------------------------------
    def list_rxnav_catalog(
        self, *, search: str | None, offset: int, limit: int
    ) -> tuple[list[dict[str, Any]], int]:
        safe_offset = max(int(offset), 0)
        safe_limit = max(int(limit), 1)
        has_rxnav_data = or_(
            exists(select(1).where(DrugRxnormCode.drug_id == Drug.id)),
            exists(
                select(1).where(
                    DrugAlias.drug_id == Drug.id,
                    func.lower(func.coalesce(DrugAlias.source, "")) == "rxnorm",
                )
            ),
        )
        conditions: list[Any] = [has_rxnav_data]
        search_pattern = _build_search_pattern(search)
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
                    alias_match,
                )
            )
        with self.session_factory() as db_session:
            filtered = and_(*conditions)
            total_rows = int(
                db_session.execute(
                    select(func.count()).select_from(Drug).where(filtered)
                ).scalar_one()
            )
            rows = db_session.execute(
                select(Drug.id, Drug.canonical_name, Drug.rxnav_last_update)
                .where(filtered)
                .order_by(
                    func.lower(func.coalesce(Drug.canonical_name, "")), Drug.id.asc()
                )
                .offset(safe_offset)
                .limit(safe_limit)
            ).all()
            return [
                {
                    "drug_id": int(row.id),
                    "drug_name": row.canonical_name,
                    "last_update": repository_values.normalize_date(
                        row.rxnav_last_update
                    ),
                }
                for row in rows
            ], total_rows

    # -------------------------------------------------------------------------
    def get_rxnav_alias_groups(self, drug_id: int) -> dict[str, Any] | None:
        safe_drug_id = int(drug_id)
        with self.session_factory() as db_session:
            drug = db_session.get(Drug, safe_drug_id)
            if drug is None:
                return None
            grouped: dict[str, list[dict[str, str]]] = {}
            seen: dict[str, set[str]] = {}
            for source_value, alias_value, alias_kind_value in db_session.execute(
                select(DrugAlias.source, DrugAlias.alias, DrugAlias.alias_kind).where(
                    DrugAlias.drug_id == safe_drug_id
                )
            ).all():
                source = repository_values.normalize_string(source_value) or "unknown"
                alias = repository_values.normalize_string(alias_value)
                alias_kind = (
                    repository_values.normalize_string(alias_kind_value) or "unknown"
                )
                if alias is None:
                    continue
                key = f"{alias.casefold()}::{alias_kind.casefold()}"
                if key in seen.setdefault(source, set()):
                    continue
                seen[source].add(key)
                grouped.setdefault(source, []).append(
                    {"alias": alias, "alias_kind": alias_kind}
                )
            return {
                "drug_id": safe_drug_id,
                "drug_name": drug.canonical_name,
                "groups": [
                    {"source": source, "aliases": aliases}
                    for source, aliases in sorted(grouped.items())
                ],
            }

    # -------------------------------------------------------------------------
    def update_rxnav_drug_name(
        self, drug_id: int, *, drug_name: str
    ) -> dict[str, Any] | None:
        clean_name = repository_values.normalize_string(drug_name)
        if clean_name is None:
            raise ValueError("Drug name is required.")
        normalized_name = normalize_drug_name(clean_name)
        if not normalized_name or not self.is_valid_drug_name(clean_name):
            raise ValueError("Drug name is invalid.")
        safe_drug_id = int(drug_id)
        with self.session_factory() as db_session:
            existing = db_session.get(Drug, safe_drug_id)
            if existing is None:
                return None
            conflicting = db_session.scalar(
                select(Drug)
                .where(
                    Drug.canonical_name_norm == normalized_name, Drug.id != safe_drug_id
                )
                .limit(1)
            )
            if conflicting is not None:
                raise ValueError("Another drug already uses this name.")
            previous_name = repository_values.normalize_string(existing.canonical_name)
            existing.canonical_name = clean_name
            existing.canonical_name_norm = normalized_name
            self.upsert_drug_alias(
                db_session,
                drug_id=safe_drug_id,
                alias=clean_name,
                alias_kind="canonical",
                source="manual",
                term_type=None,
            )
            if previous_name and previous_name.casefold() != clean_name.casefold():
                self.upsert_drug_alias(
                    db_session,
                    drug_id=safe_drug_id,
                    alias=previous_name,
                    alias_kind="canonical",
                    source="manual",
                    term_type=None,
                )
            db_session.commit()
            return {
                "drug_id": safe_drug_id,
                "drug_name": existing.canonical_name,
                "last_update": repository_values.normalize_date(
                    existing.rxnav_last_update
                ),
            }

    # -------------------------------------------------------------------------
    def delete_drug_with_cleanup(self, drug_id: int) -> bool:
        safe_drug_id = int(drug_id)
        with self.session_factory() as db_session:
            if db_session.get(Drug, safe_drug_id) is None:
                return False
            db_session.execute(
                update(ClinicalDrugMention)
                .where(ClinicalDrugMention.drug_id == safe_drug_id)
                .values(drug_id=None)
            )
            for model in (DrugAlias, DrugRxnormCode, LiverToxMonograph, KbMatchCache):
                db_session.execute(delete(model).where(model.drug_id == safe_drug_id))
            db_session.execute(delete(Drug).where(Drug.id == safe_drug_id))
            db_session.commit()
            return True

    # -------------------------------------------------------------------------
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
        clean_alias = repository_values.normalize_string(alias)
        alias_norm = normalize_drug_name(clean_alias) if clean_alias else None
        if not alias_norm or clean_alias is None:
            return
        atomic_upsert_drug_alias(
            db_session,
            drug_id=drug_id,
            alias=clean_alias,
            alias_norm=alias_norm,
            alias_kind=alias_kind,
            source=source,
            term_type=term_type,
        )

    # -------------------------------------------------------------------------
    def is_valid_drug_name(self, value: str) -> bool:
        normalized = value.strip()
        ingestion = get_server_settings().ingestion
        if (
            len(normalized) < ingestion.drug_name_min_length
            or len(normalized) > ingestion.drug_name_max_length
        ):
            return False
        if len(normalized.split()) > ingestion.drug_name_max_tokens:
            return False
        return re.fullmatch(DRUG_NAME_ALLOWED_PATTERN, normalized) is not None
