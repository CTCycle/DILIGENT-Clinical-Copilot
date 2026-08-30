from __future__ import annotations

import re
import json
from datetime import date
from typing import Any, Iterator, cast

import pandas as pd
from sqlalchemy import and_, delete, exists, func, or_, select, update
from sqlalchemy.orm import Session, selectinload

from common.constants import DRUG_NAME_ALLOWED_PATTERN, RXNORM_CATALOG_COLUMNS
from common.utils.logger import logger
from common.utils.text_utils import (
    normalize_drug_name,
    parse_synonym_list,
    split_synonym_variants,
)
from configurations.startup import get_server_settings
from repositories import values as repository_values
from repositories.context import RepositoryContext
from repositories.database.upsert import (
    dialect_insert,
    upsert_drug_alias as atomic_upsert_drug_alias,
    upsert_drug_aliases,
    upsert_drug_rxnorm_codes,
)
from repositories.queries.drugs import DrugRepositoryQueries
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
    # -------------------------------------------------------------------------
    def __init__(self, context: RepositoryContext) -> None:
        self.context = context
        self.engine = context.engine
        self.session_factory = context.session_factory

    # -------------------------------------------------------------------------
    def upsert_drugs_catalog_records(
        self,
        records: pd.DataFrame | list[dict[str, Any]],
        *,
        commit_interval: int | None = None,
        curated_aliases_by_canonical: dict[str, list[tuple[str, str]]] | None = None,
    ) -> None:
        del commit_interval
        prepared_rows = self.prepare_rxnav_rows(records)
        if not prepared_rows:
            return
        today_marker = date.today().isoformat()
        values_by_norm: dict[str, dict[str, Any]] = {}
        for row in prepared_rows:
            values_by_norm.setdefault(
                cast(str, row["_canonical_name_norm"]),
                {
                    "canonical_name": row["_canonical_name"],
                    "canonical_name_norm": row["_canonical_name_norm"],
                    "rxnav_last_update": today_marker,
                },
            )
        db_session = self.session_factory()
        try:
            drug_insert = dialect_insert(db_session, Drug).values(
                list(values_by_norm.values())
            )
            drug_insert = drug_insert.on_conflict_do_update(
                index_elements=[Drug.canonical_name_norm],
                set_={
                    "canonical_name": drug_insert.excluded.canonical_name,
                    "rxnav_last_update": drug_insert.excluded.rxnav_last_update,
                },
            )
            db_session.execute(drug_insert)
            db_session.flush()
            names = list(values_by_norm)
            drug_ids = {
                str(name): int(drug_id)
                for name, drug_id in db_session.execute(
                    select(Drug.canonical_name_norm, Drug.id).where(
                        Drug.canonical_name_norm.in_(names)
                    )
                ).all()
            }
            rxcuis = list({str(row["_rxcui"]) for row in prepared_rows})
            existing_mappings = {
                str(rxcui): int(drug_id)
                for rxcui, drug_id in db_session.execute(
                    select(DrugRxnormCode.rxcui, DrugRxnormCode.drug_id).where(
                        DrugRxnormCode.rxcui.in_(rxcuis)
                    )
                ).all()
            }
            for row in prepared_rows:
                current_drug_id = existing_mappings.get(row["_rxcui"])
                expected_drug_id = drug_ids[row["_canonical_name_norm"]]
                if current_drug_id is not None and current_drug_id != expected_drug_id:
                    raise RuntimeError(
                        f"Conflicting rxcui mapping for existing drug row (rxcui='{row['_rxcui']}', "
                        f"existing_drug_id={current_drug_id}, incoming_drug_id={expected_drug_id})"
                    )
            upsert_drug_rxnorm_codes(
                db_session,
                [
                    {
                        "drug_id": drug_ids[row["_canonical_name_norm"]],
                        "rxcui": row["_rxcui"],
                    }
                    for row in prepared_rows
                ],
            )
            aliases: dict[tuple[int, str, str, str], dict[str, Any]] = {}
            for row in prepared_rows:
                drug_id = drug_ids[row["_canonical_name_norm"]]
                candidates: list[tuple[Any, str, str]] = [
                    (row["_canonical_name"], "canonical", "derived"),
                    (row.get("_raw_name"), "raw_name", "rxnorm"),
                    (row.get("_standard_name"), "standard_name", "rxnorm"),
                ]
                candidates.extend(
                    (alias, "brand", "rxnorm")
                    for alias in self.extract_text_candidates(row.get("brand_names"))
                )
                candidates.extend(
                    (alias, "synonym", "rxnorm")
                    for alias in self.extract_synonym_candidates(row.get("synonyms"))
                )
                if curated_aliases_by_canonical:
                    candidates.extend(
                        (alias, kind, "curated")
                        for alias, kind in curated_aliases_by_canonical.get(
                            row["_canonical_name_norm"], []
                        )
                    )
                for alias, alias_kind, source in candidates:
                    clean_alias = repository_values.normalize_string(alias)
                    alias_norm = (
                        normalize_drug_name(clean_alias) if clean_alias else None
                    )
                    if not alias_norm or clean_alias is None:
                        continue
                    key = (drug_id, alias_norm, alias_kind, source)
                    aliases[key] = {
                        "drug_id": drug_id,
                        "alias": clean_alias,
                        "alias_norm": alias_norm,
                        "alias_kind": alias_kind,
                        "source": source,
                        "term_type": row.get("_term_type"),
                    }
            upsert_drug_aliases(db_session, list(aliases.values()))
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

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
    def get_drugs_catalog(
        self, *, offset: int = 0, limit: int | None = None
    ) -> pd.DataFrame:
        with self.session_factory() as db_session:
            drugs = (
                db_session.execute(
                    select(Drug)
                    .options(
                        selectinload(Drug.rxnorm_codes), selectinload(Drug.aliases)
                    )
                    .order_by(Drug.id.asc())
                    .offset(max(int(offset), 0))
                    .limit(None if limit is None else max(int(limit), 1))
                )
                .scalars()
                .unique()
                .all()
            )
        records: list[dict[str, Any]] = []
        for drug in drugs:
            rxnorm_aliases = [
                alias
                for alias in drug.aliases
                if (repository_values.normalize_string(alias.source) or "").casefold()
                == "rxnorm"
            ]
            rxcui_values = {
                normalized
                for normalized in (
                    repository_values.normalize_string(mapping.rxcui)
                    for mapping in drug.rxnorm_codes
                )
                if normalized is not None
            }
            if not rxnorm_aliases or not rxcui_values:
                continue
            raw_name = self._first_alias_model_value(rxnorm_aliases, "raw_name")
            standard_name = self._first_alias_model_value(
                rxnorm_aliases, "standard_name"
            )
            term_type = self._first_alias_model_term_type(rxnorm_aliases)
            brand_names = repository_values.join_values(
                self._alias_model_values_for_kind(rxnorm_aliases, "brand")
            )
            synonyms = sorted(
                self._alias_model_values_for_kind(rxnorm_aliases, "synonym")
            )
            for rxcui in sorted(rxcui_values):
                records.append(
                    {
                        "rxcui": rxcui,
                        "raw_name": raw_name
                        or repository_values.normalize_string(drug.canonical_name),
                        "term_type": term_type,
                        "name": standard_name
                        or repository_values.normalize_string(drug.canonical_name),
                        "brand_names": brand_names,
                        "synonyms": json.dumps(synonyms, ensure_ascii=False),
                    }
                )
        frame = pd.DataFrame(records)
        return (
            pd.DataFrame(columns=RXNORM_CATALOG_COLUMNS)
            if frame.empty
            else frame.reindex(columns=RXNORM_CATALOG_COLUMNS)
        )

    # -------------------------------------------------------------------------
    def stream_drugs_catalog(
        self, page_size: int | None = None
    ) -> Iterator[pd.DataFrame]:
        configured_size = get_server_settings().database.select_page_size
        chunk_size = configured_size if page_size is None else max(int(page_size), 1)
        offset = 0
        while True:
            frame = self.get_drugs_catalog(offset=offset, limit=chunk_size)
            if frame.empty:
                if self.get_drugs_catalog(offset=offset, limit=1).empty:
                    return
            else:
                yield frame.reset_index(drop=True)
            offset += chunk_size

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
    def resolve_drug_id(
        self,
        db_session: Session,
        *,
        matched_drug_name: str | None,
        rxcui: str | None,
        nbk_id: str | None,
    ) -> int | None:
        del nbk_id
        drug = self.get_drug_by_rxcui(db_session, rxcui)
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
        aliases = self.get_drug_alias_by_norm(db_session, normalized_name)
        if len(aliases) > 1:
            raise ValueError(
                f"Drug alias is ambiguous for normalized value '{normalized_name}'"
            )
        return int(aliases[0].drug_id) if aliases else None

    # -------------------------------------------------------------------------
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
        candidate_by_name = self.get_drug_by_canonical_name_norm(
            db_session, canonical_name_norm
        )
        resolved_ids = {
            int(candidate.id)
            for candidate in (candidate_by_rxcui, candidate_by_name)
            if candidate is not None
        }
        if len(resolved_ids) > 1:
            raise RuntimeError(
                "Conflicting drug selectors resolved to different rows "
                f"(canonical_name_norm='{canonical_name_norm}', rxnorm_rxcui='{rxnorm_rxcui}')"
            )
        candidate = candidate_by_rxcui or candidate_by_name
        if candidate is None:
            candidate = Drug(
                canonical_name=canonical_name,
                canonical_name_norm=canonical_name_norm,
                livertox_nbk_id=livertox_nbk_id if use_livertox_nbk_lookup else None,
                rxnav_last_update=repository_values.normalize_date(rxnav_last_update),
            )
            db_session.add(candidate)
            db_session.flush()
        self.upsert_drug_rxcui(
            db_session, drug_id=int(candidate.id), rxcui=rxnorm_rxcui
        )
        if use_livertox_nbk_lookup:
            self.try_assign_livertox_nbk_id(
                db_session, drug=candidate, livertox_nbk_id=livertox_nbk_id or ""
            )
        normalized_last_update = repository_values.normalize_date(rxnav_last_update)
        if normalized_last_update is not None:
            candidate.rxnav_last_update = normalized_last_update
        return candidate

    # -------------------------------------------------------------------------
    def assign_identifier_if_consistent(
        self, *, drug: Drug, field_name: str, incoming_value: str | None
    ) -> None:
        if incoming_value is None:
            return
        if field_name == "livertox_nbk_id":
            current_value = repository_values.normalize_string(drug.livertox_nbk_id)
        elif field_name == "canonical_name":
            current_value = repository_values.normalize_string(drug.canonical_name)
        elif field_name == "canonical_name_norm":
            current_value = repository_values.normalize_string(drug.canonical_name_norm)
        else:
            raise ValueError(f"Unsupported drug identifier field: {field_name}")
        if current_value is not None and current_value != incoming_value:
            raise RuntimeError(
                f"Conflicting {field_name} for existing drug row (drug_id={int(drug.id)}, "
                f"existing='{current_value}', incoming='{incoming_value}')"
            )
        if current_value is None:
            if field_name == "livertox_nbk_id":
                drug.livertox_nbk_id = incoming_value
            elif field_name == "canonical_name":
                drug.canonical_name = incoming_value
            else:
                drug.canonical_name_norm = incoming_value

    # -------------------------------------------------------------------------
    def upsert_drug_rxcui(
        self, db_session: Session, *, drug_id: int, rxcui: str | None
    ) -> None:
        normalized_rxcui = repository_values.normalize_string(rxcui)
        if normalized_rxcui is None:
            return
        existing = (
            db_session.execute(
                DrugRepositoryQueries.drug_rxcui_mapping(normalized_rxcui)
            )
            .scalars()
            .first()
        )
        if existing is None:
            db_session.add(DrugRxnormCode(drug_id=drug_id, rxcui=normalized_rxcui))
        elif int(existing.drug_id) != int(drug_id):
            raise RuntimeError(
                f"Conflicting rxcui mapping for existing drug row (rxcui='{normalized_rxcui}', "
                f"existing_drug_id={int(existing.drug_id)}, incoming_drug_id={drug_id})"
            )

    # -------------------------------------------------------------------------
    def get_drug_by_rxcui(self, db_session: Session, rxcui: str | None) -> Drug | None:
        normalized_rxcui = repository_values.normalize_string(rxcui)
        if normalized_rxcui is None:
            return None
        return (
            db_session.execute(
                DrugRepositoryQueries.drug_by_joined_rxcui(normalized_rxcui)
            )
            .scalars()
            .first()
        )

    # -------------------------------------------------------------------------
    def get_drug_by_canonical_name_norm(
        self, db_session: Session, name: str | None
    ) -> Drug | None:
        if name is None:
            return None
        return (
            db_session.execute(DrugRepositoryQueries.drug_by_canonical_name_norm(name))
            .scalars()
            .first()
        )

    # -------------------------------------------------------------------------
    def get_drug_alias_by_norm(
        self, db_session: Session, alias_norm: str | None
    ) -> list[DrugAlias]:
        if alias_norm is None:
            return []
        return list(
            db_session.execute(DrugRepositoryQueries.alias_by_norm(alias_norm))
            .scalars()
            .all()
        )

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

    # -------------------------------------------------------------------------
    def extract_text_candidates(self, value: Any) -> list[str]:
        if value is None:
            return []
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
    def unique_text(self, values: list[str]) -> list[str]:
        unique: dict[str, str] = {}
        for value in values:
            normalized = repository_values.normalize_string(value)
            if normalized is not None:
                unique.setdefault(normalized.casefold(), normalized)
        return list(unique.values())

    # -------------------------------------------------------------------------
    def try_assign_livertox_nbk_id(
        self, db_session: Session, *, drug: Drug, livertox_nbk_id: str
    ) -> None:
        del db_session
        normalized = repository_values.normalize_string(livertox_nbk_id)
        if normalized is None:
            return
        current = repository_values.normalize_string(drug.livertox_nbk_id)
        if current is None:
            drug.livertox_nbk_id = normalized
        elif current != normalized:
            logger.warning(
                "Skipping livertox_nbk_id update for drug_id=%d (existing='%s', incoming='%s')",
                int(drug.id),
                current,
                normalized,
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

    # -------------------------------------------------------------------------
    def prepare_rxnav_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        rxcui = repository_values.normalize_string(row.get("rxcui"))
        raw_name = repository_values.normalize_string(row.get("raw_name"))
        standard_name = repository_values.normalize_string(row.get("name"))
        canonical_name = standard_name or raw_name
        canonical_name_norm = (
            normalize_drug_name(canonical_name) if canonical_name else None
        )
        if rxcui is None or canonical_name is None or not canonical_name_norm:
            return None
        return {
            **row,
            "_rxcui": rxcui,
            "_raw_name": raw_name,
            "_standard_name": standard_name,
            "_canonical_name": canonical_name,
            "_canonical_name_norm": canonical_name_norm,
            "_term_type": repository_values.normalize_string(row.get("term_type")),
        }

    # -------------------------------------------------------------------------
    def prepare_rxnav_rows(
        self, records: pd.DataFrame | list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        frame = (
            records.copy()
            if isinstance(records, pd.DataFrame)
            else pd.DataFrame(records)
        )
        frame = frame.reindex(columns=RXNORM_CATALOG_COLUMNS)
        if frame.empty:
            return []
        prepared_rows: list[dict[str, Any]] = []
        rxcui_to_name_norm: dict[str, str] = {}
        for row in frame.where(pd.notnull(frame), cast(Any, None)).to_dict(
            orient="records"
        ):
            prepared = self.prepare_rxnav_row(row)
            if prepared is None:
                continue
            mapped = rxcui_to_name_norm.get(prepared["_rxcui"])
            if mapped is not None and mapped != prepared["_canonical_name_norm"]:
                raise RuntimeError(
                    f"Conflicting canonical_name_norm values for rxcui '{prepared['_rxcui']}'"
                )
            rxcui_to_name_norm[prepared["_rxcui"]] = prepared["_canonical_name_norm"]
            prepared_rows.append(prepared)
        prepared_rows.sort(key=self.rxnav_row_sort_key)
        return prepared_rows

    # -------------------------------------------------------------------------
    def rxnav_row_sort_key(self, row: dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            self.to_sortable_text(row.get(key))
            for key in (
                "_rxcui",
                "_canonical_name_norm",
                "_canonical_name",
                "_raw_name",
                "_standard_name",
                "_term_type",
            )
        )

    # -------------------------------------------------------------------------
    def to_sortable_text(self, value: Any) -> str:
        return repository_values.normalize_string(value) or ""

    # -------------------------------------------------------------------------
    def _alias_model_values_for_kind(
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
    def _first_alias_model_value(
        self, aliases: list[DrugAlias], alias_kind: str
    ) -> str | None:
        values = sorted(
            self._alias_model_values_for_kind(aliases, alias_kind), key=str.casefold
        )
        return values[0] if values else None

    # -------------------------------------------------------------------------
    def _first_alias_model_term_type(self, aliases: list[DrugAlias]) -> str | None:
        for alias in aliases:
            value = repository_values.normalize_string(alias.term_type)
            if value is not None:
                return value
        return None
