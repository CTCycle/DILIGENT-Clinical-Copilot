from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from repositories.queries.drugs import DrugRepositoryQueries
from repositories.schemas.models import (
    Drug,
    DrugAlias,
    DrugRxnormCode,
    LiverToxMonograph,
)
from repositories.serialization.evidence_data import normalize_drug_name
from services.text.synonyms import parse_synonym_list, split_synonym_variants

# Extracted from the facade helper module; functions intentionally accept the facade instance.


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
        db_session,
        canonical_name_norm,
    )
    resolved_ids: set[int] = set()
    for candidate in (candidate_by_rxcui, candidate_by_name):
        if candidate is not None:
            resolved_ids.add(int(candidate.id))
    if len(resolved_ids) > 1:
        raise RuntimeError(
            "Conflicting drug selectors resolved to different rows "
            f"(canonical_name_norm='{canonical_name_norm}', "
            f"rxnorm_rxcui='{rxnorm_rxcui}')"
        )
    candidate = candidate_by_rxcui or candidate_by_name
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
        self.try_assign_livertox_nbk_id(
            db_session,
            drug=candidate,
            livertox_nbk_id=livertox_nbk_id or "",
        )
    normalized_rxnav_last_update = self.normalize_date(rxnav_last_update)
    if normalized_rxnav_last_update is not None:
        candidate.rxnav_last_update = normalized_rxnav_last_update
    return candidate


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


def get_monograph_by_key(
    self,
    db_session: Session,
    monograph_key: str,
) -> LiverToxMonograph | None:
    return (
        db_session.execute(DrugRepositoryQueries.monograph_by_key(monograph_key))
        .scalars()
        .first()
    )


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


def extract_synonym_candidates(self, value: Any) -> list[str]:
    collected: list[str] = []
    for item in parse_synonym_list(value):
        collected.extend(split_synonym_variants(item))
    return self.unique_text(collected)


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


def group_aliases_by_kind(self, aliases: list[DrugAlias]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for alias in aliases:
        alias_value = self.normalize_string(alias.alias)
        alias_kind = self.normalize_string(alias.alias_kind)
        if alias_value is None or alias_kind is None:
            continue
        grouped.setdefault(alias_kind.casefold(), set()).add(alias_value)
    return grouped


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


def alias_model_values_for_kind(
    self,
    aliases: list[DrugAlias],
    alias_kind: str,
) -> set[str]:
    values: set[str] = set()
    for alias in aliases:
        if (
            self.normalize_string(alias.alias_kind) or ""
        ).casefold() != alias_kind.casefold():
            continue
        normalized = self.normalize_string(alias.alias)
        if normalized is not None:
            values.add(normalized)
    return values


def first_alias_value(self, aliases: pd.DataFrame, alias_kind: str) -> str | None:
    values = sorted(self.alias_values_for_kind(aliases, alias_kind), key=str.casefold)
    return values[0] if values else None


def first_alias_term_type(self, aliases: pd.DataFrame) -> str | None:
    if aliases.empty or "term_type" not in aliases.columns:
        return None
    for value in aliases["term_type"].tolist():
        normalized = self.normalize_string(value)
        if normalized is not None:
            return normalized
    return None


def first_alias_model_value(
    self,
    aliases: list[DrugAlias],
    alias_kind: str,
) -> str | None:
    values = sorted(
        self.alias_model_values_for_kind(aliases, alias_kind), key=str.casefold
    )
    return values[0] if values else None


def first_alias_model_term_type(self, aliases: list[DrugAlias]) -> str | None:
    for alias in aliases:
        normalized = self.normalize_string(alias.term_type)
        if normalized is not None:
            return normalized
    return None
