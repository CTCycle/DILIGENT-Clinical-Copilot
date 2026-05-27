from __future__ import annotations

import re
from datetime import date
from typing import Any, cast

import pandas as pd

from common.constants import (
    DRUG_NAME_ALLOWED_PATTERN,
    LIVERTOX_OPTIONAL_COLUMNS,
    LIVERTOX_REQUIRED_COLUMNS,
    RXNORM_CATALOG_COLUMNS,
)
from configurations.startup import get_server_settings
from services.text.normalization import coerce_text, normalize_drug_name

# Extracted from the facade module; functions intentionally accept the facade instance.

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

def resolve_commit_interval(self, override: int | None) -> int:
    if override is not None:
        return max(int(override), 1)
    return max(int(get_server_settings().database.insert_commit_interval), 1)

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

def rxnav_row_sort_key(self, row: dict[str, Any]) -> tuple[str, ...]:
    return (
        self.to_sortable_text(row.get("_rxcui")),
        self.to_sortable_text(row.get("_canonical_name_norm")),
        self.to_sortable_text(row.get("_canonical_name")),
        self.to_sortable_text(row.get("_raw_name")),
        self.to_sortable_text(row.get("_standard_name")),
        self.to_sortable_text(row.get("_term_type")),
    )

def sanitize_livertox_records(self, records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=LIVERTOX_REQUIRED_COLUMNS)
    for column in LIVERTOX_REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = None
    df = cast(pd.DataFrame, df[LIVERTOX_REQUIRED_COLUMNS])
    drop_columns = [
        column
        for column in LIVERTOX_REQUIRED_COLUMNS
        if column not in LIVERTOX_OPTIONAL_COLUMNS
    ]
    df = cast(pd.DataFrame, df.dropna(subset=drop_columns))
    drug_names = cast(pd.Series, df["drug_name"]).apply(coerce_text)
    df["drug_name"] = drug_names
    df = cast(pd.DataFrame, df[drug_names.notna()])
    drug_names = cast(pd.Series, df["drug_name"])
    df = cast(pd.DataFrame, df[drug_names.apply(self.is_valid_drug_name)])
    excerpts = cast(pd.Series, df["excerpt"]).apply(coerce_text)
    df["excerpt"] = excerpts
    df = cast(pd.DataFrame, df[excerpts.notna()])
    df["nbk_id"] = cast(pd.Series, df["nbk_id"]).apply(coerce_text)
    df["synonyms"] = cast(pd.Series, df["synonyms"]).apply(coerce_text)
    df = cast(
        pd.DataFrame,
        df.drop_duplicates(subset=["nbk_id", "drug_name"], keep="first"),
    )
    return df.reset_index(drop=True)

def is_valid_drug_name(self, value: str) -> bool:
    normalized = value.strip()
    min_length = get_server_settings().ingestion.drug_name_min_length
    max_length = get_server_settings().ingestion.drug_name_max_length
    max_tokens = get_server_settings().ingestion.drug_name_max_tokens
    if len(normalized) < min_length or len(normalized) > max_length:
        return False
    if len(normalized.split()) > max_tokens:
        return False
    if not re.fullmatch(DRUG_NAME_ALLOWED_PATTERN, normalized):
        return False
    return True
