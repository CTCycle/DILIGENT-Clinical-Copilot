from __future__ import annotations

import re
from datetime import date
from typing import Any, cast

import pandas as pd
from sqlalchemy import select

from common.constants import (
    DRUG_NAME_ALLOWED_PATTERN,
    LIVERTOX_OPTIONAL_COLUMNS,
    LIVERTOX_REQUIRED_COLUMNS,
    RXNORM_CATALOG_COLUMNS,
)
from common.utils.text_utils import coerce_text, normalize_drug_name
from configurations.startup import get_server_settings
from repositories.database.upsert import (
    upsert_drug_aliases,
    upsert_drug_rxnorm_codes,
)
from repositories.schemas.models import Drug, DrugRxnormCode

###############################################################################
def upsert_drugs_catalog_records(
    self,
    records: pd.DataFrame | list[dict[str, Any]],
    *,
    commit_interval: int | None = None,
    curated_aliases_by_canonical: dict[str, list[tuple[str, str]]] | None = None,
) -> None:
    prepared_rows = self.prepare_rxnav_rows(records)
    if not prepared_rows:
        return
    today_marker = date.today().isoformat()
    db_session = self.session_factory()
    try:
        drug_values = _build_drug_values(prepared_rows, today_marker)
        _upsert_drug_values(db_session, drug_values)
        db_session.flush()
        drug_ids = _load_drug_ids(db_session, drug_values)
        _validate_rxcui_ownership(db_session, prepared_rows, drug_ids)
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
        upsert_drug_aliases(
            db_session,
            _build_alias_values(
                self,
                prepared_rows,
                drug_ids,
                curated_aliases_by_canonical,
            ),
        )
        db_session.commit()
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()

###############################################################################
def _build_drug_values(
    rows: list[dict[str, Any]], rxnav_last_update: str
) -> list[dict[str, Any]]:
    values_by_norm: dict[str, dict[str, Any]] = {}
    for row in rows:
        normalized = cast(str, row["_canonical_name_norm"])
        values_by_norm.setdefault(
            normalized,
            {
                "canonical_name": row["_canonical_name"],
                "canonical_name_norm": normalized,
                "rxnav_last_update": rxnav_last_update,
            },
        )
    return list(values_by_norm.values())

###############################################################################
def _upsert_drug_values(db_session, values: list[dict[str, Any]]) -> None:
    if not values:
        return
    dialect = db_session.get_bind().dialect.name
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    elif dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        raise ValueError(f"Unsupported upsert dialect: {dialect}")
    statement = insert(Drug).values(values)
    statement = statement.on_conflict_do_update(
        index_elements=[Drug.canonical_name_norm],
        set_={
            "canonical_name": statement.excluded.canonical_name,
            "rxnav_last_update": statement.excluded.rxnav_last_update,
        },
    )
    db_session.execute(statement)

###############################################################################
def _load_drug_ids(db_session, values: list[dict[str, Any]]) -> dict[str, int]:
    names = [value["canonical_name_norm"] for value in values]
    rows = db_session.execute(
        select(Drug.canonical_name_norm, Drug.id).where(
            Drug.canonical_name_norm.in_(names)
        )
    ).all()
    return {str(normalized): int(drug_id) for normalized, drug_id in rows}

###############################################################################
def _validate_rxcui_ownership(
    db_session,
    rows: list[dict[str, Any]],
    drug_ids: dict[str, int],
) -> None:
    rxcuis = list({row["_rxcui"] for row in rows})
    existing = db_session.execute(
        select(DrugRxnormCode.rxcui, DrugRxnormCode.drug_id).where(
            DrugRxnormCode.rxcui.in_(rxcuis)
        )
    ).all()
    existing_by_rxcui = {str(rxcui): int(drug_id) for rxcui, drug_id in existing}
    for row in rows:
        current_drug_id = existing_by_rxcui.get(row["_rxcui"])
        expected_drug_id = drug_ids[row["_canonical_name_norm"]]
        if current_drug_id is not None and current_drug_id != expected_drug_id:
            raise RuntimeError(
                f"Conflicting rxcui mapping for existing drug row "
                f"(rxcui='{row['_rxcui']}', existing_drug_id={current_drug_id}, "
                f"incoming_drug_id={expected_drug_id})"
            )

###############################################################################
def _build_alias_values(
    self,
    rows: list[dict[str, Any]],
    drug_ids: dict[str, int],
    curated_aliases_by_canonical: dict[str, list[tuple[str, str]]] | None,
) -> list[dict[str, Any]]:
    values: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for row in rows:
        drug_id = drug_ids[row["_canonical_name_norm"]]
        term_type = cast(str | None, row.get("_term_type"))
        candidates: list[tuple[Any, str, str]] = [
            (row["_canonical_name"], "canonical", "derived"),
            (row.get("_raw_name"), "raw_name", "rxnorm"),
            (row.get("_standard_name"), "standard_name", "rxnorm"),
        ]
        candidates.extend(
            (brand, "brand", "rxnorm")
            for brand in self.extract_text_candidates(row.get("brand_names"))
        )
        candidates.extend(
            (synonym, "synonym", "rxnorm")
            for synonym in self.extract_synonym_candidates(row.get("synonyms"))
        )
        if curated_aliases_by_canonical:
            candidates.extend(
                (alias, kind, "curated")
                for alias, kind in curated_aliases_by_canonical.get(
                    row["_canonical_name_norm"], []
                )
            )
        for alias, alias_kind, source in candidates:
            clean_alias = self.normalize_string(alias)
            if clean_alias is None:
                continue
            alias_norm = normalize_drug_name(clean_alias)
            if not alias_norm:
                continue
            key = (drug_id, alias_norm, alias_kind, source)
            values[key] = {
                "drug_id": drug_id,
                "alias": clean_alias,
                "alias_norm": alias_norm,
                "alias_kind": alias_kind,
                "source": source,
                "term_type": term_type,
            }
    return list(values.values())

###############################################################################
def resolve_commit_interval(self, override: int | None) -> int:
    if override is not None:
        return max(int(override), 1)
    database = get_server_settings().database
    return max(int(getattr(database, "write_batch_size", database.insert_batch_size)), 1)

###############################################################################
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

###############################################################################
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

###############################################################################
def rxnav_row_sort_key(self, row: dict[str, Any]) -> tuple[str, ...]:
    return (
        self.to_sortable_text(row.get("_rxcui")),
        self.to_sortable_text(row.get("_canonical_name_norm")),
        self.to_sortable_text(row.get("_canonical_name")),
        self.to_sortable_text(row.get("_raw_name")),
        self.to_sortable_text(row.get("_standard_name")),
        self.to_sortable_text(row.get("_term_type")),
    )

###############################################################################
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

###############################################################################
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
