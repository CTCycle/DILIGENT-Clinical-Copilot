from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from DILIGENT.server.utils.configurations import server_settings

###############################################################################
class LiverToxData:
    
    def __init__(
        self,
        *,
        lookup: Any,
        livertox_df: pd.DataFrame,
        master_list_df: pd.DataFrame | None,
        drugs_catalog_df: pd.DataFrame | None,
        record_factory: Callable[..., Any],
    ) -> None:
        self.lookup = lookup
        self.record_factory = record_factory
        self.livertox_df = livertox_df
        if master_list_df is not None and not master_list_df.empty:
            self.master_list_df = master_list_df.copy()
        else:
            self.master_list_df = self.derive_master_alias_source(livertox_df)
        if drugs_catalog_df is not None and not drugs_catalog_df.empty:
            self.drugs_catalog_df = drugs_catalog_df.copy()
        else:
            self.drugs_catalog_df = None
        self.records: list[Any] = []
        self.primary_index: dict[str, Any] = {}
        self.synonym_index: dict[str, tuple[Any, str]] = {}
        self.variant_catalog: list[tuple[str, Any, str, bool]] = []
        self.token_occurrences: dict[str, list[Any]] = {}
        self.token_index: dict[str, list[Any]] = {}
        self.brand_index: dict[str, list[tuple[str, str]]] = {}
        self.ingredient_index: dict[str, list[tuple[str, str]]] = {}
        self.rows_by_name: dict[str, dict[str, Any]] = {}
        self.build_records()
        self.build_master_list_aliases()
        self.finalize_token_index()

    # -------------------------------------------------------------------------
    def build_records(self) -> None:
        if self.livertox_df is None or self.livertox_df.empty:
            return
        token_occurrences: dict[str, list[Any]] = {}
        for row in self.livertox_df.itertuples(index=False):
            raw_name = self.lookup.coerce_text(getattr(row, "drug_name", None))
            if raw_name is None:
                continue
            normalized_name = self.lookup.normalize_name(raw_name)
            if not normalized_name:
                continue
            nbk_raw = getattr(row, "nbk_id", None)
            nbk_id = str(nbk_raw).strip() if nbk_raw not in (None, "") else ""
            if not nbk_id:
                continue
            excerpt = self.lookup.coerce_text(getattr(row, "excerpt", None))
            synonyms_value = getattr(row, "synonyms", None)
            synonyms = self.lookup.parse_synonyms(synonyms_value)
            tokens = self.lookup.collect_tokens(raw_name, list(synonyms.values()))
            record = self.record_factory(
                nbk_id=nbk_id,
                drug_name=raw_name,
                normalized_name=normalized_name,
                excerpt=excerpt,
                synonyms=synonyms,
                tokens=tokens,
            )
            self.records.append(record)
            if normalized_name not in self.primary_index:
                self.primary_index[normalized_name] = record
            self.variant_catalog.append(
                (normalized_name, record, record.drug_name, True)
            )
            for normalized_synonym, original in synonyms.items():
                if normalized_synonym not in self.synonym_index:
                    self.synonym_index[normalized_synonym] = (record, original)
                self.variant_catalog.append(
                    (normalized_synonym, record, original, False)
                )
            for token in tokens:
                bucket = token_occurrences.setdefault(token, [])
                if record not in bucket:
                    bucket.append(record)
        self.records.sort(key=lambda record: record.drug_name.lower())
        self.variant_catalog.sort(key=lambda item: item[0])
        self.token_occurrences = token_occurrences

    # -------------------------------------------------------------------------
    def build_master_list_aliases(self) -> None:
        self.brand_index = {}
        self.ingredient_index = {}
        if self.master_list_df is None or self.master_list_df.empty:
            return
        for row in self.master_list_df.itertuples(index=False):
            drug_name = self.lookup.coerce_text(getattr(row, "drug_name", None))
            if drug_name is None:
                continue
            brand = self.lookup.coerce_text(getattr(row, "brand_name", None))
            ingredient = self.lookup.coerce_text(getattr(row, "ingredient", None))
            for alias_type, value in ("brand", brand), ("ingredient", ingredient):
                if value is None:
                    continue
                if value.lower() == "not available":
                    continue
                for variant in self.lookup.iter_alias_variants(value):
                    normalized_variant = self.lookup.normalize_name(variant)
                    if not normalized_variant:
                        continue
                    index = (
                        self.brand_index
                        if alias_type == "brand"
                        else self.ingredient_index
                    )
                    bucket = index.setdefault(normalized_variant, [])
                    entry = (variant, drug_name)
                    if entry not in bucket:
                        bucket.append(entry)

    # -------------------------------------------------------------------------
    def finalize_token_index(self) -> None:
        if not self.token_occurrences:
            self.token_index = {}
            return
        filtered: dict[str, list[Any]] = {}
        for token, records in self.token_occurrences.items():
            if len(records) > server_settings.drugs_matcher.token_max_frequency:
                continue
            filtered[token] = sorted(
                records, key=lambda record: record.drug_name.lower()
            )
        self.token_index = filtered

    # -------------------------------------------------------------------------
    def build_drugs_to_excerpt_mapping(
        self,
        patient_drugs: list[str],
        matches: list[Any | None],
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        row_index = self.ensure_row_index()
        for original, match in zip(patient_drugs, matches, strict=False):
            row_data: dict[str, Any] | None = None
            excerpts: list[str] = []
            if match is not None:
                normalized_key: str | None = None
                if match.record is not None:
                    normalized_key = (
                        match.record.normalized_name
                        or self.lookup.normalize_name(match.record.drug_name)
                    )
                if not normalized_key:
                    normalized_key = self.lookup.normalize_name(match.matched_name)
                if normalized_key:
                    row = row_index.get(normalized_key)
                    if row:
                        row_data = dict(row)
                excerpt_value = row_data.get("excerpt") if row_data else None
                if match.record and match.record.excerpt:
                    excerpts.append(match.record.excerpt)
                if isinstance(excerpt_value, str) and excerpt_value:
                    excerpts.append(excerpt_value)
                unique_excerpts = list(dict.fromkeys(excerpts))
            else:
                row_data = None
                unique_excerpts = []
            entries.append(
                {
                    "drug_name": original,
                    "matched_livertox_row": row_data,
                    "extracted_excerpts": unique_excerpts,
                }
            )
        return entries

    # -------------------------------------------------------------------------
    def ensure_row_index(self) -> dict[str, dict[str, Any]]:
        if self.rows_by_name:
            return self.rows_by_name
        if self.livertox_df is None or self.livertox_df.empty:
            return {}
        index: dict[str, dict[str, Any]] = {}
        for row in self.livertox_df.to_dict(orient="records"):
            drug_name = self.lookup.coerce_text(row.get("drug_name"))
            if drug_name is None:
                continue
            normalized = self.lookup.normalize_name(drug_name)
            if not normalized:
                continue
            index[normalized] = row # type: ignore
        self.rows_by_name = index
        return self.rows_by_name

    # -------------------------------------------------------------------------
    def derive_master_alias_source(
        self, dataset: pd.DataFrame | None
    ) -> pd.DataFrame | None:
        if dataset is None or dataset.empty:
            return None
        required = {"drug_name", "ingredient", "brand_name"}
        if not required.issubset(dataset.columns):
            return None
        alias = dataset[list(required)].copy()
        alias = alias.dropna(subset=["drug_name"])
        alias = alias.replace("Not available", pd.NA)
        alias = alias.dropna(how="all", subset=["ingredient", "brand_name"])
        return alias.reset_index(drop=True)
