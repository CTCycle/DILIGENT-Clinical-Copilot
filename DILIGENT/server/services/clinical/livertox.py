from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from types import SimpleNamespace
from typing import Any

import pandas as pd

from DILIGENT.server.configurations import server_settings
from DILIGENT.server.services.text.normalization import coerce_text

###############################################################################
class LiverToxData:
    
    def __init__(
        self,
        *,
        lookup: Any,
        livertox_df: pd.DataFrame,
        master_list_df: pd.DataFrame | None,
        drugs_catalog_df: pd.DataFrame | Iterable[pd.DataFrame] | None,
        record_factory: Callable[..., Any],
    ) -> None:
        self.lookup = lookup
        self.record_factory = record_factory
        self.livertox_df = livertox_df
        if isinstance(master_list_df, pd.DataFrame) and not master_list_df.empty:
            self.master_list_df = master_list_df.copy()
        else:
            self.master_list_df = self.derive_master_alias_source(livertox_df)
        self.drugs_catalog_df = self._prepare_catalog_source(drugs_catalog_df)
        self.records: list[Any] = []
        self.primary_index: dict[str, list[Any]] = {}
        self.synonym_index: dict[str, list[tuple[Any, str]]] = {}
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
            raw_name = coerce_text(getattr(row, "drug_name", None))
            if raw_name is None:
                continue
            normalized_name = self.lookup.normalize_name(raw_name)
            if not normalized_name:
                continue
            nbk_id = coerce_text(getattr(row, "nbk_id", None))
            # Some local database snapshots can contain valid monographs while
            # `nbk_id` is null for every row; keep deterministic synthetic IDs
            # so matching remains available instead of failing globally.
            if nbk_id is None:
                nbk_id = f"synthetic::{normalized_name}"
            excerpt = coerce_text(getattr(row, "excerpt", None))
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
            primary_bucket = self.primary_index.setdefault(normalized_name, [])
            if all(existing.nbk_id != record.nbk_id for existing in primary_bucket):
                primary_bucket.append(record)
            self.variant_catalog.append(
                (normalized_name, record, record.drug_name, True)
            )
            for normalized_synonym, original in synonyms.items():
                synonym_bucket = self.synonym_index.setdefault(normalized_synonym, [])
                if all(existing[0].nbk_id != record.nbk_id for existing in synonym_bucket):
                    synonym_bucket.append((record, original))
                self.variant_catalog.append(
                    (normalized_synonym, record, original, False)
                )
            for token in tokens:
                bucket = token_occurrences.setdefault(token, [])
                if record not in bucket:
                    bucket.append(record)
        self.records.sort(key=lambda record: record.drug_name.lower())
        for key, bucket in self.primary_index.items():
            self.primary_index[key] = sorted(
                bucket,
                key=lambda record: (record.drug_name.casefold(), record.nbk_id.casefold()),
            )
        for key, bucket in self.synonym_index.items():
            self.synonym_index[key] = sorted(
                bucket,
                key=lambda item: (
                    item[0].drug_name.casefold(),
                    item[0].nbk_id.casefold(),
                    item[1].casefold(),
                ),
            )
        self.variant_catalog.sort(
            key=lambda item: (item[0], item[1].drug_name.casefold(), item[1].nbk_id.casefold())
        )
        self.token_occurrences = token_occurrences

    # -------------------------------------------------------------------------
    def build_master_list_aliases(self) -> None:
        self.brand_index = {}
        self.ingredient_index = {}
        if self.master_list_df is None or self.master_list_df.empty:
            return
        for row in self.master_list_df.itertuples(index=False):
            drug_name = coerce_text(getattr(row, "drug_name", None))
            if drug_name is None:
                continue
            brand = coerce_text(getattr(row, "brand_name", None))
            ingredient = coerce_text(getattr(row, "ingredient", None))
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
        for alias_index in (self.brand_index, self.ingredient_index):
            for key, entries in alias_index.items():
                alias_index[key] = sorted(
                    entries,
                    key=lambda item: (item[0].casefold(), item[1].casefold()),
                )

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
        matches: list[Any],
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        row_index = self.ensure_row_index()
        for original, match in zip(patient_drugs, matches, strict=False):
            row_data: dict[str, Any] | None = None
            normalized_key: str = ""
            match_status = str(getattr(match, "status", "missing"))
            if match_status == "matched":
                if getattr(match, "record", None) is not None:
                    normalized_key = (
                        match.record.normalized_name
                        or self.lookup.normalize_name(match.record.drug_name)
                    )
                if not normalized_key:
                    normalized_key = self.lookup.normalize_name(
                        getattr(match, "matched_name", None)
                    )
                if normalized_key:
                    row = row_index.get(normalized_key)
                    if row:
                        row_data = dict(row)
            excerpt_candidates: list[str] = []
            if match_status == "matched":
                record_excerpt = coerce_text(
                    match.record.excerpt if getattr(match, "record", None) else None
                )
                if record_excerpt:
                    excerpt_candidates.append(record_excerpt)
                row_excerpt = coerce_text(row_data.get("excerpt") if row_data else None)
                if row_excerpt:
                    excerpt_candidates.append(row_excerpt)
            unique_excerpts = list(dict.fromkeys(excerpt_candidates))
            notes = list(getattr(match, "notes", []) or [])
            if match_status == "matched" and not unique_excerpts:
                fallback_excerpt = self.find_related_excerpt(
                    normalized_query=str(getattr(match, "normalized_query", "") or ""),
                )
                if fallback_excerpt:
                    unique_excerpts.append(fallback_excerpt)
                    notes = list(
                        dict.fromkeys([*notes, "fallback_excerpt_from_related_monograph"])
                    )
                else:
                    notes = list(dict.fromkeys([*notes, "matched_record_missing_excerpt"]))
            missing_livertox = match_status != "matched" or not unique_excerpts
            ambiguous_match = match_status == "ambiguous"
            entries.append(
                {
                    "drug_name": original,
                    "canonical_drug_name": getattr(match, "canonical_query", None),
                    "normalized_drug_name": getattr(match, "normalized_query", None),
                    "matched_livertox_row": row_data,
                    "extracted_excerpts": unique_excerpts,
                    "match_confidence": getattr(match, "confidence", None),
                    "match_reason": getattr(match, "reason", None),
                    "match_notes": notes,
                    "match_status": match_status,
                    "match_candidates": list(getattr(match, "candidate_names", []) or []),
                    "missing_livertox": missing_livertox,
                    "ambiguous_match": ambiguous_match,
                }
            )
        return entries

    # -------------------------------------------------------------------------
    def find_related_excerpt(self, normalized_query: str) -> str | None:
        query = self.lookup.normalize_name(normalized_query)
        if not query:
            return None
        if self.livertox_df is None or self.livertox_df.empty:
            return None
        best_score: tuple[int, int] | None = None
        best_excerpt: str | None = None
        for row in self.livertox_df.to_dict(orient="records"):
            drug_name = coerce_text(row.get("drug_name"))
            if drug_name is None:
                continue
            excerpt = coerce_text(row.get("excerpt"))
            if excerpt is None:
                continue
            normalized_name = self.lookup.normalize_name(drug_name)
            score = self.build_related_excerpt_score(
                normalized_query=query,
                normalized_name=normalized_name,
            )
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_excerpt = excerpt
        return best_excerpt

    # -------------------------------------------------------------------------
    def build_related_excerpt_score(
        self,
        *,
        normalized_query: str,
        normalized_name: str,
    ) -> tuple[int, int] | None:
        if not normalized_query or not normalized_name:
            return None
        if normalized_name == normalized_query:
            return (3, -abs(len(normalized_name) - len(normalized_query)))
        if normalized_name.startswith(f"{normalized_query} "):
            return (2, -abs(len(normalized_name) - len(normalized_query)))
        query_tokens = set(normalized_query.split())
        name_tokens = set(normalized_name.split())
        if query_tokens and query_tokens.issubset(name_tokens):
            return (1, -abs(len(normalized_name) - len(normalized_query)))
        return None

    # -------------------------------------------------------------------------
    def ensure_row_index(self) -> dict[str, dict[str, Any]]:
        if self.rows_by_name:
            return self.rows_by_name
        if self.livertox_df is None or self.livertox_df.empty:
            return {}
        index: dict[str, dict[str, Any]] = {}
        rows = sorted(
            self.livertox_df.to_dict(orient="records"),
            key=lambda row: (
                self.lookup.normalize_name(str(row.get("drug_name", ""))),
                str(row.get("nbk_id", "")).casefold(),
            ),
        )
        for row in rows:
            drug_name = coerce_text(row.get("drug_name"))
            if drug_name is None:
                continue
            normalized = self.lookup.normalize_name(drug_name)
            if not normalized:
                continue
            if normalized in index:
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

    # -------------------------------------------------------------------------
    def _prepare_catalog_source(
        self, source: pd.DataFrame | Iterable[pd.DataFrame] | None
    ) -> pd.DataFrame | Iterable[pd.DataFrame] | None:
        if source is None:
            return None
        if isinstance(source, pd.DataFrame):
            return source.copy() if not source.empty else None
        if isinstance(source, Iterable):
            return source
        return None

    # -------------------------------------------------------------------------
    def iter_catalog_rows(self) -> Iterator[Any]:
        catalog_source = self.drugs_catalog_df
        if catalog_source is None:
            return
        if isinstance(catalog_source, pd.DataFrame):
            yield from catalog_source.itertuples(index=False)
            return
        if isinstance(catalog_source, Iterable):
            for chunk in catalog_source:
                if isinstance(chunk, pd.DataFrame):
                    if chunk.empty:
                        continue
                    yield from chunk.itertuples(index=False)
                    continue
                if isinstance(chunk, dict):
                    yield SimpleNamespace(**chunk)
                    continue
                if hasattr(chunk, "_fields"):
                    yield chunk
                    continue
                if hasattr(chunk, "__dict__"):
                    yield chunk
                    continue
                try:
                    mapping = dict(chunk)
                except Exception:  # noqa: BLE001
                    continue
                yield SimpleNamespace(**mapping)
            return
        return
