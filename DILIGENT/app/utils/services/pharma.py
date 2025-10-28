from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from DILIGENT.app.utils.references import MATCHING_STOPWORDS as BASE_MATCHING_STOPWORDS
from DILIGENT.app.utils.updater.livertox import LiverToxUpdater

__all__ = [
    "LiverToxUpdater",
    "MonographRecord",
    "LiverToxMatch",
    "LiverToxMatcher",
]


MATCHING_STOPWORDS = BASE_MATCHING_STOPWORDS | {
    "combo",
    "combination",
    "of",
    "or",
    "patch",
    "po",
}


@dataclass(slots=True)
class MonographRecord:
    nbk_id: str
    drug_name: str
    normalized_name: str
    excerpt: str | None
    synonyms: dict[str, str]
    tokens: set[str]


###############################################################################
@dataclass(slots=True)
class LiverToxMatch:
    nbk_id: str
    matched_name: str
    confidence: float
    reason: str
    notes: list[str]
    record: MonographRecord | None = None


###############################################################################
class LiverToxMatcher:
    DIRECT_CONFIDENCE = 1.0
    MASTER_CONFIDENCE = 0.92
    SYNONYM_CONFIDENCE = 0.90
    PARTIAL_CONFIDENCE = 0.86
    FUZZY_CONFIDENCE = 0.84
    FUZZY_THRESHOLD = 0.85
    TOKEN_MAX_FREQUENCY = 3
    MIN_CONFIDENCE = 0.40

    # -------------------------------------------------------------------------
    def __init__(
        self,
        livertox_df: pd.DataFrame,
        master_list_df: pd.DataFrame | None = None,
        *,
        drugs_catalog_df: pd.DataFrame | None = None,
    ) -> None:
        self.livertox_df = livertox_df
        if master_list_df is not None and not master_list_df.empty:
            self.master_list_df = master_list_df.copy()
        else:
            self.master_list_df = self.derive_master_alias_source(livertox_df)
        if drugs_catalog_df is not None and not drugs_catalog_df.empty:
            self.drugs_catalog_df = drugs_catalog_df.copy()
        else:
            self.drugs_catalog_df = None
        self.match_cache: dict[str, LiverToxMatch | None] = {}
        self.records: list[MonographRecord] = []
        self.primary_index: dict[str, MonographRecord] = {}
        self.synonym_index: dict[str, tuple[MonographRecord, str]] = {}
        self.variant_catalog: list[tuple[str, MonographRecord, str, bool]] = []
        self.token_occurrences: dict[str, list[MonographRecord]] = {}
        self.token_index: dict[str, list[MonographRecord]] = {}
        self.brand_index: dict[str, list[tuple[str, str]]] = {}
        self.ingredient_index: dict[str, list[tuple[str, str]]] = {}
        self.rows_by_name: dict[str, dict[str, Any]] = {}
        self.catalog_alias_index: dict[str, set[str]] = {}
        self.build_records()
        self.build_master_list_aliases()
        self.build_catalog_aliases()
        self.finalize_token_index()

    # -------------------------------------------------------------------------
    async def match_drug_names(
        self, patient_drugs: list[str]
    ) -> list[LiverToxMatch | None]:
        total = len(patient_drugs)       
        results: list[LiverToxMatch | None] = [None] * total   
        for idx, name in enumerate(patient_drugs):
            normalized = self.normalize_name(name)
            if not normalized:
                continue
            cached = self.match_cache.get(normalized)
            if cached is not None or normalized in self.match_cache:
                results[idx] = cached
                continue
            lookup = self.match_query(normalized)
            if lookup is None:
                self.match_cache[normalized] = None
                continue
            record, confidence, reason, notes = lookup
            match = self.create_match(record, confidence, reason, notes)
            self.match_cache[normalized] = match
            results[idx] = match
        return results

    # -------------------------------------------------------------------------
    def build_patient_mapping(
        self,
        patient_drugs: list[str],
        matches: list[LiverToxMatch | None],
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
                        or self.normalize_name(match.record.drug_name)
                    )
                if not normalized_key:
                    normalized_key = self.normalize_name(match.matched_name)
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
            entries.append(
                {
                    "drug_name": original,
                    "matched_livertox_row": row_data,
                    "extracted_excerpts": unique_excerpts,
                }
            )
        return entries

    # -------------------------------------------------------------------------
    def match_query(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        candidates = self.collect_catalog_candidates(normalized_query)
        if not candidates:
            return None

        for normalized_value, alias_value, from_catalog in candidates:
            direct = self.match_primary(normalized_value)
            if direct is not None:
                return self.annotate_catalog_match(direct, from_catalog, alias_value)

        synonym_matches: list[
            tuple[tuple[MonographRecord, float, str, list[str]], bool, str]
        ] = []
        for normalized_value, alias_value, from_catalog in candidates:
            synonym = self.match_synonym(normalized_value)
            if synonym is not None:
                synonym_matches.append((synonym, from_catalog, alias_value))

        master_matches: list[
            tuple[tuple[MonographRecord, float, str, list[str]], bool, str]
        ] = []
        for normalized_value, alias_value, from_catalog in candidates:
            master = self.match_master_list(normalized_value)
            if master is not None:
                master_matches.append((master, from_catalog, alias_value))

        for master_match in master_matches:
            master_record, _, _, _ = master_match[0]
            for synonym_match in synonym_matches:
                synonym_record, _, _, _ = synonym_match[0]
                if synonym_record.nbk_id == master_record.nbk_id:
                    return self.annotate_catalog_match(
                        master_match[0], master_match[1], master_match[2]
                    )

        if synonym_matches:
            match_result = synonym_matches[0]
            return self.annotate_catalog_match(
                match_result[0], match_result[1], match_result[2]
            )

        if master_matches:
            match_result = master_matches[0]
            return self.annotate_catalog_match(
                match_result[0], match_result[1], match_result[2]
            )

        for normalized_value, alias_value, from_catalog in candidates:
            partial = self.match_partial(normalized_value)
            if partial is not None:
                return self.annotate_catalog_match(partial, from_catalog, alias_value)

        for normalized_value, alias_value, from_catalog in candidates:
            fuzzy = self.match_fuzzy(normalized_value)
            if fuzzy is not None:
                return self.annotate_catalog_match(fuzzy, from_catalog, alias_value)

        return None

    # -------------------------------------------------------------------------
    def collect_catalog_candidates(
        self, normalized_query: str
    ) -> list[tuple[str, str, bool]]:        
        candidates: dict[str, tuple[str, bool]] = {}
        alias_values = self.catalog_alias_index.get(normalized_query)
        if alias_values:
            for alias in sorted(alias_values, key=lambda value: value.casefold()):
                normalized_alias = self.normalize_name(alias)
                if not normalized_alias:
                    continue
                if normalized_alias not in candidates:
                    candidates[normalized_alias] = (alias, True)
        if normalized_query not in candidates:
            candidates[normalized_query] = (normalized_query, False)
        ordered: list[tuple[str, str, bool]] = []
        for normalized_value, (alias_value, from_catalog) in candidates.items():
            ordered.append((normalized_value, alias_value, from_catalog))
        return ordered

    # -------------------------------------------------------------------------
    def annotate_catalog_match(
        self,
        result: tuple[MonographRecord, float, str, list[str]],
        from_catalog: bool,
        alias_value: str,
    ) -> tuple[MonographRecord, float, str, list[str]]:
        record, confidence, reason, notes = result
        updated_notes = list(notes)
        if from_catalog:
            alias_note = self.coerce_text(alias_value)
            if alias_note:
                updated_notes.insert(0, f"catalog_alias='{alias_note}'")
        return record, confidence, reason, updated_notes

    # -------------------------------------------------------------------------
    def match_primary(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        record = self.primary_index.get(normalized_query)
        if record is None:
            return None
        return record, self.DIRECT_CONFIDENCE, "monograph_name", []

    # -------------------------------------------------------------------------
    def match_master_list(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        alias_sources = (
            ("brand", self.brand_index),
            ("ingredient", self.ingredient_index),
        )
        for alias_type, index in alias_sources:
            entries = index.get(normalized_query)
            if not entries:
                continue
            for alias_value, primary_name in entries:
                resolved = self.match_primary_name(primary_name)
                if resolved is None:
                    continue
                record, base_confidence, primary_reason, primary_notes = resolved
                notes = [
                    f"{alias_type}='{alias_value}'",
                    f"drug='{primary_name}'",
                ]
                notes.extend(primary_notes)
                reason = f"{alias_type}_{primary_reason}"
                confidence = min(self.MASTER_CONFIDENCE, base_confidence)
                return record, confidence, reason, notes
        return None

    # -------------------------------------------------------------------------
    def match_synonym(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        alias = self.synonym_index.get(normalized_query)
        if alias is None:
            return None
        record, original = alias
        notes = [f"synonym='{original}'"]
        return record, self.SYNONYM_CONFIDENCE, "synonym_match", notes

    # -------------------------------------------------------------------------
    def match_partial(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        tokens = [
            token for token in normalized_query.split() if self.is_token_valid(token)
        ]
        if not tokens:
            return None
        candidate_scores: dict[str, int] = {}
        record_lookup: dict[str, MonographRecord] = {}
        for token in tokens:
            for record in self.token_index.get(token, []):
                key = record.normalized_name or record.drug_name.lower()
                record_lookup[key] = record
                candidate_scores[key] = candidate_scores.get(key, 0) + 1
        if not candidate_scores:
            return None
        best_key = max(candidate_scores, key=candidate_scores.get)
        best_score = candidate_scores[best_key]
        tied = [key for key, score in candidate_scores.items() if score == best_score]
        if len(tied) != 1:
            return None
        best_record = record_lookup[best_key]
        matched_tokens: list[str] = []
        for token in tokens:
            for candidate in self.token_index.get(token, []):
                key = candidate.normalized_name or candidate.drug_name.lower()
                if key == best_key:
                    matched_tokens.append(token)
                    break
        notes = [f"token='{token}'" for token in sorted(set(matched_tokens))]
        return best_record, self.PARTIAL_CONFIDENCE, "partial_synonym", notes

    # -------------------------------------------------------------------------
    def match_fuzzy(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        if len(normalized_query) < 4:
            return None
        variant = self.find_best_variant(normalized_query)
        if variant is None:
            return None
        record, original, is_primary, score = variant
        reason = "fuzzy_primary" if is_primary else "fuzzy_synonym"
        notes: list[str] = [f"score={score:.2f}"]
        if not is_primary:
            notes.insert(0, f"variant='{original}'")
        return record, max(self.FUZZY_CONFIDENCE, score), reason, notes

    # -------------------------------------------------------------------------
    def match_primary_name(
        self, drug_name: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        normalized_name = self.normalize_name(drug_name)
        if not normalized_name:
            return None
        direct = self.match_primary(normalized_name)
        if direct is not None:
            record, confidence, _, _ = direct
            return record, confidence, "drug_name", []
        alias = self.synonym_index.get(normalized_name)
        if alias is None:
            return None
        record, original = alias
        notes = [f"synonym='{original}'"]
        return record, self.SYNONYM_CONFIDENCE, "drug_synonym", notes

    # -------------------------------------------------------------------------
    def find_best_variant(
        self, normalized_query: str
    ) -> tuple[MonographRecord, str, bool, float] | None:
        best: tuple[MonographRecord, str, bool, float] | None = None
        best_ratio = 0.0
        for candidate, record, original, is_primary in self.variant_catalog:
            ratio = SequenceMatcher(None, normalized_query, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best = (record, original, is_primary, ratio)
        if best is None or best_ratio < self.FUZZY_THRESHOLD:
            return None
        return best

    # -------------------------------------------------------------------------
    def build_records(self) -> None:
        if self.livertox_df is None or self.livertox_df.empty:
            return
        token_occurrences: dict[str, list[MonographRecord]] = {}
        for row in self.livertox_df.itertuples(index=False):
            raw_name = self.coerce_text(getattr(row, "drug_name", None))
            if raw_name is None:
                continue
            normalized_name = self.normalize_name(raw_name)
            if not normalized_name:
                continue
            nbk_raw = getattr(row, "nbk_id", None)
            nbk_id = str(nbk_raw).strip() if nbk_raw not in (None, "") else ""
            if not nbk_id:
                continue
            excerpt = self.coerce_text(getattr(row, "excerpt", None))
            synonyms_value = getattr(row, "synonyms", None)
            synonyms = self.parse_synonyms(synonyms_value)
            tokens = self.collect_tokens(raw_name, list(synonyms.values()))
            record = MonographRecord(
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
            drug_name = self.coerce_text(getattr(row, "drug_name", None))
            if drug_name is None:
                continue
            brand = self.coerce_text(getattr(row, "brand_name", None))
            ingredient = self.coerce_text(getattr(row, "ingredient", None))
            for alias_type, value in ("brand", brand), ("ingredient", ingredient):
                if value is None:
                    continue
                if value.lower() == "not available":
                    continue
                for variant in self.iter_alias_variants(value):
                    normalized_variant = self.normalize_name(variant)
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
    def build_catalog_aliases(self) -> None:
        self.catalog_alias_index = {}
        if self.drugs_catalog_df is None or self.drugs_catalog_df.empty:
            return
        for row in self.drugs_catalog_df.itertuples(index=False):
            alias_values = self.collect_catalog_alias_values(row)
            if not alias_values:
                continue
            for alias in alias_values:
                for variant in self.iter_alias_variants(alias):
                    normalized = self.normalize_name(variant)
                    if not normalized:
                        continue
                    bucket = self.catalog_alias_index.setdefault(normalized, set())
                    bucket.update(alias_values)

    # -------------------------------------------------------------------------
    def collect_catalog_alias_values(self, row) -> set[str]:
        alias_values: set[str] = set()
        raw_name = self.coerce_text(getattr(row, "raw_name", None))
        if raw_name:
            alias_values.add(raw_name)
        concept_name = self.coerce_text(getattr(row, "name", None))
        if concept_name:
            alias_values.add(concept_name)
        for brand in self.parse_catalog_brand_names(getattr(row, "brand_names", None)):
            alias_values.add(brand)
        for synonym in self.parse_catalog_synonyms(getattr(row, "synonyms", None)):
            alias_values.add(synonym)
        cleaned = {value for value in alias_values if value}
        return cleaned

    # -------------------------------------------------------------------------
    def parse_catalog_brand_names(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            segments = re.split(r"[,;/\n]+", value)
        elif isinstance(value, (list, tuple, set)):
            segments = []
            for entry in value:
                segments.extend(re.split(r"[,;/\n]+", str(entry)))
        else:
            segments = [value]
        names: list[str] = []
        for segment in segments:
            text = self.coerce_text(segment)
            if text:
                names.append(text)
        return names

    # -------------------------------------------------------------------------
    def parse_catalog_synonyms(self, value: Any) -> list[str]:
        raw_values = self.extract_synonym_strings(value)
        synonyms: list[str] = []
        for raw in raw_values:
            text = self.coerce_text(raw)
            if text:
                synonyms.append(text)
        return synonyms

    # -------------------------------------------------------------------------
    def finalize_token_index(self) -> None:
        if not self.token_occurrences:
            self.token_index = {}
            return
        filtered: dict[str, list[MonographRecord]] = {}
        for token, records in self.token_occurrences.items():
            if len(records) > self.TOKEN_MAX_FREQUENCY:
                continue
            filtered[token] = sorted(
                records, key=lambda record: record.drug_name.lower()
            )
        self.token_index = filtered

    # -------------------------------------------------------------------------
    def coerce_text(self, value: Any) -> str | None:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        if isinstance(value, str):
            text = value.strip()
            return text or None
        text = str(value).strip()
        return text or None

    # -------------------------------------------------------------------------
    def iter_alias_variants(self, value: str) -> list[str]:
        normalized_value = self.normalize_whitespace(value)
        if not normalized_value:
            return []
        variants: set[str] = {normalized_value}
        for segment in re.split(r"[;,/\n]+", value):
            candidate = self.normalize_whitespace(segment)
            if candidate:
                variants.add(candidate)
        return list(variants)

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
    def parse_synonyms(self, value: Any) -> dict[str, str]:
        synonyms: dict[str, str] = {}
        raw_values = self.extract_synonym_strings(value)
        if not raw_values:
            text = self.coerce_text(value)
            if text is None:
                return {}
            raw_values = [text]
        for raw in raw_values:
            text = self.coerce_text(raw)
            if text is None:
                continue
            for candidate in re.split(r"[;,/\n]+", text):
                for variant in self.expand_variant(candidate):
                    normalized = self.normalize_name(variant)
                    if not normalized:
                        continue
                    if normalized in MATCHING_STOPWORDS:
                        continue
                    if len(normalized) < 4 and " " not in normalized:
                        continue
                    if normalized not in synonyms:
                        synonyms[normalized] = variant
        return synonyms

    # -------------------------------------------------------------------------
    def extract_synonym_strings(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, dict):
            collected: list[str] = []
            for entry in value.values():
                collected.extend(self.extract_synonym_strings(entry))
            return collected
        if isinstance(value, (list, tuple, set)):
            collected: list[str] = []
            for entry in value:
                collected.extend(self.extract_synonym_strings(entry))
            return collected
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
                parsed = self.try_parse_json(stripped)
                if isinstance(parsed, dict) or isinstance(parsed, list):
                    return self.extract_synonym_strings(parsed)
            return [value]
        text = self.coerce_text(value)
        if text is None:
            return []
        return self.extract_synonym_strings(text)

    # -------------------------------------------------------------------------
    def try_parse_json(self, value: str) -> Any:
        if not value:
            return None
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None

    # -------------------------------------------------------------------------
    def expand_variant(self, value: str) -> list[str]:
        normalized = self.normalize_whitespace(value)
        if not normalized:
            return []
        variants = {normalized}
        for segment in re.split(r"[()]", normalized):
            candidate = segment.strip(" -")
            if candidate:
                variants.add(candidate)
        return list(variants)

    # -------------------------------------------------------------------------
    def collect_tokens(self, primary: str, synonyms: list[str]) -> set[str]:
        tokens: set[str] = set()
        for source in [primary, *synonyms]:
            tokens.update(self.tokenize(source))
        return tokens

    # -------------------------------------------------------------------------
    def tokenize(self, value: str) -> set[str]:
        normalized = self.normalize_name(value)
        if not normalized:
            return set()
        return {token for token in normalized.split() if self.is_token_valid(token)}

    # -------------------------------------------------------------------------
    def is_token_valid(self, token: str) -> bool:
        if len(token) < 4:
            return False
        if token in MATCHING_STOPWORDS:
            return False
        return not token.isdigit()

    # -------------------------------------------------------------------------
    def normalize_whitespace(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    # -------------------------------------------------------------------------
    def ensure_row_index(self) -> dict[str, dict[str, Any]]:
        if self.rows_by_name:
            return self.rows_by_name
        if self.livertox_df is None or self.livertox_df.empty:
            return {}
        index: dict[str, dict[str, Any]] = {}
        for row in self.livertox_df.to_dict(orient="records"):
            drug_name = self.coerce_text(row.get("drug_name"))
            if drug_name is None:
                continue
            normalized = self.normalize_name(drug_name)
            if not normalized:
                continue
            index[normalized] = row
        self.rows_by_name = index
        return self.rows_by_name

    # -------------------------------------------------------------------------
    def create_match(
        self,
        record: MonographRecord,
        confidence: float,
        reason: str,
        notes: list[str] | None,
    ) -> LiverToxMatch:
        normalized_confidence = round(min(max(confidence, self.MIN_CONFIDENCE), 1.0), 2)
        cleaned_notes = list(dict.fromkeys(note for note in (notes or []) if note))
        return LiverToxMatch(
            nbk_id=record.nbk_id,
            matched_name=record.drug_name,
            confidence=normalized_confidence,
            reason=reason,
            notes=cleaned_notes,
            record=record,
        )

    # -------------------------------------------------------------------------
    def normalize_name(self, name: str) -> str:
        normalized = (
            unicodedata.normalize("NFKD", name)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized
