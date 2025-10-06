from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import pandas as pd


MATCHING_STOPWORDS = {
    "and",
    "any",
    "apply",
    "caps",
    "capsule",
    "capsules",
    "chewable",
    "cream",
    "dose",
    "doses",
    "drink",
    "drops",
    "elixir",
    "enteric",
    "extended",
    "foam",
    "for",
    "free",
    "gel",
    "granules",
    "im",
    "inj",
    "injection",
    "intramuscular",
    "intravenous",
    "iv",
    "kit",
    "liquid",
    "lotion",
    "mg",
    "ml",
    "nasal",
    "ointment",
    "ophthalmic",
    "oral",
    "plus",
    "pack",
    "packet",
    "packets",
    "combo",
    "combination",
    "of",
    "or",
    "patch",
    "po",
    "powder",
    "prefilled",
    "release",
    "sc",
    "sol",
    "solution",
    "soln",
    "spray",
    "sterile",
    "subcutaneous",
    "suppository",
    "susp",
    "suspension",
    "sustained",
    "tablet",
    "tablets",
    "syringe",
    "syrup",
    "the",
    "that",
    "these",
    "this",
    "treat",
    "treated",
    "treating",
    "treatment",
    "topical",
    "used",
    "using",
    "vial",
    "with",
    "without",
}
###############################################################################
@dataclass(slots=True, eq=False)
class MonographRecord:
    nbk_id: str
    drug_name: str
    normalized_name: str
    excerpt: str | None


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
###############################################################################
@dataclass(slots=True)
class AliasEntry:
    record: MonographRecord
    alias_type: str
    display_name: str


###############################################################################
class LiverToxMatcher:
    DIRECT_CONFIDENCE = 1.0
    ALIAS_CONFIDENCE = 0.95
    MASTER_ALIAS_CONFIDENCE = 0.9
    FUZZY_MONOGRAPH_CONFIDENCE = 0.88
    FUZZY_ALIAS_CONFIDENCE = 0.85
    MIN_CONFIDENCE = 0.40
    FUZZY_CUTOFF = 0.84
    ALIAS_FUZZY_CUTOFF = 0.82

    # -------------------------------------------------------------------------
    def __init__(
        self,
        livertox_df: pd.DataFrame,
        *,
        master_list_df: pd.DataFrame | None = None,
    ) -> None:
        self.livertox_df = livertox_df
        self.master_list_df = master_list_df
        self.match_cache: dict[str, LiverToxMatch | None] = {}
        self.records: list[MonographRecord] = []
        self.records_by_normalized: dict[str, MonographRecord] = {}
        self.rows_by_nbk: dict[str, dict[str, Any]] = {}
        self.alias_index: dict[str, list[AliasEntry]] = {}
        self.alias_keys: list[str] = []
        self.master_alias_index: dict[str, list[AliasEntry]] = {}
        self.master_alias_keys: list[str] = []
        self._build_records()
        self._build_master_list_aliases()

    # -------------------------------------------------------------------------
    async def match_drug_names(
        self, patient_drugs: list[str]
    ) -> list[LiverToxMatch | None]:
        if not patient_drugs:
            return []
        if not self.records:
            return [None] * len(patient_drugs)
        results: list[LiverToxMatch | None] = []
        for name in patient_drugs:
            normalized = self._normalize_name(name) if name else ""
            if not normalized:
                results.append(None)
                continue
            if normalized in self.match_cache:
                results.append(self.match_cache[normalized])
                continue
            match = self._deterministic_lookup(normalized)
            self.match_cache[normalized] = match
            results.append(match)
        return results

    # -------------------------------------------------------------------------
    def build_patient_mapping(
        self,
        patient_drugs: list[str],
        matches: list[LiverToxMatch | None],
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        nbk_to_row = self._ensure_row_index()
        for original, match in zip(patient_drugs, matches, strict=False):
            row_data: dict[str, Any] | None = None
            excerpts: list[str] = []
            if match is not None:
                row_data = dict(nbk_to_row.get(match.nbk_id, {})) or None
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
    def _build_records(self) -> None:
        if self.livertox_df is None or self.livertox_df.empty:
            return
        processed: list[MonographRecord] = []
        normalized_map: dict[str, MonographRecord] = {}
        for row in self.livertox_df.itertuples(index=False):
            raw_name = str(getattr(row, "drug_name", "") or "").strip()
            if not raw_name:
                continue
            normalized_name = self._normalize_name(raw_name)
            if not normalized_name:
                continue
            primary_variant = self._normalize_name(raw_name.split("(")[0])
            nbk_raw = getattr(row, "nbk_id", None)
            nbk_id = str(nbk_raw).strip() if nbk_raw not in (None, "") else ""
            excerpt_value = self._coerce_text(getattr(row, "excerpt", None))
            synonyms = self._parse_synonyms(getattr(row, "synonyms", None))
            record = MonographRecord(
                nbk_id=nbk_id,
                drug_name=raw_name,
                normalized_name=normalized_name,
                excerpt=excerpt_value,
            )
            processed.append(record)
            normalized_map.setdefault(normalized_name, record)
            if primary_variant and primary_variant != normalized_name:
                normalized_map.setdefault(primary_variant, record)
            for synonym in synonyms:
                normalized_synonym = self._normalize_name(synonym)
                if not normalized_synonym or normalized_synonym == normalized_name:
                    continue
                normalized_map.setdefault(normalized_synonym, record)
                self._register_alias(synonym, "synonym", record)
        if not processed:
            return
        processed.sort(key=lambda item: item.drug_name.lower())
        self.records = processed
        self.records_by_normalized = {
            key: value for key, value in normalized_map.items() if value is not None
        }

    # -------------------------------------------------------------------------
    def _build_master_list_aliases(self) -> None:
        if self.master_list_df is None or self.master_list_df.empty:
            return
        for row in self.master_list_df.itertuples(index=False):
            chapter = self._coerce_text(getattr(row, "chapter_title", None))
            record = self._resolve_chapter_record(chapter)
            if record is None:
                continue
            if chapter:
                self._register_alias(chapter, "chapter", record)
            brand = self._coerce_text(getattr(row, "brand_name", None))
            if brand:
                self._register_alias(brand, "brand", record)
            ingredient = self._coerce_text(getattr(row, "ingredient", None))
            if ingredient:
                self._register_alias(ingredient, "ingredient", record)

    # -------------------------------------------------------------------------
    def _register_alias(
        self, value: str, alias_type: str, record: MonographRecord
    ) -> None:
        normalized = self._normalize_alias_value(value, alias_type)
        if not normalized:
            return
        entry = AliasEntry(record=record, alias_type=alias_type, display_name=value)
        bucket = self.alias_index.setdefault(normalized, [])
        is_master_alias = alias_type in {"brand", "ingredient", "chapter"}
        if entry not in bucket:
            bucket.append(entry)
            if normalized not in self.alias_keys:
                self.alias_keys.append(normalized)
        if is_master_alias:
            master_bucket = self.master_alias_index.setdefault(normalized, [])
            if entry not in master_bucket:
                master_bucket.append(entry)
                if normalized not in self.master_alias_keys:
                    self.master_alias_keys.append(normalized)

    # -------------------------------------------------------------------------
    def _resolve_chapter_record(
        self, chapter_title: str | None
    ) -> MonographRecord | None:
        if not chapter_title:
            return None
        normalized = self._normalize_name(chapter_title)
        if not normalized:
            return None
        direct = self.records_by_normalized.get(normalized)
        if direct is not None:
            return direct
        fuzzy = self._find_fuzzy_monogram(normalized, cutoff=self.FUZZY_CUTOFF)
        if fuzzy is None:
            return None
        record, _ = fuzzy
        return record

    # -------------------------------------------------------------------------
    def _coerce_text(self, value: Any) -> str | None:
        if value in (None, ""):
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        text = str(value).strip()
        return text or None

    # -------------------------------------------------------------------------
    def _normalize_alias_value(self, value: str, alias_type: str) -> str | None:
        normalized = self._normalize_name(value)
        if not normalized:
            return None
        tokens = normalized.split()
        if not tokens:
            return None
        meaningful_tokens = [
            token
            for token in tokens
            if len(token) >= 3 and token not in MATCHING_STOPWORDS
        ]
        if not meaningful_tokens:
            return None
        return normalized

    # -------------------------------------------------------------------------
    def _parse_synonyms(self, value: Any) -> list[str]:
        text = self._coerce_text(value)
        if text is None:
            return []
        sanitized = text.replace(";", ",").replace("/", ",").replace("\n", ",")
        sanitized = re.sub(r"\b(?:and|or)\b", ",", sanitized, flags=re.IGNORECASE)
        candidates = [segment.strip() for segment in sanitized.split(",")]
        candidates.extend(
            segment.strip() for segment in re.findall(r"\(([^)]+)\)", text)
        )
        candidates.extend(
            segment.strip() for segment in re.findall(r"\[([^\]]+)\]", text)
        )
        synonyms: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            cleaned = re.sub(r"^[\[{(]+", "", candidate)
            cleaned = re.sub(r"[\]})]+$", "", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if len(cleaned) < 3:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            if self._normalize_alias_value(cleaned, "synonym") is None:
                continue
            synonyms.append(cleaned)
        return synonyms

    # -------------------------------------------------------------------------
    def _find_fuzzy_monogram(
        self, normalized_query: str, *, cutoff: float | None = None
    ) -> tuple[MonographRecord, str] | None:
        if not normalized_query or not self.records_by_normalized:
            return None
        keys = list(self.records_by_normalized.keys())
        threshold = cutoff if cutoff is not None else self.FUZZY_CUTOFF
        match = self._best_fuzzy_key(normalized_query, keys, threshold)
        if match is None:
            return None
        key = match
        record = self.records_by_normalized.get(key)
        if record is None:
            return None
        return record, key

    # -------------------------------------------------------------------------
    def _find_fuzzy_alias_key(
        self,
        normalized_query: str,
        *,
        keys: list[str],
        alias_index: dict[str, list[AliasEntry]],
        allowed_types: set[str] | None,
    ) -> str | None:
        if not normalized_query or not keys:
            return None
        match = self._best_fuzzy_key(
            normalized_query, keys, self.ALIAS_FUZZY_CUTOFF
        )
        if match is None:
            return None
        entries = alias_index.get(match)
        if not entries:
            return None
        candidate = self._select_alias(entries, allowed_types)
        if candidate is None:
            return None
        return match

    # -------------------------------------------------------------------------
    def _best_fuzzy_key(
        self,
        normalized_query: str,
        keys: list[str],
        cutoff: float,
    ) -> str | None:
        if not normalized_query or not keys:
            return None
        best_key: str | None = None
        best_ratio = 0.0
        first_char = normalized_query[0]
        enforce_first_char = first_char.isalpha()
        for key in keys:
            if abs(len(key) - len(normalized_query)) > 3:
                continue
            if (
                enforce_first_char
                and key
                and key[0].isalpha()
                and key[0] != first_char
            ):
                continue
            ratio = difflib.SequenceMatcher(None, normalized_query, key).ratio()
            if ratio < cutoff:
                continue
            if ratio > best_ratio:
                best_ratio = ratio
                best_key = key
        return best_key

    # -------------------------------------------------------------------------
    def _select_alias(
        self,
        entries: list[AliasEntry] | None,
        allowed_types: set[str] | None,
    ) -> AliasEntry | None:
        if not entries:
            return None
        alias_priority = {
            "chapter": 0,
            "brand": 1,
            "ingredient": 2,
            "synonym": 3,
        }
        best_entry: AliasEntry | None = None
        best_priority = 10
        for entry in entries:
            if allowed_types is not None and entry.alias_type not in allowed_types:
                continue
            priority = alias_priority.get(entry.alias_type, 99)
            if best_entry is None or priority < best_priority:
                best_entry = entry
                best_priority = priority
        return best_entry

    # -------------------------------------------------------------------------
    def _alias_metadata(
        self, alias_type: str, *, fuzzy: bool
    ) -> tuple[float, str]:
        if alias_type in {"brand", "ingredient", "chapter"}:
            confidence = (
                self.FUZZY_ALIAS_CONFIDENCE if fuzzy else self.MASTER_ALIAS_CONFIDENCE
            )
        else:
            confidence = self.FUZZY_ALIAS_CONFIDENCE if fuzzy else self.ALIAS_CONFIDENCE
        suffix = "fuzzy" if fuzzy else "alias"
        reason = f"{alias_type}_{suffix}"
        return confidence, reason

    # -------------------------------------------------------------------------
    def _alias_note(self, entry: AliasEntry) -> str:
        return f"{entry.alias_type}='{entry.display_name}'"

    # -------------------------------------------------------------------------
    def _match_alias(
        self,
        normalized_query: str,
        *,
        allow_fuzzy: bool,
        index: dict[str, list[AliasEntry]] | None = None,
        keys: list[str] | None = None,
        allowed_types: set[str] | None = None,
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        alias_index = index if index is not None else self.alias_index
        alias_keys = keys if keys is not None else self.alias_keys
        if not normalized_query or not alias_index:
            return None
        entries = alias_index.get(normalized_query)
        entry = self._select_alias(entries, allowed_types)
        if entry is not None:
            confidence, reason = self._alias_metadata(entry.alias_type, fuzzy=False)
            note = self._alias_note(entry)
            return entry.record, confidence, reason, [note]
        if not allow_fuzzy or not alias_keys:
            return None
        fuzzy_key = self._find_fuzzy_alias_key(
            normalized_query,
            keys=alias_keys,
            alias_index=alias_index,
            allowed_types=allowed_types,
        )
        if fuzzy_key is None:
            return None
        entry = self._select_alias(alias_index.get(fuzzy_key), allowed_types)
        if entry is None:
            return None
        confidence, reason = self._alias_metadata(entry.alias_type, fuzzy=True)
        note = self._alias_note(entry)
        return entry.record, confidence, reason, [note]

    # -------------------------------------------------------------------------
    def _ensure_row_index(self) -> dict[str, dict[str, Any]]:
        if self.rows_by_nbk:
            return self.rows_by_nbk
        if self.livertox_df is None or self.livertox_df.empty:
            return {}
        index: dict[str, dict[str, Any]] = {}
        for row in self.livertox_df.to_dict(orient="records"):
            nbk_id = str(row.get("nbk_id") or "").strip()
            if not nbk_id:
                continue
            index[nbk_id] = row
        self.rows_by_nbk = index
        return self.rows_by_nbk

    # -------------------------------------------------------------------------
    def _create_match(
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
    def _deterministic_lookup(self, normalized_query: str) -> LiverToxMatch | None:
        if not normalized_query:
            return None
        direct = self.records_by_normalized.get(normalized_query)
        if direct is not None:
            return self._create_match(direct, self.DIRECT_CONFIDENCE, "direct_match", [])
        fuzzy = self._find_fuzzy_monogram(normalized_query)
        if fuzzy is not None:
            record, _ = fuzzy
            note = f"matched='{record.drug_name}'"
            return self._create_match(
                record,
                self.FUZZY_MONOGRAPH_CONFIDENCE,
                "fuzzy_match",
                [note],
            )
        master_alias = self._match_alias(
            normalized_query,
            allow_fuzzy=False,
            index=self.master_alias_index,
            keys=self.master_alias_keys,
        )
        if master_alias is not None:
            record, confidence, reason, notes = master_alias
            return self._create_match(record, confidence, reason, notes)
        master_alias_fuzzy = self._match_alias(
            normalized_query,
            allow_fuzzy=True,
            index=self.master_alias_index,
            keys=self.master_alias_keys,
        )
        if master_alias_fuzzy is not None:
            record, confidence, reason, notes = master_alias_fuzzy
            return self._create_match(record, confidence, reason, notes)
        alias_match = self._match_alias(
            normalized_query,
            allow_fuzzy=False,
            allowed_types={"synonym"},
        )
        if alias_match is not None:
            record, confidence, reason, notes = alias_match
            return self._create_match(record, confidence, reason, notes)
        alias_fuzzy = self._match_alias(
            normalized_query,
            allow_fuzzy=True,
            allowed_types={"synonym"},
        )
        if alias_fuzzy is not None:
            record, confidence, reason, notes = alias_fuzzy
            return self._create_match(record, confidence, reason, notes)
        return None

    # -------------------------------------------------------------------------
    def _normalize_name(self, name: str) -> str:
        normalized = (
            unicodedata.normalize("NFKD", name)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized
