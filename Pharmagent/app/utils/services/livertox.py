from __future__ import annotations

import asyncio
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

import pandas as pd

from Pharmagent.app.api.models.prompts import (
    LIVERTOX_MATCH_LIST_USER_PROMPT,
    LIVERTOX_MATCH_SYSTEM_PROMPT,
)
from Pharmagent.app.api.models.providers import initialize_llm_client
from Pharmagent.app.api.schemas.clinical import LiverToxBatchMatchSuggestion
from Pharmagent.app.configurations import ClientRuntimeConfig
from Pharmagent.app.constants import (
    LIVERTOX_LLM_TIMEOUT_SECONDS,
    LIVERTOX_SKIP_DETERMINISTIC_RATIO,
    LIVERTOX_YIELD_INTERVAL,
    LLM_NULL_MATCH_NAMES,
    DEFAULT_LLM_TIMEOUT_SECONDS,
)
from Pharmagent.app.logger import logger


ALIAS_STOPWORDS = {"and", "or", "with", "plus", "the", "of"}


###############################################################################
@dataclass(slots=True)
class MonographRecord:
    nbk_id: str
    drug_name: str
    normalized_name: str
    tokens: set[str]
    excerpt: str | None
    aliases: dict[str, str]


###############################################################################
@dataclass(slots=True)
class LiverToxLookupEntry:
    record: MonographRecord
    primary: str
    aliases: dict[str, str]


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
    ALIAS_CONFIDENCE = 0.95
    MIN_CONFIDENCE = 0.40
    DETERMINISTIC_THRESHOLD = 0.86
    LLM_DEFAULT_CONFIDENCE = 0.65
    LLM_TIMEOUT_SECONDS = LIVERTOX_LLM_TIMEOUT_SECONDS
    YIELD_INTERVAL = LIVERTOX_YIELD_INTERVAL
    DETERMINISTIC_SKIP_RATIO = LIVERTOX_SKIP_DETERMINISTIC_RATIO

    # -------------------------------------------------------------------------
    def __init__(
        self,
        livertox_df: pd.DataFrame,
        *,
        llm_client: Any | None = None,
    ) -> None:
        self.livertox_df = livertox_df
        self.llm_client = llm_client or initialize_llm_client(
            purpose="parser", timeout_s=DEFAULT_LLM_TIMEOUT_SECONDS
        )
        self.match_cache: dict[str, LiverToxMatch | None] = {}
        self.records: list[MonographRecord] = []
        self.records_by_normalized: dict[str, MonographRecord] = {}
        self.lookup_entries: list[LiverToxLookupEntry] = []
        self.primary_lookup: dict[str, LiverToxLookupEntry] = {}
        self.alias_records_by_normalized: dict[
            str, tuple[LiverToxLookupEntry, str]
        ] = {}
        self.rows_by_nbk: dict[str, dict[str, Any]] = {}
        self.candidate_prompt_block: str | None = None
        self._build_records()

    # -------------------------------------------------------------------------
    async def match_drug_names(
        self,
        patient_drugs: list[str],
        *,
        livertox_drugs: list[LiverToxLookupEntry] | None = None,
    ) -> list[LiverToxMatch | None]:
        total = len(patient_drugs)
        if total == 0:
            return []
        results: list[LiverToxMatch | None] = [None] * total
        if not self.records:
            return results

        # Step 1: normalize input names once and reuse throughout the flow.
        normalized_queries = [self._normalize_name(name) for name in patient_drugs]
        lookup_entries = livertox_drugs or self.lookup_entries
        unresolved_indices: list[int] = []
        deterministic_matches = 0
        eligible_total = 0

        # Step 2: attempt cache hits and deterministic matches before any LLM call.
        for idx, normalized in enumerate(normalized_queries):
            if idx and idx % self.YIELD_INTERVAL == 0:
                await asyncio.sleep(0)
            if not normalized:
                unresolved_indices.append(idx)
                continue
            eligible_total += 1
            cached = self.match_cache.get(normalized)
            if cached is not None:
                results[idx] = cached
                if cached.reason != "llm_fallback":
                    deterministic_matches += 1
                else:
                    unresolved_indices.append(idx)
                continue
            deterministic = self._deterministic_lookup(
                normalized,
                lookup_entries=lookup_entries,
            )
            if deterministic is None:
                self.match_cache.setdefault(normalized, None)
                unresolved_indices.append(idx)
                continue
            record, confidence, reason, notes = deterministic
            match = self._create_match(record, confidence, reason, notes)
            self.match_cache[normalized] = match
            results[idx] = match
            if reason != "llm_fallback":
                deterministic_matches += 1
            else:
                unresolved_indices.append(idx)

        if not unresolved_indices:
            return results

        # Step 3: fall back to the language model only when deterministic coverage is low.
        deterministic_ratio = deterministic_matches / max(eligible_total, 1)
        if deterministic_ratio >= self.DETERMINISTIC_SKIP_RATIO:
            return results

        fallback_matches = await self._llm_batch_match_lookup(
            patient_drugs,
            normalized_queries=normalized_queries,
        )

        # Step 4: merge LLM suggestions back into the response cache.
        for idx in unresolved_indices:
            normalized = normalized_queries[idx]
            if not normalized:
                continue
            fallback = fallback_matches[idx] if idx < len(fallback_matches) else None
            if fallback is None:
                self.match_cache[normalized] = None
                continue
            record, confidence, reason, notes = fallback
            match = self._create_match(record, confidence, reason, notes)
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
        lookup_entries: list[LiverToxLookupEntry] = []
        primary_lookup: dict[str, LiverToxLookupEntry] = {}
        alias_lookup: dict[str, tuple[LiverToxLookupEntry, str]] = {}
        # Step A: flatten aliases coming from the dataset columns.
        for row in self.livertox_df.itertuples(index=False):
            raw_name = str(getattr(row, "drug_name", "") or "").strip()
            if not raw_name:
                continue
            normalized_name = self._normalize_name(raw_name)
            if not normalized_name:
                continue
            nbk_raw = getattr(row, "nbk_id", None)
            nbk_id = str(nbk_raw).strip() if nbk_raw is not None else ""
            excerpt_raw = getattr(row, "excerpt", None)
            excerpt = str(excerpt_raw) if excerpt_raw not in (None, "") else None
            primary_variant = self._normalize_name(raw_name.split("(")[0])
            alias_values = self._collect_aliases(raw_name, row)
            alias_map: dict[str, str] = {}
            tokens = {token for token in normalized_name.split() if token}
            if primary_variant and primary_variant != normalized_name:
                tokens.update(token for token in primary_variant.split() if token)
            for alias in alias_values:
                normalized_alias = self._normalize_name(alias)
                if not normalized_alias or normalized_alias == normalized_name:
                    continue
                alias_map.setdefault(normalized_alias, alias)
                tokens.update(token for token in normalized_alias.split() if token)
            record = MonographRecord(
                nbk_id=nbk_id,
                drug_name=raw_name,
                normalized_name=normalized_name,
                tokens=tokens,
                excerpt=excerpt,
                aliases=alias_map,
            )
            processed.append(record)
            normalized_map.setdefault(normalized_name, record)
            if primary_variant:
                normalized_map.setdefault(primary_variant, record)
            lookup_entry = LiverToxLookupEntry(
                record=record,
                primary=normalized_name,
                aliases=alias_map,
            )
            lookup_entries.append(lookup_entry)
            if normalized_name not in primary_lookup:
                primary_lookup[normalized_name] = lookup_entry
            if primary_variant and primary_variant not in primary_lookup:
                primary_lookup[primary_variant] = lookup_entry
            for normalized_alias, original_alias in alias_map.items():
                alias_lookup.setdefault(normalized_alias, (lookup_entry, original_alias))
        if not processed:
            return
        processed.sort(key=lambda item: item.drug_name.lower())
        self.records = processed
        self.records_by_normalized = {
            key: value for key, value in normalized_map.items() if value is not None
        }
        self.lookup_entries = lookup_entries
        self.primary_lookup = primary_lookup
        self.alias_records_by_normalized = alias_lookup
        self.candidate_prompt_block = "\n".join(
            f"- {record.drug_name}" for record in self.records
        )

    # -------------------------------------------------------------------------
    def _collect_aliases(self, raw_name: str, row: Any) -> set[str]:
        aliases: set[str] = set()
        if "(" in raw_name and ")" in raw_name:
            alias_section = raw_name.split("(", 1)[1].split(")", 1)[0]
            aliases.update(self._extract_aliases(alias_section))
        for column in ("additional_names", "synonyms"):
            value = getattr(row, column, None)
            aliases.update(self._extract_aliases(value))
        expanded: set[str] = set()
        for alias in aliases:
            expanded.update(self._expand_alias(alias))
        return {alias for alias in expanded if alias}

    # -------------------------------------------------------------------------
    def _extract_aliases(self, value: Any) -> set[str]:
        if value in (None, ""):
            return set()
        if isinstance(value, float) and pd.isna(value):
            return set()
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return set()
        cleaned = re.sub(r"[\n;|]+", ",", text)
        parts = [part.strip() for part in cleaned.split(",") if part.strip()]
        return set(parts)

    # -------------------------------------------------------------------------
    def _expand_alias(self, alias: str) -> set[str]:
        alias = alias.strip()
        if not alias:
            return set()
        variants: set[str] = {alias}
        normalized = re.sub(r"[\\/]+", ",", alias)
        normalized = re.sub(r"\bwith\b|\band\b|\bor\b", ",", normalized, flags=re.I)
        for chunk in normalized.split(","):
            cleaned = chunk.strip(" -")
            if cleaned:
                variants.add(cleaned)
        tokens = [
            token.strip()
            for token in re.split(r"[\s\-+/]+", alias)
            if token.strip()
        ]
        filtered_tokens = [
            token
            for token in tokens
            if token.lower() not in ALIAS_STOPWORDS and len(token) > 2
        ]
        variants.update(filtered_tokens)
        if len(filtered_tokens) > 1:
            variants.add(" ".join(filtered_tokens))
            variants.add("".join(filtered_tokens))
        return variants

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
    def _deterministic_lookup(
        self,
        normalized_query: str,
        *,
        lookup_entries: list[LiverToxLookupEntry],
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        if not normalized_query:
            return None
        primary_entry = self.primary_lookup.get(normalized_query)
        if primary_entry is not None:
            return primary_entry.record, self.DIRECT_CONFIDENCE, "direct_match", []
        alias_entry = self.alias_records_by_normalized.get(normalized_query)
        if alias_entry is not None:
            entry, alias_original = alias_entry
            alias_note = None
            if alias_original and alias_original.lower() != entry.record.drug_name.lower():
                alias_note = f"alias='{alias_original}'"
            notes = [alias_note] if alias_note else []
            return entry.record, self.ALIAS_CONFIDENCE, "alias_match", notes
        return self._fuzzy_lookup(normalized_query, lookup_entries)

    # -------------------------------------------------------------------------
    def _find_best_record(
        self, normalized_value: str
    ) -> tuple[MonographRecord, float] | None:
        best_entry = self._find_best_entry(normalized_value)
        if best_entry is None:
            return None
        entry, score = best_entry
        return entry.record, score

    # -------------------------------------------------------------------------
    def _fuzzy_lookup(
        self,
        normalized_query: str,
        lookup_entries: list[LiverToxLookupEntry],
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        best_entry = self._find_best_entry(normalized_query, lookup_entries)
        if best_entry is None:
            return None
        entry, score = best_entry
        if score < self.DETERMINISTIC_THRESHOLD:
            return None
        notes = [f"fuzzy_score={score:.2f}"]
        return entry.record, score, "fuzzy_match", notes

    # -------------------------------------------------------------------------
    def _find_best_entry(
        self,
        normalized_value: str,
        lookup_entries: list[LiverToxLookupEntry] | None = None,
    ) -> tuple[LiverToxLookupEntry, float] | None:
        entries = lookup_entries or self.lookup_entries
        best_score = 0.0
        best_entry: LiverToxLookupEntry | None = None
        value_tokens = {token for token in normalized_value.split() if token}
        for entry in entries:
            score = self._score_match(value_tokens, entry.record.tokens)
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_entry is None:
            return None
        return best_entry, best_score

    # -------------------------------------------------------------------------
    def _score_match(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        overlap = len(left & right)
        union = len(left | right)
        if union == 0:
            return 0.0
        return overlap / union

    # -------------------------------------------------------------------------
    async def _llm_batch_match_lookup(
        self,
        patient_drugs: list[str],
        *,
        normalized_queries: list[str],
    ) -> list[tuple[MonographRecord, float, str, list[str]] | None]:
        total = len(patient_drugs)
        if total == 0 or not self.records:
            return []
        # The prompt block is cached so repeated calls do not rebuild the monograph list.
        candidate_block = self.candidate_prompt_block or ""
        if not candidate_block:
            return [None] * total
        drugs_block = "\n".join(f"- {name}" if name else "-" for name in patient_drugs)
        prompt = LIVERTOX_MATCH_LIST_USER_PROMPT.format(
            patient_drugs=drugs_block,
            candidates=candidate_block,
        )
        model_name = ClientRuntimeConfig.get_parsing_model()
        logger.debug(
            "Dispatching batch LLM match for %s drugs against %s candidates (prompt length=%s chars)",
            total,
            len(self.records),
            len(prompt),
        )
        start_time = time.perf_counter()
        try:
            suggestion = await asyncio.wait_for(
                self.llm_client.llm_structured_call(
                    model=model_name,
                    system_prompt=LIVERTOX_MATCH_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    schema=LiverToxBatchMatchSuggestion,
                    temperature=0.0,
                ),
                timeout=self.LLM_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            duration = time.perf_counter() - start_time
            logger.error(
                "LLM batch match timed out after %.2fs using model '%s'",
                duration,
                model_name,
            )
            return [None] * total
        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - start_time
            logger.error(
                "LLM batch match failed after %.2fs: %s",
                duration,
                exc,
            )
            return [None] * total
        duration = time.perf_counter() - start_time
        logger.debug(
            "Batch LLM matching completed in %.2fs using model '%s'",
            duration,
            model_name,
        )
        matches = suggestion.matches if suggestion else []
        if not matches:
            logger.warning(
                "LLM returned no matches for %s patient drugs",
                total,
            )
            return [None] * total
        if len(matches) != total:
            logger.warning(
                "LLM returned %s matches for %s patient drugs",
                len(matches),
                total,
            )
        normalized_to_items: dict[str, list[Any]] = {}
        for item in matches:
            normalized_drug = self._normalize_name(getattr(item, "drug_name", "") or "")
            if not normalized_drug:
                continue
            bucket = normalized_to_items.setdefault(normalized_drug, [])
            bucket.append(item)
        results: list[tuple[MonographRecord, float, str, list[str]] | None] = [
            None
        ] * total
        for idx, original in enumerate(patient_drugs):
            normalized_query = (
                normalized_queries[idx] if idx < len(normalized_queries) else ""
            )
            if not normalized_query:
                continue
            item: Any | None = None
            if idx < len(matches):
                candidate = matches[idx]
                candidate_normalized = self._normalize_name(
                    getattr(candidate, "drug_name", "") or ""
                )
                if candidate_normalized == normalized_query:
                    item = candidate
                    bucket = normalized_to_items.get(normalized_query)
                    if bucket:
                        try:
                            bucket.remove(candidate)
                        except ValueError:
                            pass
            if item is None:
                bucket = normalized_to_items.get(normalized_query)
                if bucket:
                    item = bucket.pop(0)
            if item is None:
                logger.debug(
                    "LLM did not return a usable match for '%s'",
                    original,
                )
                continue
            match_name = (getattr(item, "match_name", "") or "").strip()
            normalized_match = self._normalize_name(match_name)
            if normalized_match in LLM_NULL_MATCH_NAMES:
                logger.debug(
                    "LLM explicitly reported no match for '%s'",
                    original,
                )
                continue
            if not normalized_match:
                continue
            record_entry = self.primary_lookup.get(normalized_match)
            record: MonographRecord | None = (
                record_entry.record if record_entry is not None else None
            )
            confidence_raw = getattr(item, "confidence", None)
            confidence = (
                float(confidence_raw)
                if confidence_raw is not None
                else self.LLM_DEFAULT_CONFIDENCE
            )
            notes: list[str] = [f"LLM selected '{match_name}' for '{original}'"]
            rationale = (getattr(item, "rationale", "") or "").strip()
            if rationale:
                notes.append(rationale)
            if record is None:
                alias_entry = self.alias_records_by_normalized.get(normalized_match)
                if alias_entry is not None:
                    entry, alias_original = alias_entry
                    record = entry.record
                    if alias_original:
                        notes.append(f"alias='{alias_original}'")
            if record is None:
                best = self._find_best_record(normalized_match)
                if best is None and normalized_match != normalized_query:
                    best = self._find_best_record(normalized_query)
                if best is None:
                    logger.debug(
                        "Unable to map LLM suggestion '%s' for '%s' to a monograph",
                        match_name,
                        original,
                    )
                    continue
                record, score = best
                notes.append(
                    f"Mapped suggestion to '{record.drug_name}' (score={score:.2f})"
                )
                confidence = max(confidence, score)
            results[idx] = (record, confidence, "llm_fallback", notes)
        return results

    # -------------------------------------------------------------------------
    def _normalize_name(self, name: str) -> str:
        normalized = (
            unicodedata.normalize("NFKD", name)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        # Always lowercase before stripping punctuation to keep matching case-insensitive.
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized
