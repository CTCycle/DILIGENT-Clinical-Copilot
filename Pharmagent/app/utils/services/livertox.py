from __future__ import annotations

import os
import re
import time
import unicodedata
from collections.abc import Awaitable, Callable
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
from Pharmagent.app.logger import logger

try:
    from langsmith import traceable as langsmith_traceable
except ModuleNotFoundError:
    langsmith_traceable = None  # type: ignore[assignment]
except Exception:  # noqa: BLE001
    langsmith_traceable = None  # type: ignore[assignment]


LANGSMITH_ENV_KEYS = (
    "LANGSMITH_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACE",
    "PHARMAGENT_LANGSMITH_TRACE",
)


# -------------------------------------------------------------------------
def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


LANGSMITH_TRACING_ENABLED = any(
    bool(os.getenv("LANGSMITH_API_KEY"))
    if key == "LANGSMITH_API_KEY"
    else _is_truthy(os.getenv(key))
    for key in LANGSMITH_ENV_KEYS
)
_LANGSMITH_WARNING_EMITTED = False


###############################################################################
@dataclass(slots=True)
class MonographRecord:
    nbk_id: str
    drug_name: str
    normalized_name: str
    tokens: set[str]
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
class LiverToxMatcher:
    DIRECT_CONFIDENCE = 1.0
    ALIAS_CONFIDENCE = 0.95
    MIN_CONFIDENCE = 0.40
    DETERMINISTIC_THRESHOLD = 0.86
    LLM_DEFAULT_CONFIDENCE = 0.65

    # -------------------------------------------------------------------------
    def __init__(
        self,
        livertox_df: pd.DataFrame,
        *,
        llm_client: Any | None = None,
    ) -> None:
        self.livertox_df = livertox_df
        self.llm_client = llm_client or initialize_llm_client(
            purpose="agent", timeout_s=300.0
        )
        self.match_cache: dict[str, LiverToxMatch | None] = {}
        self.records: list[MonographRecord] = []
        self.records_by_normalized: dict[str, MonographRecord] = {}
        self.rows_by_nbk: dict[str, dict[str, Any]] = {}
        self.candidate_prompt_block: str | None = None
        self.langsmith_batch_lookup: Callable[..., Awaitable[Any]] | None = None
        self.langsmith_llm_call: Callable[..., Awaitable[Any]] | None = None
        if LANGSMITH_TRACING_ENABLED and langsmith_traceable is None:
            global _LANGSMITH_WARNING_EMITTED
            if not _LANGSMITH_WARNING_EMITTED:
                logger.warning(
                    "LangSmith tracing requested but the 'langsmith' package is not installed.",
                )
                _LANGSMITH_WARNING_EMITTED = True
        if LANGSMITH_TRACING_ENABLED and langsmith_traceable is not None:
            self.langsmith_batch_lookup = self._wrap_with_langsmith(
                name="livertox_batch_match_lookup",
                run_type="chain",
                target=self._llm_batch_match_lookup_impl,
            )
            self.langsmith_llm_call = self._wrap_with_langsmith(
                name="livertox_batch_llm_structured_call",
                run_type="llm",
                target=self.llm_client.llm_structured_call,
            )
        self._build_records()

    # -------------------------------------------------------------------------
    async def match_drug_names(
        self, patient_drugs: list[str]
    ) -> list[LiverToxMatch | None]:
        total = len(patient_drugs)
        if total == 0:
            return []
        results: list[LiverToxMatch | None] = [None] * total
        if not self.records:
            return results
        normalized_queries = [self._normalize_name(name) for name in patient_drugs]
        unresolved_indices: list[int] = []
        for idx, (original, normalized) in enumerate(
            zip(patient_drugs, normalized_queries, strict=False)
        ):
            if not normalized:
                continue
            cached = self.match_cache.get(normalized)
            if cached is not None or normalized in self.match_cache:
                results[idx] = cached
                continue
            deterministic = self._deterministic_lookup(normalized)
            if deterministic is not None:
                record, confidence, reason, extra_notes = deterministic
                match = self._create_match(record, confidence, reason, extra_notes)
                self.match_cache[normalized] = match
                results[idx] = match
                continue
            unresolved_indices.append(idx)
        if unresolved_indices:
            fallback_matches = await self._llm_batch_match_lookup(
                patient_drugs,
                normalized_queries=normalized_queries,
            )
            for idx in unresolved_indices:
                normalized = normalized_queries[idx]
                if not normalized:
                    continue
                fallback = fallback_matches[idx] if idx < len(fallback_matches) else None
                if fallback is None:
                    self.match_cache[normalized] = None
                    continue
                record, confidence, reason, extra_notes = fallback
                match = self._create_match(record, confidence, reason, extra_notes)
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
            tokens = {token for token in normalized_name.split() if token}
            record = MonographRecord(
                nbk_id=nbk_id,
                drug_name=raw_name,
                normalized_name=normalized_name,
                tokens=tokens,
                excerpt=excerpt,
            )
            processed.append(record)
            if normalized_name not in normalized_map:
                normalized_map[normalized_name] = record
            primary_name = self._normalize_name(raw_name.split("(")[0])
            if primary_name and primary_name not in normalized_map:
                normalized_map[primary_name] = record
            alias_section = None
            if "(" in raw_name and ")" in raw_name:
                alias_section = raw_name.split("(", 1)[1].split(")", 1)[0]
            if alias_section:
                for alias in alias_section.split(","):
                    alias_normalized = self._normalize_name(alias)
                    if alias_normalized and alias_normalized not in normalized_map:
                        normalized_map[alias_normalized] = record
        if not processed:
            return
        processed.sort(key=lambda item: item.drug_name.lower())
        self.records = processed
        self.records_by_normalized = normalized_map
        self.candidate_prompt_block = "\n".join(
            f"- {record.drug_name}" for record in self.records
        )

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
        normalized_confidence = round(
            min(max(confidence, self.MIN_CONFIDENCE), 1.0), 2
        )
        cleaned_notes = list(
            dict.fromkeys(note for note in (notes or []) if note)
        )
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
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        record = self.records_by_normalized.get(normalized_query)
        if record is not None:
            return record, self.DIRECT_CONFIDENCE, "direct_match", []
        query_tokens = {token for token in normalized_query.split() if token}
        best_score = 0.0
        best_record: MonographRecord | None = None
        reason = "fuzzy_match"
        notes: list[str] = []
        for record_candidate in self.records:
            score = self._score_match(query_tokens, record_candidate.tokens)
            if score > best_score:
                best_score = score
                best_record = record_candidate
        if best_record is None or best_score < self.DETERMINISTIC_THRESHOLD:
            alias_match = self._match_alias(normalized_query)
            if alias_match is not None:
                alias_record, alias_notes = alias_match
                return (
                    alias_record,
                    self.ALIAS_CONFIDENCE,
                    "alias_match",
                    alias_notes,
                )
            return None
        notes.append(f"fuzzy_score={best_score:.2f}")
        return best_record, best_score, reason, notes

    # -------------------------------------------------------------------------
    def _match_alias(
        self, normalized_query: str
    ) -> tuple[MonographRecord, list[str]] | None:
        best = self._find_best_record(normalized_query)
        if best is None:
            return None
        record, score = best
        if score < self.DETERMINISTIC_THRESHOLD:
            return None
        return record, [f"alias_score={score:.2f}"]

    # -------------------------------------------------------------------------
    def _find_best_record(
        self, normalized_value: str
    ) -> tuple[MonographRecord, float] | None:
        best_score = 0.0
        best_record: MonographRecord | None = None
        value_tokens = {token for token in normalized_value.split() if token}
        for record in self.records:
            score = self._score_match(value_tokens, record.tokens)
            if score > best_score:
                best_score = score
                best_record = record
        if best_record is None:
            return None
        return best_record, best_score

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
        if self.langsmith_batch_lookup is not None:
            return await self.langsmith_batch_lookup(
                patient_drugs,
                normalized_queries=normalized_queries,
            )
        return await self._llm_batch_match_lookup_impl(
            patient_drugs,
            normalized_queries=normalized_queries,
        )

    # -------------------------------------------------------------------------
    async def _llm_batch_match_lookup_impl(
        self,
        patient_drugs: list[str],
        *,
        normalized_queries: list[str],
    ) -> list[tuple[MonographRecord, float, str, list[str]] | None]:
        total = len(patient_drugs)
        if total == 0 or not self.records:
            return []
        candidate_block = self.candidate_prompt_block or ""
        if not candidate_block:
            return [None] * total
        drugs_block = "\n".join(
            f"- {name}" if name else "-"
            for name in patient_drugs
        )
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
        llm_call = (
            self.langsmith_llm_call
            if self.langsmith_llm_call is not None
            else self.llm_client.llm_structured_call
        )
        start_time = time.perf_counter()
        try:
            suggestion = await llm_call(
                model=model_name,
                system_prompt=LIVERTOX_MATCH_SYSTEM_PROMPT,
                user_prompt=prompt,
                schema=LiverToxBatchMatchSuggestion,
                temperature=0.0,
            )
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
        if len(matches) != total:
            logger.warning(
                "LLM returned %s matches for %s patient drugs",
                len(matches),
                total,
            )
            return [None] * total
        results: list[tuple[MonographRecord, float, str, list[str]] | None] = [
            None
        ] * total
        for idx, (item, original, normalized_query) in enumerate(
            zip(matches, patient_drugs, normalized_queries, strict=False)
        ):
            if not normalized_query:
                continue
            llm_drug_name = (item.drug_name or "").strip()
            if llm_drug_name and llm_drug_name != original:
                logger.debug(
                    "LLM drug name mismatch at index %s: '%s' vs '%s'",
                    idx,
                    llm_drug_name,
                    original,
                )
            match_name = (item.match_name or "").strip()
            if not match_name:
                continue
            normalized_match = self._normalize_name(match_name)
            record = self.records_by_normalized.get(normalized_match)
            confidence = (
                item.confidence
                if item.confidence is not None
                else self.LLM_DEFAULT_CONFIDENCE
            )
            notes: list[str] = [
                f"LLM selected '{match_name}' for '{original}'"
            ]
            if item.rationale:
                notes.append(item.rationale.strip())
            if record is None:
                best = self._find_best_record(normalized_match)
                if best is None:
                    continue
                record, score = best
                notes.append(
                    f"Mapped suggestion to '{record.drug_name}' (score={score:.2f})"
                )
                confidence = min(max(confidence, score), 1.0)
            results[idx] = (record, confidence, "llm_fallback", notes)
        return results

    # -------------------------------------------------------------------------
    def _wrap_with_langsmith(
        self,
        *,
        name: str,
        run_type: str,
        target: Callable[..., Awaitable[Any]],
    ) -> Callable[..., Awaitable[Any]] | None:
        if not LANGSMITH_TRACING_ENABLED or langsmith_traceable is None:
            return None
        try:
            decorator = langsmith_traceable(name=name, run_type=run_type)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Unable to create LangSmith tracer '%s': %s",
                name,
                exc,
            )
            return None
        try:
            return decorator(target)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Unable to wrap '%s' with LangSmith tracing: %s",
                name,
                exc,
            )
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

    # -------------------------------------------------------------------------
