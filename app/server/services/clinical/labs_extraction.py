from __future__ import annotations

from collections.abc import Callable
from typing import Any

from common.utils.logger import logger
from domain.clinical.entities import (
    ClinicalLabEntry,
    LiverInjuryOnsetContext,
    PatientData,
    PatientLabTimeline,
)
from domain.clinical.extras import LabExtractionPayload
from services.clinical.extraction_strategy import decide_extraction_strategy

###############################################################################
async def extract_from_payload(
    extractor: Any,
    payload: PatientData,
    *,
    already_cleaned: bool = False,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[PatientLabTimeline, LiverInjuryOnsetContext | None]:
    primary_labs_text = (
        (payload.laboratory_analysis or "")
        if already_cleaned
        else extractor.clean_text(payload.laboratory_analysis)
    )
    deterministic_entries: list[ClinicalLabEntry] = []
    timeline_entries: list[ClinicalLabEntry] = []
    onset_context: LiverInjuryOnsetContext | None = None
    extractor.emit_progress(progress_callback, 0.0)
    deterministic_entries.extend(
        extractor.extract_entries_from_text(
            text=primary_labs_text,
            source="laboratory_analysis",
            visit_date=payload.visit_date,
        )
    )
    extractor.emit_progress(progress_callback, 0.2)
    merged_source_text = primary_labs_text
    if merged_source_text:
        try:
            await extractor.ensure_client()
            if extractor.client is None:
                raise RuntimeError("LLM client is not initialized for lab extraction")
            llm_unavailable = False
            try:
                parsed = await extractor.llm_extract_full_text(
                    text=merged_source_text,
                    reinforced=False,
                    expected_candidates=deterministic_entries,
                )
            except Exception as exc:
                logger.warning(
                    "Clinical lab extraction unavailable after %d attempts; using deterministic parser output only: %s",
                    extractor.extraction_retry_attempts,
                    exc,
                )
                parsed = LabExtractionPayload(entries=[], onset_context=None)
                llm_unavailable = True
            if not parsed.entries and extractor.has_explicit_lab_signal(merged_source_text):
                try:
                    reinforced = await extractor.llm_extract_full_text(
                        text=merged_source_text,
                        reinforced=True,
                        expected_candidates=deterministic_entries,
                    )
                except Exception as exc:
                    logger.warning("Reinforced clinical lab extraction failed: %s", exc)
                else:
                    if reinforced.entries:
                        parsed = reinforced
            feedback, missing_candidates = extractor.validate_lab_entries_against_candidates(
                list(parsed.entries), deterministic_entries
            )
            if feedback and deterministic_entries:
                try:
                    reinforced = await extractor.llm_extract_full_text(
                        text=merged_source_text,
                        reinforced=True,
                        expected_candidates=deterministic_entries,
                        validation_feedback=feedback,
                        previous_wrong_output=parsed.model_dump_json(),
                    )
                except Exception as exc:
                    logger.warning("Feedback clinical lab extraction failed: %s", exc)
                else:
                    parsed = reinforced
                    feedback, missing_candidates = extractor.validate_lab_entries_against_candidates(
                        list(parsed.entries), deterministic_entries
                    )
            timeline_entries.extend(parsed.entries)
            extractor.emit_progress(progress_callback, 0.7)
            if llm_unavailable or (not parsed.entries and deterministic_entries) or missing_candidates:
                if not parsed.entries and deterministic_entries:
                    logger.warning(
                        "LLM lab extraction returned no entries despite detectable lab markers; using deterministic lab parser output."
                    )
                if missing_candidates:
                    logger.warning(
                        "LLM lab extraction missed %d grounded candidates after retry; merging deterministic fallback candidates.",
                        len(missing_candidates),
                    )
                    timeline_entries.extend(missing_candidates)
                elif llm_unavailable or not parsed.entries:
                    timeline_entries.extend(deterministic_entries)
            onset_context = parsed.onset_context
        except Exception as exc:
            logger.warning(
                "Clinical lab extraction unavailable; using deterministic parser output only: %s",
                exc,
            )
            timeline_entries.extend(deterministic_entries)
    else:
        timeline_entries.extend(deterministic_entries)
    extractor.emit_progress(progress_callback, 0.85)
    normalized: list[ClinicalLabEntry] = []
    seen: set[tuple[str, str, str, str]] = set()
    for entry in timeline_entries:
        prepared = extractor.normalize_entry(entry, visit_date=payload.visit_date)
        if prepared is None:
            continue
        key = extractor.dedupe_key(prepared)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(prepared)
    normalized.sort(key=extractor.lab_entry_sort_key)
    extractor.emit_progress(progress_callback, 1.0)
    return PatientLabTimeline(entries=normalized), onset_context


###############################################################################
async def extract_from_payload_with_audit(
    extractor: Any,
    payload: PatientData,
    *,
    already_cleaned: bool = False,
    progress_callback: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    primary_labs_text = (
        (payload.laboratory_analysis or "")
        if already_cleaned
        else extractor.clean_text(payload.laboratory_analysis)
    )
    deterministic_entries = extractor.extract_entries_from_text(
        text=primary_labs_text,
        source="laboratory_analysis",
        visit_date=payload.visit_date,
    )
    meaningful_lab_lines = sum(
        1
        for line in primary_labs_text.splitlines()
        if line.strip() and extractor.has_explicit_lab_signal(line)
    )
    decision = decide_extraction_strategy(
        section="laboratory_history",
        meaningful_line_count=meaningful_lab_lines,
        parsed_line_count=len(deterministic_entries),
        unresolved_line_count=max(0, meaningful_lab_lines - len(deterministic_entries)),
        evidence_span_count=len(deterministic_entries),
    )
    timeline, onset_context = await extract_from_payload(
        extractor,
        payload,
        already_cleaned=already_cleaned,
        progress_callback=progress_callback,
    )
    return {
        "lab_timeline": timeline,
        "onset_context": onset_context,
        "strategy": decision.strategy,
        "decision": decision.model_dump(),
        "unresolved_lines": [],
        "confidence": decision.confidence,
        "warnings": [],
    }
