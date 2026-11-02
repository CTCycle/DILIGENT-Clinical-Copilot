# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from common.catalogs.model_choices import get_cloud_model_choices
from common.prompts.timeline import (
    PATIENT_TIMELINE_EXTRACTION_SYSTEM_PROMPT,
    build_patient_timeline_user_prompt,
)
from common.utils.logger import logger
from services.llm.runtime_config import LLMRuntimeConfig
from services.llm.generation_policy import GenerationPurpose
from configurations.startup import get_server_settings
from domain.patient_timeline import (
    PatientTimeline,
    PatientTimelineEvent,
    PatientTimelineExtraction,
)
from common.utils.clinical_dates import DateOrder, case_date_order
from domain.timeline_dates import (
    EvidenceDate,
    TimelineDateInterval,
    find_evidence_dates,
    normalize_timeline_interval,
)
from services.llm.client_runtime import ensure_runtime_client
from services.llm.provider_factory import select_llm_provider

ISO_PARTIAL_DATE_RE = re.compile(r"\b\d{4}(?:-\d{2}(?:-\d{2})?)?\b")
CANONICAL_TIMELINE_SOURCE_FIELDS = (
    "anamnesis",
    "drugs",
    "laboratory_analysis",
)


###############################################################################
class _EmptyTimelineExtractionError(ValueError):
    """Raised when clinical source fields yield no usable timeline events."""

    error_code = "invalid_response"
    retryable = False


###############################################################################
def build_timeline_source(source_payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only clinician-entered session text as timeline model input.

    Generated reports, pipeline artifacts, runtime settings, and patient
    identity are excluded so the model cannot quote model-generated text as
    evidence and no identity data is sent to the provider.
    """
    scoped: dict[str, Any] = {}
    visit_date = source_payload.get("visit_date")
    if isinstance(visit_date, str) and visit_date.strip():
        scoped["visit_date"] = visit_date.strip()
    for field_name in CANONICAL_TIMELINE_SOURCE_FIELDS:
        value = source_payload.get(field_name)
        if isinstance(value, str) and value.strip():
            scoped[field_name] = value
    sections = source_payload.get("sections")
    if isinstance(sections, dict):
        scoped_sections = {
            str(name): value
            for name, value in sections.items()
            if isinstance(value, str) and value.strip()
        }
        if scoped_sections:
            scoped["sections"] = scoped_sections
    return scoped


###############################################################################
def _iter_source_text_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested_value in value.values():
            yield from _iter_source_text_values(nested_value)
    elif isinstance(value, (list, tuple)):
        for nested_value in value:
            yield from _iter_source_text_values(nested_value)


###############################################################################
def _normalize_evidence_text(value: str) -> str:
    return " ".join(value.casefold().split())


###############################################################################
def _has_source_evidence(source_payload: dict[str, Any], evidence: str) -> bool:
    normalized_evidence = _normalize_evidence_text(evidence)
    return bool(normalized_evidence) and any(
        normalized_evidence in _normalize_evidence_text(source_text)
        for source_text in _iter_source_text_values(source_payload)
    )


###############################################################################
def _canonical_source_text_fields(
    source_payload: dict[str, Any],
) -> dict[str, list[str]]:
    """Return the named clinical source fields, collapsing section duplicates."""
    fields: dict[str, list[str]] = {}
    for field_name in CANONICAL_TIMELINE_SOURCE_FIELDS:
        value = source_payload.get(field_name)
        if isinstance(value, str) and value.strip():
            fields[field_name] = [value]

    sections = source_payload.get("sections")
    if isinstance(sections, dict):
        for raw_name, value in sections.items():
            field_name = str(raw_name).strip()
            if not field_name or len(field_name) > 80:
                continue
            if isinstance(value, str) and value.strip():
                fields.setdefault(field_name, []).append(value)
    return fields


###############################################################################
def _source_field_for_evidence(
    source_payload: dict[str, Any], evidence: str | None
) -> str | None:
    """Resolve provenance only when evidence identifies one named source field."""
    if not evidence or not evidence.strip():
        return None
    normalized_evidence = _normalize_evidence_text(evidence)
    matches = [
        field_name
        for field_name, source_texts in _canonical_source_text_fields(
            source_payload
        ).items()
        if any(
            normalized_evidence in _normalize_evidence_text(source_text)
            for source_text in source_texts
        )
    ]
    return matches[0] if len(matches) == 1 else None

_PRECISION_RANK = {"year": 0, "month": 1, "day": 2}


###############################################################################
class PatientTimelineExtractor:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        client: Any | None = None,
        temperature: float = 0.0,
        timeout_s: float | None = None,
    ) -> None:
        self.temperature = float(temperature)
        self.timeout_s = float(
            get_server_settings().runtime.parser_llm_timeout
            if timeout_s is None
            else timeout_s
        )
        self.client: Any | None = client
        self.model: str = ""
        self.extraction_retry_attempts = 3
        self.client_lock = asyncio.Lock()
        self.client_loop_id: int | None = None
        if client is None:
            self.client_provider: str | None = None
            self.runtime_revision = -1
            self.runtime_signature: tuple[str, str] | None = None
        else:
            self.client_provider = "injected"
            self.runtime_revision = LLMRuntimeConfig.get_revision()
            self.runtime_signature = None

    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_optional_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    # -------------------------------------------------------------------------
    @classmethod
    def _resolve_provider_model_from_runtime_settings(
        cls,
        runtime_settings: dict[str, Any] | None,
    ) -> tuple[str, str]:
        if runtime_settings is None:
            return LLMRuntimeConfig.resolve_provider_and_model("timeline")

        use_cloud_services = bool(runtime_settings.get("use_cloud_services"))
        timeline_model = cls._coerce_optional_text(
            runtime_settings.get("timeline_model")
        )
        llm_provider = cls._coerce_optional_text(
            runtime_settings.get("llm_provider")
        ).lower()
        if llm_provider not in get_cloud_model_choices():
            raise ValueError(f"Unsupported cloud provider: {llm_provider or '<empty>'}")
        if not timeline_model:
            raise ValueError("Runtime settings require an explicit timeline_model.")

        if use_cloud_services:
            return llm_provider, timeline_model

        return "ollama", timeline_model

    # -------------------------------------------------------------------------
    async def ensure_client(
        self,
        *,
        runtime_settings: dict[str, Any] | None = None,
    ) -> None:
        provider, model = self._resolve_provider_model_from_runtime_settings(
            runtime_settings
        )
        revision = LLMRuntimeConfig.get_revision() if runtime_settings is None else -1
        signature = (provider, model)
        await ensure_runtime_client(
            self,
            provider=provider,
            model=model,
            revision=revision,
            signature=signature,
            track_revision=runtime_settings is None,
            track_signature=runtime_settings is not None,
            client_factory=lambda selected_provider, selected_model: (
                select_llm_provider(
                    provider=selected_provider,
                    timeout_s=self.timeout_s,
                    default_model=selected_model,
                    max_retries=0,
                )
            ),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_date_token(value: str | None) -> str | None:
        interval = normalize_timeline_interval(value)
        return interval.value if interval else None

    # -------------------------------------------------------------------------
    @staticmethod
    def event_sort_key(
        event: PatientTimelineEvent, source_index: int = 0
    ) -> tuple[int, int, int, int]:
        """Order dated events by start day, coarser precision first, then keep
        the model's narrative order for ties and for undated events."""
        interval = normalize_timeline_interval(event.event_date)
        if interval is None:
            return (1, 0, 0, source_index)
        return (
            0,
            interval.start_day,
            _PRECISION_RANK.get(interval.precision, 0),
            source_index,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def event_dedupe_key(event: PatientTimelineEvent) -> tuple[str, str, str]:
        return (
            event.title.casefold(),
            (event.event_date or "").casefold(),
            (event.relative_time or "").casefold(),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def classify_complexity(source_payload: dict[str, Any]) -> str:
        serialized = json.dumps(
            source_payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        date_count = len(ISO_PARTIAL_DATE_RE.findall(serialized))
        if len(serialized) <= 12000 and date_count <= 8:
            return "simple"
        if len(serialized) <= 50000 and date_count <= 24:
            return "moderate"
        return "complex"

    # -------------------------------------------------------------------------
    @staticmethod
    def _date_supported_by(
        interval: TimelineDateInterval, evidence_dates: list[EvidenceDate]
    ) -> EvidenceDate | None:
        """Return the evidence date that states this date at equal precision."""
        for candidate in evidence_dates:
            if candidate.interval.value == interval.value:
                return candidate
        return None

    # -------------------------------------------------------------------------
    def ground_event_dates(
        self, payload: dict[str, Any], date_order: DateOrder
    ) -> None:
        """Reconcile model dates with the dates stated in the quoted evidence.

        A date counts as explicit only when the evidence states it. A model date
        that contradicts the single date in the evidence of an explicitly dated
        event is corrected; model dates the evidence does not state are kept
        but labelled inferred with a review reason. Numbers that are doses or
        laboratory values are never read as dates.
        """
        evidence_dates = find_evidence_dates(payload.get("source_evidence"), date_order)
        start = normalize_timeline_interval(payload.get("event_date"))
        end = normalize_timeline_interval(payload.get("event_date_end"))
        reasons: list[str] = []
        start_explicit = False

        if start is None:
            if len(evidence_dates) == 1:
                start = evidence_dates[0].interval
                start_explicit = True
                payload["extracted_timing_text"] = (
                    payload.get("extracted_timing_text") or evidence_dates[0].text
                )
                if payload.get("timing_type") == "uncertain":
                    payload["timing_type"] = "explicit_date"
        elif (supported := self._date_supported_by(start, evidence_dates)) is not None:
            start_explicit = True
            payload["extracted_timing_text"] = (
                payload.get("extracted_timing_text") or supported.text
            )
        elif (
            len(evidence_dates) == 1
            and payload.get("timing_type") in {"explicit_date", "uncertain"}
            and not (
                evidence_dates[0].interval.start_day <= start.start_day
                and start.end_day <= evidence_dates[0].interval.end_day
            )
        ):
            reasons.append(
                f"Model date {start.value} was corrected to the date stated in the "
                f"evidence ({evidence_dates[0].text})."
            )
            start = evidence_dates[0].interval
            start_explicit = True
        elif evidence_dates:
            reasons.append(
                "The date is not stated at this precision in the quoted evidence."
            )
        else:
            reasons.append("The date is not stated in the quoted evidence.")

        end_explicit = True
        if start is not None and end is not None:
            if end.start_day < start.start_day:
                end = None
                reasons.append("The extracted date range was reversed and was removed.")
            else:
                end_explicit = self._date_supported_by(end, evidence_dates) is not None
                if not end_explicit:
                    reasons.append("The range end is not stated in the quoted evidence.")
        elif payload.get("event_date_end"):
            end = None
            reasons.append("The extracted date range was invalid and was removed.")

        payload["event_date"] = start.value if start else None
        payload["event_date_end"] = end.value if start and end else None
        payload["date_precision"] = start.precision if start else None
        if start is None:
            if payload.get("date_certainty") == "explicit":
                payload["date_certainty"] = "uncertain"
        else:
            payload["date_certainty"] = (
                "explicit" if start_explicit and end_explicit else "inferred"
            )
        if reasons:
            model_reason = payload.get("uncertainty_reason")
            payload["uncertainty_reason"] = " ".join(
                [*reasons, *([model_reason] if model_reason else [])]
            )[:500]

    # -------------------------------------------------------------------------
    def normalize_events(
        self,
        events: list[PatientTimelineEvent],
        date_order: DateOrder = "day_first",
    ) -> list[PatientTimelineEvent]:
        deduped: dict[
            tuple[str, str, str], tuple[int, PatientTimelineEvent, set[str]]
        ] = {}
        for source_index, item in enumerate(events):
            if not item.source_evidence or not item.source_evidence.strip():
                continue
            payload = item.model_dump()
            self.ground_event_dates(payload, date_order)
            event = PatientTimelineEvent(**payload)
            key = self.event_dedupe_key(event)
            previous = deduped.get(key)
            if previous is None:
                deduped[key] = (source_index, event, set())
                continue
            previous_index, previous_event, aliases = previous
            previous_score = (
                previous_event.confidence
                if previous_event.confidence is not None
                else -1.0
            )
            current_score = event.confidence if event.confidence is not None else -1.0
            if current_score > previous_score:
                aliases.add(previous_event.event_id)
                deduped[key] = (previous_index, event, aliases)
            else:
                aliases.add(event.event_id)

        ordered = sorted(
            deduped.values(),
            key=lambda entry: self.event_sort_key(entry[1], entry[0]),
        )
        # Event ids must be unique for selection and linking; merged duplicates
        # keep resolving to the event that survived deduplication.
        id_map: dict[str, str] = {}
        final_ids: list[str] = []
        used_ids: set[str] = set()
        for _, event, aliases in ordered:
            base_id = event.event_id[:110]
            final_id = base_id
            suffix = 2
            while final_id in used_ids:
                final_id = f"{base_id}-{suffix}"
                suffix += 1
            used_ids.add(final_id)
            final_ids.append(final_id)
            for alias in (event.event_id, *sorted(aliases)):
                id_map.setdefault(alias, final_id)

        normalized: list[PatientTimelineEvent] = []
        for index, ((_, event, _), final_id) in enumerate(zip(ordered, final_ids)):
            links = [
                id_map[link]
                for link in event.linked_patient_event_ids
                if link in id_map and id_map[link] != final_id
            ]
            normalized.append(
                event.model_copy(
                    update={
                        "event_id": final_id,
                        "sort_order": index,
                        "linked_patient_event_ids": list(dict.fromkeys(links)),
                    }
                )
            )
        return normalized

    # -------------------------------------------------------------------------
    async def extract_timeline(
        self,
        *,
        session_id: int,
        source_payload: dict[str, Any],
        runtime_settings: dict[str, Any] | None = None,
    ) -> PatientTimeline:
        await self.ensure_client(runtime_settings=runtime_settings)
        if self.client is None:
            raise RuntimeError("LLM client is not initialized for timeline extraction")

        timeline_source = build_timeline_source(source_payload)
        date_order = case_date_order(
            *(timeline_source.get(name) for name in CANONICAL_TIMELINE_SOURCE_FIELDS)
        )
        source_payload_json = json.dumps(
            timeline_source,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        source_payload_hash = hashlib.sha256(
            source_payload_json.encode("utf-8")
        ).hexdigest()
        timeline_complexity = self.classify_complexity(timeline_source)
        user_prompt = build_patient_timeline_user_prompt(
            source_payload_json=source_payload_json,
            source_payload_hash=source_payload_hash,
            date_order=date_order,
        )
        parsed: PatientTimelineExtraction | None = None
        for attempt in range(1, self.extraction_retry_attempts + 1):
            try:
                parsed = await self.client.llm_structured_call(
                    model=self.model,
                    system_prompt=PATIENT_TIMELINE_EXTRACTION_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    schema=PatientTimelineExtraction,
                    purpose=GenerationPurpose.TIMELINE_EXTRACTION,
                    timeline_complexity=timeline_complexity,
                    use_json_mode=True,
                    max_repair_attempts=2,
                )
                break
            except Exception as exc:
                error_code = str(getattr(exc, "error_code", "unknown"))
                retryable = bool(getattr(exc, "retryable", False))
                if attempt >= self.extraction_retry_attempts or not retryable:
                    logger.warning(
                        "Timeline extraction stopped session_id=%s attempt=%d/%d "
                        "error_code=%s error_type=%s",
                        session_id,
                        attempt,
                        self.extraction_retry_attempts,
                        error_code,
                        type(exc).__name__,
                        exc_info=True,
                    )
                    raise
                delay = min(6.0, 0.75 * (2 ** (attempt - 1)))
                logger.warning(
                    "Retrying timeline extraction attempt %d/%d (delay %.2fs) "
                    "error_code=%s error_type=%s",
                    attempt,
                    self.extraction_retry_attempts,
                    delay,
                    error_code,
                    type(exc).__name__,
                )
                await asyncio.sleep(delay)

        if parsed is None:
            raise RuntimeError("Failed to extract patient timeline")

        # Each event must quote clinician-entered source text verbatim. Events
        # whose quote cannot be found are discarded and disclosed, never kept.
        verified_events = [
            event.model_copy(
                update={
                    "source": _source_field_for_evidence(
                        timeline_source, event.source_evidence
                    )
                }
            )
            for event in parsed.events
            if event.source_evidence
            and event.source_evidence.strip()
            and _has_source_evidence(timeline_source, event.source_evidence)
        ]
        discarded_count = len(parsed.events) - len(verified_events)
        normalized_events = self.normalize_events(verified_events, date_order)
        has_clinical_source_text = any(
            name in timeline_source for name in CANONICAL_TIMELINE_SOURCE_FIELDS
        )
        if not normalized_events and has_clinical_source_text:
            raise _EmptyTimelineExtractionError(
                "Timeline extraction produced no evidence-backed events despite "
                "populated clinical source fields."
            )
        generation_note = (
            f"{discarded_count} model "
            f"{'event was' if discarded_count == 1 else 'events were'} discarded "
            "because the quoted evidence was missing or not found in the session source."
            if discarded_count
            else None
        )
        return PatientTimeline(
            session_id=int(session_id),
            generated_at=datetime.now(UTC),
            generation_status="llm_generated",
            generation_note=generation_note,
            source_payload_hash=source_payload_hash,
            events=normalized_events,
        )
