from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

from common.utils.logger import logger
from configurations.llm_configs import LLMRuntimeConfig
from domain.patient_timeline import PatientTimeline, PatientTimelineEvent
from services.inspection.normalization import (
    extract_lab_marker,
    first_iso_date,
    normalize_text,
)
from services.inspection.runtime import coerce_optional_str


def get_session_timeline(service: Any, session_id: int) -> PatientTimeline | None:
    payload = service.serializer.get_session_result_payload(session_id)
    if not isinstance(payload, dict):
        return None
    timeline_payload = payload.get("patient_timeline")
    if not isinstance(timeline_payload, dict):
        return None
    try:
        return PatientTimeline.model_validate(timeline_payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Invalid persisted timeline payload for session_id=%s: %s",
            session_id,
            exc,
        )
        return None


def build_fallback_timeline(
    service: Any,
    *,
    session_id: int,
    source: dict[str, Any],
) -> PatientTimeline:
    events: list[PatientTimelineEvent] = []
    visit_date = first_iso_date(source.get("visit_date"))

    drugs_text = normalize_text(source.get("drugs"))
    if drugs_text:
        events.append(
            PatientTimelineEvent(
                event_id="therapy-1",
                title="Therapy context",
                description=drugs_text[:450],
                event_type="therapy",
                timing_type="relative",
                event_date=visit_date,
                extracted_timing_text=visit_date,
                source="fallback_parser",
                source_evidence=drugs_text[:1000],
                sort_order=10,
            )
        )

    anamnesis_text = normalize_text(source.get("anamnesis"))
    if anamnesis_text:
        events.append(
            PatientTimelineEvent(
                event_id="disease-1",
                title="Clinical symptom context",
                description=anamnesis_text[:450],
                event_type="disease",
                timing_type="relative",
                event_date=visit_date,
                extracted_timing_text=visit_date,
                source="fallback_parser",
                source_evidence=anamnesis_text[:1000],
                sort_order=20,
            )
        )

    labs_text = normalize_text(source.get("laboratory_analysis"))
    if labs_text:
        marker = extract_lab_marker(labs_text)
        events.append(
            PatientTimelineEvent(
                event_id="lab-1",
                title=marker or "Laboratory findings",
                description=labs_text[:450],
                event_type="lab",
                timing_type="explicit_date" if visit_date else "relative",
                event_date=visit_date,
                extracted_timing_text=visit_date,
                source="fallback_parser",
                source_evidence=labs_text[:1000],
                sort_order=30,
            )
        )

    if not events:
        events.append(
            PatientTimelineEvent(
                event_id="other-1",
                title="Session clinical context",
                description="Structured timeline was unavailable; fallback summary retained.",
                event_type="other",
                timing_type="uncertain",
                source="fallback_parser",
                source_evidence="No structured timeline-relevant fields were available.",
                sort_order=100,
            )
        )

    return PatientTimeline(
        session_id=session_id,
        generated_at=datetime.now(UTC),
        events=events,
    )


def generate_session_timeline(
    service: Any,
    session_id: int,
    *,
    force_regenerate: bool = False,
) -> PatientTimeline | None:
    safe_session_id = int(session_id)
    now = time.monotonic()
    with service.timeline_generation_lock:
        cooldown_until = service.timeline_generation_cooldown_until.get(
            safe_session_id, 0.0
        )
        if now < cooldown_until:
            raise RuntimeError(
                "Timeline regeneration is cooling down. Please wait a few seconds and retry."
            )
        if safe_session_id in service.timeline_generation_inflight:
            raise RuntimeError(
                "Timeline regeneration is already in progress for this session."
            )
        service.timeline_generation_inflight.add(safe_session_id)
    if not force_regenerate:
        cached = get_session_timeline(service, session_id)
        if cached is not None:
            with service.timeline_generation_lock:
                service.timeline_generation_inflight.discard(safe_session_id)
            return cached
    try:
        source = service.serializer.get_session_timeline_source(session_id)
        if source is None:
            return None
        session_payload = source.get("session_result_payload")
        if not isinstance(session_payload, dict):
            session_payload = {}
        runtime_settings = session_payload.get("runtime_settings")
        if not isinstance(runtime_settings, dict):
            runtime_settings = {}

        timeline_timeout_s = max(
            20.0,
            min(
                300.0,
                float(getattr(service.timeline_extractor, "timeout_s", 90.0)) + 20.0,
            ),
        )
        text_extraction_model = coerce_optional_str(
            runtime_settings.get("text_extraction_model")
        ) or coerce_optional_str(source.get("text_extraction_model"))
        clinical_model = coerce_optional_str(
            runtime_settings.get("clinical_model")
        ) or coerce_optional_str(source.get("clinical_model"))
        requested_runtime_settings = {
            "use_cloud_services": LLMRuntimeConfig.is_cloud_enabled(),
            "llm_provider": LLMRuntimeConfig.get_llm_provider(),
            "cloud_model": LLMRuntimeConfig.get_cloud_model(),
            "text_extraction_model": LLMRuntimeConfig.get_text_extraction_model()
            or text_extraction_model,
            "clinical_model": LLMRuntimeConfig.get_clinical_model() or clinical_model,
            "ollama_temperature": LLMRuntimeConfig.get_ollama_temperature(),
            "cloud_temperature": LLMRuntimeConfig.get_cloud_temperature(),
            "ollama_reasoning": LLMRuntimeConfig.is_ollama_reasoning_enabled(),
        }

        try:
            timeline = asyncio.run(
                asyncio.wait_for(
                    service.timeline_extractor.extract_timeline(
                        session_id=session_id,
                        source_payload=source,
                        runtime_settings=requested_runtime_settings,
                    ),
                    timeout=timeline_timeout_s,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Timeline extraction unavailable for session_id=%s, using deterministic fallback: %s",
                session_id,
                exc,
            )
            timeline = build_fallback_timeline(
                service,
                session_id=safe_session_id,
                source=source,
            )

        session_payload["runtime_settings"] = {
            "use_cloud_services": (
                requested_runtime_settings["use_cloud_services"]
                if requested_runtime_settings["use_cloud_services"] is not None
                else LLMRuntimeConfig.is_cloud_enabled()
            ),
            "llm_provider": (
                requested_runtime_settings["llm_provider"]
                if requested_runtime_settings["llm_provider"]
                else LLMRuntimeConfig.get_llm_provider()
            ),
            "cloud_model": (
                requested_runtime_settings["cloud_model"]
                if requested_runtime_settings["cloud_model"]
                else LLMRuntimeConfig.get_cloud_model()
            ),
            "text_extraction_model": (
                requested_runtime_settings["text_extraction_model"]
                if requested_runtime_settings["text_extraction_model"]
                else LLMRuntimeConfig.get_text_extraction_model()
            ),
            "clinical_model": (
                requested_runtime_settings["clinical_model"]
                if requested_runtime_settings["clinical_model"]
                else LLMRuntimeConfig.get_clinical_model()
            ),
            "ollama_temperature": (
                requested_runtime_settings["ollama_temperature"]
                if requested_runtime_settings["ollama_temperature"] is not None
                else LLMRuntimeConfig.get_ollama_temperature()
            ),
            "cloud_temperature": (
                requested_runtime_settings["cloud_temperature"]
                if requested_runtime_settings["cloud_temperature"] is not None
                else LLMRuntimeConfig.get_cloud_temperature()
            ),
            "ollama_reasoning": (
                requested_runtime_settings["ollama_reasoning"]
                if requested_runtime_settings["ollama_reasoning"] is not None
                else LLMRuntimeConfig.is_ollama_reasoning_enabled()
            ),
        }
        session_payload["patient_timeline"] = timeline.model_dump(mode="json")
        service.serializer.upsert_session_result_payload(session_id, session_payload)
        with service.timeline_generation_lock:
            service.timeline_generation_cooldown_until.pop(safe_session_id, None)
        return timeline
    finally:
        with service.timeline_generation_lock:
            service.timeline_generation_inflight.discard(safe_session_id)
