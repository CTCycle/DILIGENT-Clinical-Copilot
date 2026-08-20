from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast

from common.utils.logger import logger
from services.llm.runtime_config import LLMRuntimeConfig
from domain.patient_timeline import (
    PatientTimeline,
    PatientTimelineEvent,
    PatientTimelineGenerationErrorCode,
)
from services.inspection.normalization import (
    extract_lab_marker,
    normalize_text,
)
from services.inspection.runtime import coerce_optional_str

###############################################################################
def _report_progress(
    callback: Callable[[float, str], None] | None,
    progress: float,
    message: str,
) -> None:
    if callback is not None:
        callback(progress, message)

_TIMELINE_ERROR_CODES: frozenset[PatientTimelineGenerationErrorCode] = frozenset(
    {
        "network_unavailable",
        "timeout",
        "authentication",
        "rate_limited",
        "upstream_error",
        "invalid_response",
        "configuration",
        "provider_error",
        "unknown",
    }
)

###############################################################################
def _timeline_error_code(exc: BaseException) -> PatientTimelineGenerationErrorCode:
    current: BaseException | None = exc
    visited: set[int] = set()
    provider_error_seen = False
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        candidate = str(getattr(current, "error_code", "") or "").strip()
        if candidate in _TIMELINE_ERROR_CODES and candidate != "provider_error":
            return cast(PatientTimelineGenerationErrorCode, candidate)
        provider_error_seen = provider_error_seen or candidate == "provider_error"
        text = " ".join(str(current).split()).casefold()
        if any(token in text for token in ("401", "403", "authentication", "api key", "credential")):
            return "authentication"
        if "429" in text or "rate limit" in text or "rate-limit" in text:
            return "rate_limited"
        if any(token in text for token in ("timed out", "timeout", "time out")):
            return "timeout"
        if any(token in text for token in ("connect", "network", "socket", "winerror 10013")):
            return "network_unavailable"
        if any(token in text for token in ("invalid json", "validation", "structured output")):
            return "invalid_response"
        if "http 5" in text or "service unavailable" in text:
            return "upstream_error"
        if any(
            token in text
            for token in (
                "no active",
                "model is required",
                "unknown provider",
                "not configured",
                "not found for provider",
                "unsupported provider",
                "does not include the requested model",
                "does not declare a transport endpoint",
                "api key is required",
            )
        ):
            return "configuration"
        current = current.__cause__ or current.__context__
    return "provider_error" if provider_error_seen else "unknown"

###############################################################################
def _timeline_fallback_note(
    *,
    use_cloud_services: bool,
    provider: object,
    model: object,
    error_code: PatientTimelineGenerationErrorCode,
) -> str:
    provider_label = coerce_optional_str(provider) or "configured provider"
    model_label = coerce_optional_str(model) or "configured model"
    if use_cloud_services:
        provider_key = provider_label.casefold().replace("-", "_")
        credential_hint = (
            "the active OpenCode access key"
            if provider_key == "opencode_go"
            else f"the active {provider_label} credentials"
        )
        messages = {
            "network_unavailable": "The provider could not be reached. Check outbound HTTPS access and retry.",
            "timeout": "The provider did not respond before the configured timeout. Retry when it is responsive.",
            "authentication": f"The provider rejected authentication. Check {credential_hint} and retry.",
            "rate_limited": "The provider rate-limited the request. Wait briefly and retry.",
            "upstream_error": "The provider returned a temporary upstream error. Retry shortly.",
            "invalid_response": "The provider returned invalid structured data. Retry the extraction.",
            "configuration": "The provider or model configuration is incomplete. Check model settings and retry.",
            "provider_error": "The provider request failed. Check backend logs and retry.",
        }
        message = messages.get(error_code, "The provider request failed unexpectedly. Check backend logs and retry.")
        return f"Cloud timeline extraction using {provider_label} / {model_label} did not complete. {message}"
    local_messages = {
        "timeout": "The local model runtime did not respond before the configured timeout.",
        "invalid_response": "The local model returned invalid structured data.",
        "configuration": "The local model configuration is incomplete.",
        "network_unavailable": "The local model runtime could not be reached.",
    }
    message = local_messages.get(
        error_code,
        "Check the configured local model runtime and backend logs for details.",
    )
    return f"Local timeline extraction did not complete. {message} Retry when ready."

###############################################################################
class InspectionTimelineMixin:
    session_timeline_repository: Any
    timeline_extractor: Any
    timeline_generation_lock: Any
    timeline_generation_inflight: set[int]
    timeline_generation_cooldown_until: dict[int, float]
    jobs: Any

    # -------------------------------------------------------------------------
    def get_session_timeline(self, session_id: int) -> PatientTimeline | None:
        payload = self.session_timeline_repository.get_latest_session_timeline_record(session_id)
        if not isinstance(payload, dict):
            return None
        try:
            return PatientTimeline.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Invalid persisted timeline record for session_id=%s: %s",
                session_id,
                exc,
            )
            return None

    # -------------------------------------------------------------------------
    def get_session_timeline_by_id(
        self,
        session_id: int,
        timeline_id: int,
    ) -> PatientTimeline | None:
        payload = self.session_timeline_repository.get_session_timeline_record(session_id, timeline_id)
        if not isinstance(payload, dict):
            return None
        try:
            return PatientTimeline.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Invalid persisted timeline record for session_id=%s timeline_id=%s: %s",
                session_id,
                timeline_id,
                exc,
            )
            return None

    # -------------------------------------------------------------------------
    def list_session_timelines(self, session_id: int) -> list[dict[str, Any]]:
        return self.session_timeline_repository.list_session_timelines(session_id)

    # -------------------------------------------------------------------------
    def delete_session_timeline(self, session_id: int, timeline_id: int) -> bool:
        return self.session_timeline_repository.delete_session_timeline_record(session_id, timeline_id)

    # -------------------------------------------------------------------------
    def _build_timeline_runtime_settings(
        self,
        *,
        source: dict[str, Any],
    ) -> dict[str, Any]:
        session_payload = source.get("session_result_payload")
        persisted = session_payload.get("runtime_settings") if isinstance(session_payload, dict) else None
        settings = dict(persisted) if isinstance(persisted, dict) else {}
        provider, model = LLMRuntimeConfig.resolve_provider_and_model("timeline")
        settings.setdefault("use_cloud_services", LLMRuntimeConfig.is_cloud_enabled())
        settings.setdefault("llm_provider", LLMRuntimeConfig.get_llm_provider())
        settings.setdefault("cloud_model", LLMRuntimeConfig.get_cloud_model())
        settings.setdefault(
            "text_extraction_model",
            LLMRuntimeConfig.get_text_extraction_model()
            or coerce_optional_str(source.get("text_extraction_model")),
        )
        settings.setdefault(
            "clinical_model",
            LLMRuntimeConfig.get_clinical_model()
            or coerce_optional_str(source.get("clinical_model")),
        )
        if "timeline_model" not in settings:
            legacy_role = (
                "cloud_model"
                if settings["use_cloud_services"]
                else "text_extraction_model"
            )
            settings["timeline_model"] = coerce_optional_str(settings.get(legacy_role)) or model
        settings.setdefault("ollama_reasoning", LLMRuntimeConfig.is_ollama_reasoning_enabled())
        settings.setdefault("ollama_seed", LLMRuntimeConfig.get_ollama_seed())
        return settings

    # -------------------------------------------------------------------------
    def build_fallback_timeline(
        self,
        *,
        session_id: int,
        source: dict[str, Any],
        generation_note: str | None = None,
        generation_error_code: PatientTimelineGenerationErrorCode | None = None,
    ) -> PatientTimeline:
        events: list[PatientTimelineEvent] = []

        drugs_text = normalize_text(source.get("drugs"))
        if drugs_text:
            events.append(
                PatientTimelineEvent(
                    event_id="therapy-1",
                    title="Therapy context",
                    description=drugs_text[:450],
                    event_type="therapy",
                    timing_type="uncertain",
                    event_date=None,
                    extracted_timing_text=None,
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
                    timing_type="uncertain",
                    event_date=None,
                    extracted_timing_text=None,
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
                    timing_type="uncertain",
                    event_date=None,
                    extracted_timing_text=None,
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
            generation_status="fallback",
            generation_note=generation_note
            or "Timeline extraction was unavailable; deterministic fallback events were built from persisted session fields.",
            generation_error_code=generation_error_code,
            events=events,
        )

    # -------------------------------------------------------------------------
    def generate_session_timeline(
        self,
        session_id: int,
        *,
        force_regenerate: bool = False,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> PatientTimeline | None:
        safe_session_id = int(session_id)
        now = time.monotonic()
        with self.timeline_generation_lock:
            cooldown_until = self.timeline_generation_cooldown_until.get(
                safe_session_id, 0.0
            )
            if now < cooldown_until:
                raise RuntimeError(
                    "Timeline regeneration is cooling down. Please wait a few seconds and retry."
                )
            if safe_session_id in self.timeline_generation_inflight:
                raise RuntimeError(
                    "Timeline regeneration is already in progress for this session."
                )
            self.timeline_generation_inflight.add(safe_session_id)
        if not force_regenerate:
            cached = self.get_session_timeline(session_id)
            if cached is not None:
                with self.timeline_generation_lock:
                    self.timeline_generation_inflight.discard(safe_session_id)
                return cached
        try:
            _report_progress(progress_callback, 5, "Preparing session timeline source")
            source = self.session_timeline_repository.get_session_timeline_source(session_id)
            if source is None:
                return None
            timeline_timeout_s = max(
                20.0,
                min(
                    300.0,
                    float(getattr(self.timeline_extractor, "timeout_s", 90.0)) + 20.0,
                ),
            )
            requested_runtime_settings = self._build_timeline_runtime_settings(source=source)
            _report_progress(progress_callback, 15, "Configuring timeline model")
            source_model = (
                requested_runtime_settings["timeline_model"]
                or requested_runtime_settings["cloud_model"]
                or requested_runtime_settings["text_extraction_model"]
            )

            try:
                _report_progress(progress_callback, 25, "Extracting clinical timeline events")
                with LLMRuntimeConfig.override_for_run(requested_runtime_settings):
                    timeline = asyncio.run(
                        asyncio.wait_for(
                            self.timeline_extractor.extract_timeline(
                                session_id=session_id,
                                source_payload=source,
                                runtime_settings=requested_runtime_settings,
                            ),
                            timeout=timeline_timeout_s,
                        )
                    )
                timeline = PatientTimeline(
                    **{
                        **timeline.model_dump(),
                        "generation_status": "llm_generated",
                        "generation_note": None,
                        "source_model": source_model,
                        "source_kind": (
                            "cloud"
                            if requested_runtime_settings["use_cloud_services"]
                            else "local"
                        ),
                        "model_provider": requested_runtime_settings["llm_provider"],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                error_code = _timeline_error_code(exc)
                logger.warning(
                    "Timeline extraction unavailable for session_id=%s, using deterministic fallback "
                    "error_code=%s error_type=%s",
                    session_id,
                    error_code,
                    type(exc).__name__,
                    exc_info=True,
                )
                timeline = self.build_fallback_timeline(
                    session_id=safe_session_id,
                    source=source,
                    generation_note=_timeline_fallback_note(
                        use_cloud_services=bool(requested_runtime_settings["use_cloud_services"]),
                        provider=requested_runtime_settings.get("llm_provider"),
                        model=source_model,
                        error_code=error_code,
                    ),
                    generation_error_code=error_code,
                )
                timeline = PatientTimeline(
                    **{
                        **timeline.model_dump(),
                        "source_model": source_model,
                        "source_kind": (
                            "cloud"
                            if requested_runtime_settings["use_cloud_services"]
                            else "local"
                        ),
                        "model_provider": requested_runtime_settings["llm_provider"],
                    }
                )
                _report_progress(progress_callback, 82, "Using deterministic fallback chronology")
            else:
                _report_progress(progress_callback, 82, "Timeline events extracted")
            _report_progress(progress_callback, 92, "Saving generated timeline")
            persisted = self.session_timeline_repository.create_session_timeline_record(
                session_id,
                timeline.model_dump(mode="json"),
            )
            with self.timeline_generation_lock:
                self.timeline_generation_cooldown_until.pop(safe_session_id, None)
            if isinstance(persisted, dict):
                validated = PatientTimeline.model_validate(persisted)
                _report_progress(progress_callback, 98, "Timeline saved")
                return validated
            _report_progress(progress_callback, 98, "Timeline saved")
            return timeline
        finally:
            with self.timeline_generation_lock:
                self.timeline_generation_inflight.discard(safe_session_id)
