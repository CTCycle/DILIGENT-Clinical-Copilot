from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any

from common.catalogs.model_choices import get_cloud_model_choices
from common.prompts.timeline import PATIENT_TIMELINE_EXTRACTION_PROMPT
from common.utils.logger import logger
from services.llm.runtime_config import LLMRuntimeConfig
from services.llm.generation_policy import GenerationPurpose
from configurations.startup import get_server_settings
from domain.patient_timeline import (
    PatientTimeline,
    PatientTimelineEvent,
    PatientTimelineExtraction,
)
from domain.timeline_dates import normalize_timeline_interval, timeline_date_sort_key
from services.llm.client_runtime import ensure_runtime_client
from services.llm.provider_factory import select_llm_provider

DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DATE_SHORT_RE = re.compile(r"^\d{4}-\d{2}$")
DATE_YEAR_RE = re.compile(r"^\d{4}$")
ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
ISO_PARTIAL_DATE_RE = re.compile(r"\b\d{4}(?:-\d{2}(?:-\d{2})?)?\b")

###############################################################################
class PatientTimelineExtractor:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        client: Any | None = None,
        temperature: float = 0.0,
        timeout_s: float = get_server_settings().runtime.parser_llm_timeout,
    ) -> None:
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
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
            return LLMRuntimeConfig.resolve_provider_and_model("parser")

        use_cloud_services = bool(runtime_settings.get("use_cloud_services"))
        text_extraction_model = cls._coerce_optional_text(
            runtime_settings.get("text_extraction_model")
        )
        clinical_model = cls._coerce_optional_text(
            runtime_settings.get("clinical_model")
        )
        cloud_model = cls._coerce_optional_text(runtime_settings.get("cloud_model"))
        llm_provider = cls._coerce_optional_text(
            runtime_settings.get("llm_provider")
        ).lower()
        if llm_provider not in get_cloud_model_choices():
            llm_provider = LLMRuntimeConfig.get_llm_provider().strip().lower()
        if llm_provider not in get_cloud_model_choices():
            llm_provider = "openai"

        if use_cloud_services:
            model = cloud_model or text_extraction_model or clinical_model
            return llm_provider, model

        return "ollama", text_extraction_model or clinical_model

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
            client_factory=lambda selected_provider, selected_model: select_llm_provider(
                provider=selected_provider,
                timeout_s=self.timeout_s,
                default_model=selected_model,
                max_retries=0,
            ),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_date_token(value: str | None) -> str | None:
        interval = normalize_timeline_interval(value)
        return interval.value if interval else None

    # -------------------------------------------------------------------------
    @classmethod
    def event_sort_key(cls, event: PatientTimelineEvent) -> tuple[int, str, int, str]:
        normalized_date = cls.normalize_date_token(event.event_date)
        if normalized_date:
            return (0, str(timeline_date_sort_key(normalized_date)[0]), event.sort_order, event.title.casefold())
        relative = (event.relative_time or "").casefold()
        return (1, relative, event.sort_order, event.title.casefold())

    # -------------------------------------------------------------------------
    @staticmethod
    def event_dedupe_key(event: PatientTimelineEvent) -> tuple[str, str, str]:
        return (
            event.title.casefold(),
            (event.event_date or "").casefold(),
            (event.relative_time or "").casefold(),
        )

    # -------------------------------------------------------------------------
    def normalize_events(
        self, events: list[PatientTimelineEvent]
    ) -> list[PatientTimelineEvent]:
        deduped: dict[tuple[str, str, str], PatientTimelineEvent] = {}
        for item in events:
            if not item.source_evidence or not item.source_evidence.strip():
                continue
            payload = item.model_dump()
            evidence_dates = list(dict.fromkeys(ISO_PARTIAL_DATE_RE.findall(item.source_evidence or "")))
            # An explicit source span is authoritative for a single dated event.
            # This prevents a model from carrying a nearby lab date onto a different
            # observation while retaining ambiguous multi-date evidence unchanged.
            normalized_model_date = self.normalize_date_token(item.event_date)
            normalized_model_end = self.normalize_date_token(item.event_date_end)
            evidence_date = evidence_dates[0] if len(evidence_dates) == 1 else None
            if (
                evidence_date
                and normalized_model_date
                and normalized_model_date.startswith(evidence_date)
                and len(normalized_model_date) > len(evidence_date)
            ):
                # A year token embedded in natural-language evidence is not more
                # authoritative than a model date that preserves the same year at
                # day or month precision.
                evidence_date = normalized_model_date
            payload["event_date"] = evidence_date or normalized_model_date
            payload["event_date_end"] = normalized_model_end
            interval = normalize_timeline_interval(payload["event_date"], normalized_model_end)
            if interval is not None:
                payload["event_date"] = interval.value
                payload["event_date_end"] = interval.end_value
            elif payload["event_date_end"]:
                payload["event_date_end"] = None
                payload["date_certainty"] = "uncertain"
                payload["uncertainty_reason"] = "The extracted date range was reversed or invalid."
            if payload["event_date"] and payload.get("date_precision") is None:
                payload["date_precision"] = "day" if DATE_PREFIX_RE.fullmatch(payload["event_date"]) else "month" if DATE_SHORT_RE.fullmatch(payload["event_date"]) else "year"
            if payload["event_date"] and payload.get("date_certainty") == "uncertain":
                payload["date_certainty"] = "explicit" if len(evidence_dates) == 1 else "inferred"
            event = PatientTimelineEvent(
                **payload,
            )
            key = self.event_dedupe_key(event)
            previous = deduped.get(key)
            if previous is None:
                deduped[key] = event
                continue
            previous_score = (
                previous.confidence if previous.confidence is not None else -1.0
            )
            current_score = event.confidence if event.confidence is not None else -1.0
            if current_score > previous_score:
                deduped[key] = event
        ordered = sorted(deduped.values(), key=self.event_sort_key)
        normalized: list[PatientTimelineEvent] = []
        for index, item in enumerate(ordered):
            payload = item.model_dump()
            payload["sort_order"] = index
            normalized.append(PatientTimelineEvent(**payload))
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

        source_payload_json = json.dumps(
            source_payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        source_payload_hash = hashlib.sha256(
            source_payload_json.encode("utf-8")
        ).hexdigest()
        user_prompt = (
            "Build a structured clinical timeline from this patient session payload.\n"
            "Focus on therapy start/stop, disease manifestation, lab milestones, and other dated events.\n\n"
            f"Source payload SHA-256: {source_payload_hash}\n"
            f"Canonical JSON payload:\n{source_payload_json}"
        )
        parsed: PatientTimelineExtraction | None = None
        for attempt in range(1, self.extraction_retry_attempts + 1):
            try:
                parsed = await self.client.llm_structured_call(
                    model=self.model,
                    system_prompt=PATIENT_TIMELINE_EXTRACTION_PROMPT.strip(),
                    user_prompt=user_prompt,
                    schema=PatientTimelineExtraction,
                    purpose=GenerationPurpose.STRUCTURED_EXTRACTION,
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

        normalized_events = self.normalize_events(parsed.events)
        return PatientTimeline(
            session_id=int(session_id),
            generated_at=datetime.now(UTC),
            generation_status="llm_generated",
            source_payload_hash=source_payload_hash,
            events=normalized_events,
        )
