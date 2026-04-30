from __future__ import annotations

import asyncio
import re
from datetime import datetime, UTC
from typing import Any

from common.utils.logger import logger
from common.constants import CLOUD_MODEL_CHOICES
from configurations.llm_configs import LLMRuntimeConfig
from configurations.startup import server_settings
from domain.patient_timeline import (
    PatientTimeline,
    PatientTimelineEvent,
    PatientTimelineExtraction,
)
from services.clinical.prompts import PATIENT_TIMELINE_EXTRACTION_PROMPT
from services.llm.client_runtime import ensure_runtime_client
from services.llm.providers import (
    initialize_llm_client,
    select_llm_provider,
)


DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DATE_SHORT_RE = re.compile(r"^\d{4}-\d{2}$")


class PatientTimelineExtractor:
    def __init__(
        self,
        *,
        client: Any | None = None,
        temperature: float = 0.0,
        timeout_s: float = server_settings.external_data.default_llm_timeout,
    ) -> None:
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.client: Any | None = client
        self.model: str = ""
        self.extraction_retry_attempts = 2
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
        if llm_provider not in CLOUD_MODEL_CHOICES:
            llm_provider = LLMRuntimeConfig.get_llm_provider().strip().lower()
        if llm_provider not in CLOUD_MODEL_CHOICES:
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
            client_factory=(
                lambda _selected_provider, _selected_model: (
                    initialize_llm_client(
                        purpose="parser",
                        timeout_s=self.timeout_s,
                    )
                    if runtime_settings is None
                    else select_llm_provider(
                        provider=provider,
                        timeout_s=self.timeout_s,
                        default_model=model,
                    )
                )
            ),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_date_token(value: str | None) -> str | None:
        if value is None:
            return None
        candidate = str(value).strip()
        if not candidate:
            return None
        if DATE_PREFIX_RE.fullmatch(candidate):
            return candidate
        if DATE_SHORT_RE.fullmatch(candidate):
            return f"{candidate}-01"
        return candidate

    # -------------------------------------------------------------------------
    @classmethod
    def event_sort_key(cls, event: PatientTimelineEvent) -> tuple[int, str, int, str]:
        normalized_date = cls.normalize_date_token(event.event_date)
        if normalized_date and DATE_PREFIX_RE.fullmatch(normalized_date):
            return (0, normalized_date, event.sort_order, event.title.casefold())
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
            payload = item.model_dump()
            payload["event_date"] = self.normalize_date_token(item.event_date)
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

        user_prompt = (
            "Build a structured clinical timeline from this patient session payload.\n"
            "Focus on therapy start/stop, disease manifestation, lab milestones, and other dated events.\n\n"
            f"{source_payload}"
        )
        parsed: PatientTimelineExtraction | None = None
        for attempt in range(1, self.extraction_retry_attempts + 1):
            try:
                parsed = await self.client.llm_structured_call(
                    model=self.model,
                    system_prompt=PATIENT_TIMELINE_EXTRACTION_PROMPT.strip(),
                    user_prompt=user_prompt,
                    schema=PatientTimelineExtraction,
                    temperature=self.temperature,
                    use_json_mode=True,
                    max_repair_attempts=2,
                )
                break
            except Exception as exc:
                if attempt >= self.extraction_retry_attempts:
                    raise RuntimeError("Failed to extract patient timeline") from exc
                delay = min(6.0, 0.75 * (2 ** (attempt - 1)))
                logger.warning(
                    "Retrying timeline extraction attempt %d/%d (delay %.2fs): %s",
                    attempt,
                    self.extraction_retry_attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        if parsed is None:
            raise RuntimeError("Failed to extract patient timeline")

        normalized_events = self.normalize_events(parsed.events)
        return PatientTimeline(
            session_id=int(session_id),
            generated_at=datetime.now(UTC),
            events=normalized_events,
        )

