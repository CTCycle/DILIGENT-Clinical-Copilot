from __future__ import annotations

import asyncio
import contextlib
import re
from datetime import datetime, UTC
from typing import Any

from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.configurations.llm_configs import LLMRuntimeConfig
from DILIGENT.server.configurations.startup import server_settings
from DILIGENT.server.domain.patient_timeline import (
    PatientTimeline,
    PatientTimelineEvent,
    PatientTimelineExtraction,
)
from DILIGENT.server.services.prompts import PATIENT_TIMELINE_EXTRACTION_PROMPT
from DILIGENT.server.services.llm.providers import initialize_llm_client


DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DATE_SHORT_RE = re.compile(r"^\d{4}-\d{2}$")


class PatientTimelineExtractor:
    def __init__(
        self,
        *,
        client: Any | None = None,
        temperature: float = 0.0,
        timeout_s: float = server_settings.external_data.disease_llm_timeout,
    ) -> None:
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.client: Any | None = client
        self.model: str = ""
        self.extraction_retry_attempts = 2
        self.client_lock = asyncio.Lock()
        if client is None:
            self.client_provider: str | None = None
            self.runtime_revision = -1
        else:
            self.client_provider = "injected"
            self.runtime_revision = LLMRuntimeConfig.get_revision()

    # -------------------------------------------------------------------------
    async def ensure_client(self) -> None:
        async with self.client_lock:
            revision = LLMRuntimeConfig.get_revision()
            provider, model = LLMRuntimeConfig.resolve_provider_and_model("parser")
            if self.client_provider == "injected" and self.client is not None:
                self.model = model
                self.runtime_revision = revision
                return
            needs_refresh = (
                self.client is None
                or self.client_provider != provider
                or self.runtime_revision != revision
            )
            if needs_refresh:
                if self.client is not None:
                    with contextlib.suppress(Exception):
                        await self.client.close()
                self.client = initialize_llm_client(
                    purpose="parser",
                    timeout_s=self.timeout_s,
                )
                self.client_provider = provider
                self.extraction_retry_attempts = 4 if provider in {"openai", "gemini"} else 2
            self.runtime_revision = revision
            self.model = model
            if self.client is not None and model and hasattr(self.client, "default_model"):
                self.client.default_model = model  # type: ignore[attr-defined]

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
    def normalize_events(self, events: list[PatientTimelineEvent]) -> list[PatientTimelineEvent]:
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
            previous_score = previous.confidence if previous.confidence is not None else -1.0
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
    ) -> PatientTimeline:
        await self.ensure_client()
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
