from __future__ import annotations

import asyncio
import contextlib
import re
import unicodedata
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field

from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.configurations import LLMRuntimeConfig, server_settings
from DILIGENT.server.domain.clinical import (
    ClinicalLabEntry,
    LiverInjuryOnsetContext,
    PatientData,
    PatientLabTimeline,
)
from DILIGENT.server.models.prompts import CLINICAL_LAB_EXTRACTION_PROMPT
from DILIGENT.server.models.providers import initialize_llm_client


###############################################################################
RATE_LIMIT_WAIT_HINT_RE = re.compile(
    r"please\s+try\s+again\s+in\s+([0-9]+(?:\.[0-9]+)?)s",
    re.IGNORECASE,
)
NUMERIC_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
MARKER_ALIASES: dict[str, tuple[str, ...]] = {
    "ALT": ("alt", "alat", "gpt"),
    "AST": ("ast", "asat", "got"),
    "ALP": ("alp", "alkp", "alkaline phosphatase"),
    "TBIL": ("tbil", "total bilirubin", "bilirubin total", "bilirubin"),
    "DBIL": ("dbil", "direct bilirubin", "bilirubin direct"),
    "GGT": ("ggt", "gamma gt", "gamma-glutamyl transferase"),
    "INR": ("inr",),
    "ALB": ("albumin", "alb"),
}


###############################################################################
class LabExtractionPayload(BaseModel):
    entries: list[ClinicalLabEntry] = Field(default_factory=list)
    onset_context: LiverInjuryOnsetContext | None = Field(default=None)


###############################################################################
class ClinicalLabExtractor:
    CHUNK_MAX_CHARS = 2600

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
                self.extraction_retry_attempts = (
                    4 if provider in {"openai", "gemini"} else 2
                )
            self.runtime_revision = revision
            self.model = model
            if (
                self.client is not None
                and model
                and hasattr(self.client, "default_model")
            ):
                self.client.default_model = model  # type: ignore[attr-defined]

    # -------------------------------------------------------------------------
    @staticmethod
    def emit_progress(
        progress_callback: Callable[[float], None] | None,
        fraction: float,
    ) -> None:
        if progress_callback is None:
            return
        bounded_fraction = min(1.0, max(0.0, float(fraction)))
        progress_callback(bounded_fraction)

    # -------------------------------------------------------------------------
    def clean_text(self, text: str | None) -> str:
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.strip() for line in normalized.split("\n") if line.strip()]
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    def chunk_text(self, text: str) -> list[str]:
        normalized = self.clean_text(text)
        if not normalized:
            return []
        lines = [line.strip() for line in normalized.split("\n") if line.strip()]
        chunks: list[str] = []
        current_lines: list[str] = []
        current_size = 0
        for line in lines:
            line_size = len(line) + 1
            if current_lines and current_size + line_size > self.CHUNK_MAX_CHARS:
                chunks.append("\n".join(current_lines))
                current_lines = [line]
                current_size = line_size
                continue
            current_lines.append(line)
            current_size += line_size
        if current_lines:
            chunks.append("\n".join(current_lines))
        return chunks or [normalized]

    # -------------------------------------------------------------------------
    @staticmethod
    def parse_numeric(raw: str | None) -> float | None:
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            return float(raw)
        text = str(raw).strip()
        if not text:
            return None
        match = NUMERIC_RE.search(text)
        if not match:
            return None
        try:
            return float(match.group().replace(",", "."))
        except ValueError:
            return None

    # -------------------------------------------------------------------------
    def normalize_marker_name(self, raw: str | None) -> str:
        text = (raw or "").strip().lower()
        if not text:
            return "UNKNOWN"
        for canonical, aliases in MARKER_ALIASES.items():
            if text == canonical.lower():
                return canonical
            for alias in aliases:
                if alias in text:
                    return canonical
        return text.upper()

    # -------------------------------------------------------------------------
    @staticmethod
    def try_parse_date(value: str) -> date | None:
        cleaned = value.strip()
        if not cleaned:
            return None
        iso_candidate = cleaned.replace(".", "-").replace("/", "-")
        try:
            return date.fromisoformat(iso_candidate)
        except ValueError:
            pass
        for fmt in ("%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d", "%d.%m.%Y", "%Y.%m.%d"):
            try:
                return datetime.strptime(cleaned, fmt).date()
            except ValueError:
                continue
        return None

    # -------------------------------------------------------------------------
    def normalize_date_with_visit_year(
        self,
        raw_date: str | None,
        visit_date: date | None,
    ) -> str | None:
        if raw_date is None:
            return None
        text = str(raw_date).strip()
        if not text:
            return None
        normalized = text.replace("/", "-").replace(".", "-").replace(",", "-")
        tokens = [token for token in normalized.split("-") if token]
        candidates: list[str] = []
        if visit_date is not None and len(tokens) == 2:
            day, month = tokens
            candidates.extend(
                [
                    f"{day.zfill(2)}-{month.zfill(2)}-{visit_date.year}",
                    f"{month.zfill(2)}-{day.zfill(2)}-{visit_date.year}",
                    f"{visit_date.year}-{month.zfill(2)}-{day.zfill(2)}",
                ]
            )
        candidates.extend(["-".join(tokens), text, normalized])
        checked: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in checked:
                continue
            checked.add(candidate)
            parsed = self.try_parse_date(candidate)
            if parsed is not None:
                return parsed.isoformat()
        return text

    # -------------------------------------------------------------------------
    @staticmethod
    def dedupe_key(entry: ClinicalLabEntry) -> tuple[str, str, str, str]:
        value_token = (
            f"{entry.value:.6f}" if isinstance(entry.value, (int, float)) else (entry.value_text or "")
        )
        date_token = (entry.sample_date or "").strip().lower()
        return (
            entry.marker_name.strip().upper(),
            date_token,
            value_token.strip().lower(),
            entry.source,
        )

    # -------------------------------------------------------------------------
    def normalize_entry(
        self,
        entry: ClinicalLabEntry,
        *,
        visit_date: date | None,
    ) -> ClinicalLabEntry | None:
        marker = self.normalize_marker_name(entry.marker_name)
        if marker == "UNKNOWN":
            return None
        normalized_date = self.normalize_date_with_visit_year(entry.sample_date, visit_date)
        return ClinicalLabEntry(
            marker_name=marker,
            value=entry.value if entry.value is not None else self.parse_numeric(entry.value_text),
            value_text=entry.value_text,
            unit=entry.unit,
            upper_limit_normal=(
                entry.upper_limit_normal
                if entry.upper_limit_normal is not None
                else self.parse_numeric(entry.upper_limit_text)
            ),
            upper_limit_text=entry.upper_limit_text,
            sample_date=normalized_date,
            relative_time=(entry.relative_time or "").strip() or None,
            evidence=(entry.evidence or "").strip() or None,
            source=entry.source,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def extract_rate_limit_wait_hint_seconds(exc: Exception) -> float | None:
        message = str(exc)
        match = RATE_LIMIT_WAIT_HINT_RE.search(message)
        if match is None:
            return None
        try:
            parsed = float(match.group(1))
        except (TypeError, ValueError):
            return None
        if parsed <= 0:
            return None
        return min(parsed + 0.25, 30.0)

    # -------------------------------------------------------------------------
    def retry_backoff_seconds(self, attempt: int, *, exc: Exception | None = None) -> float:
        if exc is not None:
            hinted_wait = self.extract_rate_limit_wait_hint_seconds(exc)
            if hinted_wait is not None:
                return hinted_wait
        normalized_attempt = max(int(attempt), 1)
        return min(8.0, 0.75 * (2 ** (normalized_attempt - 1)))

    # -------------------------------------------------------------------------
    async def extract_from_payload(
        self,
        payload: PatientData,
        *,
        progress_callback: Callable[[float], None] | None = None,
    ) -> tuple[PatientLabTimeline, LiverInjuryOnsetContext | None]:
        primary_labs_text = self.clean_text(payload.laboratory_analysis)
        supplemental_anamnesis_text = self.clean_text(payload.anamnesis)
        timeline_entries: list[ClinicalLabEntry] = []
        onset_context: LiverInjuryOnsetContext | None = None
        self.emit_progress(progress_callback, 0.0)

        merged_source_text = "\n\n".join(
            block for block in (primary_labs_text, supplemental_anamnesis_text) if block
        )
        if merged_source_text:
            await self.ensure_client()
            if self.client is None:
                raise RuntimeError("LLM client is not initialized for lab extraction")
            chunks = self.chunk_text(merged_source_text)
            llm_entries: list[ClinicalLabEntry] = []
            llm_onset: LiverInjuryOnsetContext | None = None
            for index, chunk in enumerate(chunks, start=1):
                user_prompt = (
                    "Extract longitudinal liver-related labs and onset clues from this clinical chunk.\n"
                    f"[Chunk {index}/{len(chunks)}]\n{chunk}"
                )
                for attempt in range(1, self.extraction_retry_attempts + 1):
                    try:
                        parsed = await self.client.llm_structured_call(
                            model=self.model,
                            system_prompt=CLINICAL_LAB_EXTRACTION_PROMPT.strip(),
                            user_prompt=user_prompt,
                            schema=LabExtractionPayload,
                            temperature=self.temperature,
                            use_json_mode=True,
                            max_repair_attempts=2,
                        )
                        break
                    except Exception as exc:
                        if attempt >= self.extraction_retry_attempts:
                            raise RuntimeError("Failed to extract labs from anamnesis") from exc
                        delay = self.retry_backoff_seconds(attempt, exc=exc)
                        logger.warning(
                            (
                            "Retrying clinical lab extraction for chunk %d/%d "
                            "(attempt %d/%d, delay %.2fs): %s"
                        ),
                            index,
                            len(chunks),
                            attempt,
                            self.extraction_retry_attempts,
                            delay,
                            exc,
                        )
                        await asyncio.sleep(delay)
                llm_entries.extend(parsed.entries)
                if llm_onset is None and parsed.onset_context is not None:
                    llm_onset = parsed.onset_context
                self.emit_progress(progress_callback, (index / max(len(chunks), 1)) * 0.7)
            timeline_entries.extend(llm_entries)
            onset_context = llm_onset

        self.emit_progress(progress_callback, 0.85)

        normalized: list[ClinicalLabEntry] = []
        seen: set[tuple[str, str, str, str]] = set()
        for entry in timeline_entries:
            prepared = self.normalize_entry(entry, visit_date=payload.visit_date)
            if prepared is None:
                continue
            key = self.dedupe_key(prepared)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(prepared)

        def sort_key(item: ClinicalLabEntry) -> tuple[int, str, str]:
            if item.sample_date:
                parsed = self.try_parse_date(item.sample_date)
                if parsed is not None:
                    return (0, parsed.isoformat(), item.marker_name)
            return (1, item.relative_time or "", item.marker_name)

        normalized.sort(key=sort_key)
        self.emit_progress(progress_callback, 1.0)
        return PatientLabTimeline(entries=normalized), onset_context

