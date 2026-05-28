from __future__ import annotations

import asyncio
import re
import unicodedata
from collections.abc import Callable
from typing import Any

from common.utils.logger import logger
from configurations.llm_configs import LLMRuntimeConfig
from configurations.startup import get_server_settings
from domain.clinical.entities import (
    DiseaseContextEntry,
    PatientDiseaseContext,
)
from services.clinical.deterministic_extraction import extract_deterministic_diseases
from services.llm.client_runtime import ensure_runtime_client
from services.llm.prompts import (
    ANAMNESIS_DISEASE_EXTRACTION_PROMPT,
)
from services.llm.provider_factory import select_llm_provider
from services.text.normalization import normalize_token

###############################################################################
RATE_LIMIT_WAIT_HINT_RE = re.compile(
    r"please\s+try\s+again\s+in\s+(\d+(?:\.\d+)?)s",
    re.IGNORECASE,
)


###############################################################################
class DiseaseExtractor:
    CHUNK_MAX_CHARS = 2600

    def __init__(
        self,
        *,
        client: Any | None = None,
        temperature: float = 0.0,
        timeout_s: float = get_server_settings().runtime.default_llm_timeout,
    ) -> None:
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.client: Any | None = client
        self.model: str = ""
        # Prefer fast deterministic fallback over long retry loops.
        self.extraction_retry_attempts = 1
        self.client_lock = asyncio.Lock()
        self.client_loop_id: int | None = None
        self.forced_provider: str | None = None
        self.forced_model: str | None = None
        if client is None:
            self.client_provider: str | None = None
            self.runtime_revision = -1
        else:
            self.client_provider = "injected"
            self.runtime_revision = LLMRuntimeConfig.get_revision()

    # -------------------------------------------------------------------------
    async def ensure_client(self) -> None:
        revision = LLMRuntimeConfig.get_revision()
        resolved_provider, resolved_model = LLMRuntimeConfig.resolve_provider_and_model(
            "parser"
        )
        provider = self.forced_provider or resolved_provider
        model = self.forced_model or resolved_model
        await ensure_runtime_client(
            self,
            provider=provider,
            model=model,
            revision=revision,
            client_factory=lambda selected_provider, selected_model: (
                select_llm_provider(
                    provider=selected_provider,
                    default_model=selected_model,
                    timeout_s=self.timeout_s,
                    max_retries=0,
                )
            ),
        )

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
        lines: list[str] = []
        for raw_line in normalized.split("\n"):
            stripped = raw_line.strip()
            if stripped:
                lines.append(stripped)
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
    def sanitize_text(self, value: str | None, *, max_words: int) -> str | None:
        if value is None:
            return None
        candidate = re.sub(r"\s+", " ", str(value)).strip(" \t,;:")
        if not candidate:
            return None
        if len(candidate.split()) > max_words:
            return None
        return candidate

    # -------------------------------------------------------------------------
    def normalize_entry(self, entry: DiseaseContextEntry) -> DiseaseContextEntry | None:
        name = self.sanitize_text(entry.name, max_words=12)
        if name is None:
            return None
        occurrence_time = self.sanitize_text(entry.occurrence_time, max_words=10)
        timeline = self.sanitize_text(entry.timeline, max_words=16)
        severity = self.sanitize_text(entry.severity, max_words=8)
        diagnosis_status = self.sanitize_text(entry.diagnosis_status, max_words=10)
        symptoms = self.sanitize_text(entry.symptoms, max_words=30)
        clinical_context = self.sanitize_text(entry.clinical_context, max_words=30)
        evidence = self.sanitize_text(entry.evidence, max_words=30)
        return DiseaseContextEntry(
            name=name,
            occurrence_time=occurrence_time,
            timeline=timeline,
            severity=severity,
            diagnosis_status=diagnosis_status,
            symptoms=symptoms,
            clinical_context=clinical_context,
            chronic=entry.chronic,
            hepatic_related=entry.hepatic_related,
            evidence=evidence,
        )

    # -------------------------------------------------------------------------
    def entry_score(self, entry: DiseaseContextEntry) -> int:
        score = 1
        if entry.occurrence_time:
            score += 1
        if entry.timeline:
            score += 1
        if entry.severity:
            score += 1
        if entry.diagnosis_status:
            score += 1
        if entry.symptoms:
            score += 1
        if entry.clinical_context:
            score += 1
        if entry.chronic is not None:
            score += 1
        if entry.hepatic_related is not None:
            score += 1
        if entry.evidence:
            score += 1
        return score

    # -------------------------------------------------------------------------
    def deduplicate_entries(
        self,
        entries: list[DiseaseContextEntry],
    ) -> list[DiseaseContextEntry]:
        selected: dict[str, DiseaseContextEntry] = {}
        order: list[str] = []
        for entry in entries:
            lookup_key = normalize_token(entry.name)
            if not lookup_key:
                continue
            existing = selected.get(lookup_key)
            if existing is None:
                selected[lookup_key] = entry
                order.append(lookup_key)
                continue
            if self.entry_score(entry) > self.entry_score(existing):
                selected[lookup_key] = entry
        return [selected[key] for key in order if key in selected]

    # -------------------------------------------------------------------------
    @staticmethod
    def extract_rate_limit_wait_hint_seconds(exc: Exception) -> float | None:
        message = str(exc)
        match = RATE_LIMIT_WAIT_HINT_RE.search(message)
        if match is None:
            return None
        try:
            parsed = float(match.group(1))
        except TypeError, ValueError:
            return None
        if parsed <= 0:
            return None
        return min(parsed + 0.25, 30.0)

    # -------------------------------------------------------------------------
    def retry_backoff_seconds(
        self, attempt: int, *, exc: Exception | None = None
    ) -> float:
        if exc is not None:
            hinted_wait = self.extract_rate_limit_wait_hint_seconds(exc)
            if hinted_wait is not None:
                return hinted_wait
        normalized_attempt = max(int(attempt), 1)
        return min(2.0, 0.5 * (2 ** (normalized_attempt - 1)))

    # -------------------------------------------------------------------------
    async def extract_diseases_from_anamnesis(
        self,
        anamnesis: str | None,
        *,
        progress_callback: Callable[[float], None] | None = None,
    ) -> PatientDiseaseContext:
        cleaned = self.clean_text(anamnesis)
        if not cleaned:
            return PatientDiseaseContext(entries=[])

        deterministic = extract_deterministic_diseases(cleaned)
        accumulated_entries = list(deterministic.context.entries)
        self.emit_progress(progress_callback, 0.15 if accumulated_entries else 0.0)

        unresolved_source = "\n".join(deterministic.unresolved_lines).strip()
        if not unresolved_source:
            self.emit_progress(progress_callback, 1.0)
            return PatientDiseaseContext(
                entries=self.deduplicate_entries(accumulated_entries)
            )

        await self.ensure_client()
        if self.client is None:
            self.emit_progress(progress_callback, 1.0)
            return PatientDiseaseContext(
                entries=self.deduplicate_entries(accumulated_entries)
            )

        chunks = self.chunk_text(unresolved_source)
        raw_entries: list[DiseaseContextEntry] = []
        self.emit_progress(progress_callback, 0.0)
        for index, chunk in enumerate(chunks, start=1):
            user_prompt = (
                "Extract diseases from this anamnesis chunk, with temporal and hepatic metadata.\n"
                f"[Chunk {index}/{len(chunks)}]\n{chunk}"
            )
            parsed: PatientDiseaseContext | None = None
            for attempt in range(1, self.extraction_retry_attempts + 1):
                try:
                    parsed = await asyncio.wait_for(
                        self.client.llm_structured_call(
                            model=self.model,
                            system_prompt=ANAMNESIS_DISEASE_EXTRACTION_PROMPT.strip(),
                            user_prompt=user_prompt,
                            schema=PatientDiseaseContext,
                            temperature=self.temperature,
                            use_json_mode=True,
                            max_repair_attempts=1,
                        ),
                        timeout=max(5.0, float(self.timeout_s)),
                    )
                    break
                except Exception as exc:
                    if attempt >= self.extraction_retry_attempts:
                        raise RuntimeError(
                            "Failed to extract diseases from anamnesis"
                        ) from exc
                    delay = self.retry_backoff_seconds(attempt, exc=exc)
                    logger.warning(
                        (
                            "Retrying anamnesis disease extraction for chunk %d/%d "
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
            if parsed is None:
                raise RuntimeError("Failed to extract diseases from anamnesis")
            raw_entries.extend(parsed.entries)
            self.emit_progress(
                progress_callback,
                (index / max(len(chunks), 1)) * 0.9,
            )

        normalized_entries: list[DiseaseContextEntry] = []
        for entry in raw_entries:
            normalized = self.normalize_entry(entry)
            if normalized is not None:
                normalized_entries.append(normalized)

        deduplicated = self.deduplicate_entries(normalized_entries)
        deduplicated = self.deduplicate_entries([*accumulated_entries, *deduplicated])
        logger.info(
            "Anamnesis disease extraction produced %s entries (%s raw LLM entries, %s deterministic entries).",
            len(deduplicated),
            len(raw_entries),
            len(accumulated_entries),
        )
        self.emit_progress(progress_callback, 1.0)
        return PatientDiseaseContext(entries=deduplicated)
