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
from common.prompts.extraction import ANAMNESIS_DISEASE_EXTRACTION_PROMPT
from services.clinical.deterministic_extraction import extract_deterministic_diseases
from services.llm.client_runtime import ensure_runtime_client
from services.llm.provider_factory import select_llm_provider
from common.utils.text_utils import normalize_token

###############################################################################
RATE_LIMIT_WAIT_HINT_RE = re.compile(
    r"please\s+try\s+again\s+in\s+(\d+(?:\.\d+)?)s",
    re.IGNORECASE,
)

###############################################################################
class DiseaseExtractor:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        client: Any | None = None,
        temperature: float = 0.0,
        timeout_s: float = get_server_settings().runtime.disease_llm_timeout,
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
    def minimum_timeout_s() -> float:
        return float(get_server_settings().runtime.minimum_llm_timeout)

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
            source_span=entry.source_span,
            confidence=entry.confidence,
            attribution=entry.attribution,
        )

    # -------------------------------------------------------------------------
    def validate_entry_evidence(
        self,
        entry: DiseaseContextEntry,
        source_text: str,
    ) -> DiseaseContextEntry:
        evidence = (entry.evidence or "").strip()
        if evidence and evidence in source_text:
            start = source_text.index(evidence)
            return entry.model_copy(
                update={
                    "source_span": entry.source_span or [start, start + len(evidence)],
                    "confidence": entry.confidence or "high",
                    "attribution": entry.attribution or "patient",
                }
            )
        name = (entry.name or "").strip()
        start = source_text.casefold().find(name.casefold()) if name else -1
        if start >= 0:
            end = start + len(name)
            return entry.model_copy(
                update={
                    "evidence": source_text[start:end],
                    "source_span": [start, end],
                    "confidence": entry.confidence or "moderate",
                    "attribution": entry.attribution or "patient",
                }
            )
        return entry.model_copy(
            update={
                "confidence": "low",
                "attribution": entry.attribution or "unclear",
            }
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
    def format_expected_candidates(entries: list[DiseaseContextEntry]) -> str:
        lines: list[str] = []
        for entry in entries[:30]:
            parts = [entry.name]
            if entry.evidence:
                parts.append(f"evidence={entry.evidence[:180]}")
            if entry.chronic is not None:
                parts.append(f"chronic={'yes' if entry.chronic else 'no'}")
            if entry.hepatic_related is not None:
                parts.append(
                    f"hepatic_related={'yes' if entry.hepatic_related else 'no'}"
                )
            lines.append("- " + " | ".join(parts))
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    def validate_disease_coverage(
        self,
        entries: list[DiseaseContextEntry],
        candidates: list[DiseaseContextEntry],
    ) -> tuple[list[str], list[DiseaseContextEntry]]:
        present = {normalize_token(entry.name) for entry in entries}
        missing = [
            candidate
            for candidate in candidates
            if normalize_token(candidate.name) not in present
        ]
        feedback: list[str] = []
        if missing:
            feedback.append(
                "Missing grounded disease candidates: "
                + "; ".join(entry.name for entry in missing[:12])
            )
        return feedback, missing

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

        self.emit_progress(progress_callback, 0.0)
        deterministic = extract_deterministic_diseases(cleaned)
        deterministic_candidates = self.deduplicate_entries(
            list(deterministic.context.entries)
        )
        try:
            await self.ensure_client()
            if self.client is None:
                raise RuntimeError("LLM client is not initialized for disease extraction")
            candidate_text = self.format_expected_candidates(deterministic_candidates)
            user_prompt = (
                "Extract diseases from this full anamnesis text, with temporal and hepatic metadata.\n"
                f"{cleaned}"
            )
            if candidate_text:
                user_prompt = (
                    f"{user_prompt}\n\n"
                    "Grounded candidate checklist from source text. Use it to avoid omissions, but still "
                    "return only clinically relevant diseases/conditions supported by the source:\n"
                    f"{candidate_text}"
                )
            last_wrong_output = ""
            last_errors: list[str] = []
            max_attempts = max(1, self.extraction_retry_attempts + 1)
            for attempt in range(1, max_attempts + 1):
                if attempt > 1:
                    user_prompt = (
                        "Retry the disease extraction. The previous output was rejected by semantic validation.\n"
                        "Return only clinically relevant diseases/conditions explicitly supported by the source.\n\n"
                        f"Validation errors:\n- {'; '.join(last_errors)}\n\n"
                        f"Grounded candidate checklist:\n{candidate_text}\n\n"
                        f"Previous wrong output:\n{last_wrong_output}\n\n"
                        f"Source text:\n{cleaned}"
                    )
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
                        timeout=max(self.minimum_timeout_s(), float(self.timeout_s)),
                    )
                except Exception as exc:
                    if attempt >= max_attempts:
                        raise
                    delay = self.retry_backoff_seconds(attempt, exc=exc)
                    logger.warning(
                        (
                            "Retrying anamnesis disease extraction "
                            "(attempt %d/%d, delay %.2fs): %s"
                        ),
                        attempt,
                        max_attempts,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                    continue
                last_wrong_output = parsed.model_dump_json()
                normalized_entries: list[DiseaseContextEntry] = []
                for entry in parsed.entries:
                    normalized = self.normalize_entry(entry)
                    if normalized is not None:
                        normalized_entries.append(
                            self.validate_entry_evidence(normalized, cleaned)
                        )
                deduplicated = self.deduplicate_entries(normalized_entries)
                feedback, missing_candidates = self.validate_disease_coverage(
                    deduplicated,
                    deterministic_candidates,
                )
                coverage_ok = not missing_candidates or (
                    len(deduplicated) >= max(1, int(len(deterministic_candidates) * 0.8))
                )
                if deduplicated and coverage_ok:
                    self.emit_progress(progress_callback, 1.0)
                    logger.info(
                        "Anamnesis disease LLM extraction produced %s entries (%s raw LLM entries).",
                        len(deduplicated),
                        len(parsed.entries),
                    )
                    return PatientDiseaseContext(entries=deduplicated)
                if attempt >= max_attempts and deduplicated:
                    logger.warning(
                        "LLM disease extraction missed %d grounded candidates after retry; merging deterministic fallback candidates.",
                        len(missing_candidates),
                    )
                    merged = self.deduplicate_entries([*deduplicated, *missing_candidates])
                    self.emit_progress(progress_callback, 1.0)
                    return PatientDiseaseContext(entries=merged)
                last_errors = feedback or [
                    "The model returned no valid disease entries despite disease-like source evidence."
                ]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Anamnesis disease LLM extraction failed; using deterministic fallback: %s",
                exc,
            )

        self.emit_progress(progress_callback, 1.0)
        return PatientDiseaseContext(
            entries=deterministic_candidates
        )
