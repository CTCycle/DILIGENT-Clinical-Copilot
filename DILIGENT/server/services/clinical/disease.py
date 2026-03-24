from __future__ import annotations

import asyncio
import contextlib
import re
import unicodedata
from collections.abc import Callable
from typing import Any

from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.configurations import LLMRuntimeConfig, server_settings
from DILIGENT.server.domain.clinical import DiseaseContextEntry, PatientDiseaseContext
from DILIGENT.server.models.prompts import ANAMNESIS_DISEASE_EXTRACTION_PROMPT
from DILIGENT.server.models.providers import initialize_llm_client
from DILIGENT.server.services.text.normalization import normalize_token


###############################################################################
class DiseaseExtractor:
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
        evidence = self.sanitize_text(entry.evidence, max_words=30)
        return DiseaseContextEntry(
            name=name,
            occurrence_time=occurrence_time,
            chronic=entry.chronic,
            hepatic_related=entry.hepatic_related,
            evidence=evidence,
        )

    # -------------------------------------------------------------------------
    def entry_score(self, entry: DiseaseContextEntry) -> int:
        score = 1
        if entry.occurrence_time:
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
    async def extract_diseases_from_anamnesis(
        self,
        anamnesis: str | None,
        *,
        progress_callback: Callable[[float], None] | None = None,
    ) -> PatientDiseaseContext:
        cleaned = self.clean_text(anamnesis)
        if not cleaned:
            return PatientDiseaseContext(entries=[])

        await self.ensure_client()
        if self.client is None:
            raise RuntimeError("LLM client is not initialized for disease extraction")

        chunks = self.chunk_text(cleaned)
        raw_entries: list[DiseaseContextEntry] = []
        self.emit_progress(progress_callback, 0.0)
        for index, chunk in enumerate(chunks, start=1):
            user_prompt = (
                "Extract diseases from this anamnesis chunk, with temporal and hepatic metadata.\n"
                f"[Chunk {index}/{len(chunks)}]\n{chunk}"
            )
            try:
                parsed = await self.client.llm_structured_call(
                    model=self.model,
                    system_prompt=ANAMNESIS_DISEASE_EXTRACTION_PROMPT.strip(),
                    user_prompt=user_prompt,
                    schema=PatientDiseaseContext,
                    temperature=self.temperature,
                    use_json_mode=True,
                    max_repair_attempts=2,
                )
            except Exception as exc:
                raise RuntimeError("Failed to extract diseases from anamnesis") from exc
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
        logger.info(
            "Anamnesis disease extraction produced %s entries (%s raw).",
            len(deduplicated),
            len(raw_entries),
        )
        self.emit_progress(progress_callback, 1.0)
        return PatientDiseaseContext(entries=deduplicated)
