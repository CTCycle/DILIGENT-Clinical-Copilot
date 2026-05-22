from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Any

from configurations.llm_configs import LLMRuntimeConfig
from configurations.startup import server_settings
from domain.clinical.extraction import LlmClinicalSectionTextDraft
from domain.clinical.entities import ClinicalSectionExtractionResult
from services.llm.client_runtime import ensure_runtime_client
from services.llm.prompts import CLINICAL_SECTION_EXTRACTION_PROMPT
from services.llm.provider_factory import select_llm_provider
from services.session.clinical_section_parsers import (
    extract_sections_from_markers,
    find_section_markers,
    validate_sections_against_source,
)


class ClinicalInputExtractionError(RuntimeError):
    pass


def validate_extracted_sections_against_source(
    source_text: str,
    anamnesis: str,
    therapy: str,
    lab_analysis: str,
) -> bool:
    return validate_sections_against_source(
        source_text,
        {
            "anamnesis": anamnesis,
            "drugs": therapy,
            "laboratory_analysis": lab_analysis,
        },
    )


class ClinicalInputExtractor:
    def __init__(
        self,
        *,
        client: Any | None = None,
        timeout_s: float = server_settings.runtime.default_llm_timeout,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.client: Any | None = client
        self.model: str = ""
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

    async def ensure_client(self) -> None:
        revision = LLMRuntimeConfig.get_revision()
        resolved_provider, resolved_model = LLMRuntimeConfig.resolve_provider_and_model("parser")
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

    def _deterministic_extract(self, clinical_input: str) -> ClinicalSectionExtractionResult | None:
        markers = find_section_markers(clinical_input)
        sections = extract_sections_from_markers(clinical_input, markers)
        if sections is None:
            return None
        strict_verbatim = validate_sections_against_source(clinical_input, sections)
        return ClinicalSectionExtractionResult(
            source_text=clinical_input,
            anamnesis=sections["anamnesis"],
            drugs=sections["drugs"],
            laboratory_analysis=sections["laboratory_analysis"],
            line_ranges={},
            confidence=0.95 if strict_verbatim else 0.7,
        )

    def _coarse_fallback_extract(self, clinical_input: str) -> ClinicalSectionExtractionResult:
        lines = [line for line in clinical_input.splitlines() if line.strip()]
        if not lines:
            raise ClinicalInputExtractionError("clinical_input is empty")
        chunk = max(len(lines) // 3, 1)
        anamnesis = "\n".join(lines[:chunk]).strip()
        drugs = "\n".join(lines[chunk : chunk * 2]).strip()
        laboratory_analysis = "\n".join(lines[chunk * 2 :]).strip()
        if not drugs:
            drugs = anamnesis
        if not laboratory_analysis:
            laboratory_analysis = "\n".join(lines[-chunk:]).strip() or anamnesis
        return ClinicalSectionExtractionResult(
            source_text=clinical_input,
            anamnesis=anamnesis or clinical_input.strip(),
            drugs=drugs or clinical_input.strip(),
            laboratory_analysis=laboratory_analysis or clinical_input.strip(),
            line_ranges={},
            confidence=0.3,
        )

    @staticmethod
    def _normalize_json_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    async def extract(
        self,
        *,
        clinical_input: str,
        progress_callback: Callable[[float], None] | None = None,
    ) -> ClinicalSectionExtractionResult:
        if not clinical_input.strip():
            raise ClinicalInputExtractionError("clinical_input is empty")
        if progress_callback is not None:
            progress_callback(0.0)

        deterministic = self._deterministic_extract(clinical_input)
        if deterministic is not None:
            if progress_callback is not None:
                progress_callback(1.0)
            return deterministic

        try:
            await self.ensure_client()
        except Exception:
            if progress_callback is not None:
                progress_callback(1.0)
            return self._coarse_fallback_extract(clinical_input)
        if self.client is None:
            if progress_callback is not None:
                progress_callback(1.0)
            return self._coarse_fallback_extract(clinical_input)

        try:
            extraction = await self.client.llm_structured_call(
                model=self.model,
                system_prompt=CLINICAL_SECTION_EXTRACTION_PROMPT,
                user_prompt=clinical_input,
                schema=LlmClinicalSectionTextDraft,
                temperature=0.0,
                use_json_mode=True,
                max_repair_attempts=1,
            )
        except Exception as exc:  # noqa: BLE001
            if deterministic is not None:
                if progress_callback is not None:
                    progress_callback(1.0)
                return deterministic
            if progress_callback is not None:
                progress_callback(1.0)
            return self._coarse_fallback_extract(clinical_input)

        draft = LlmClinicalSectionTextDraft.model_validate(extraction)
        anamnesis = self._normalize_json_text(draft.anamnesis)
        therapy = self._normalize_json_text(draft.therapy)
        lab_analysis = self._normalize_json_text(draft.lab_analysis)
        if not anamnesis or not therapy or not lab_analysis:
            if deterministic is not None:
                if progress_callback is not None:
                    progress_callback(1.0)
                return deterministic
            if progress_callback is not None:
                progress_callback(1.0)
            return self._coarse_fallback_extract(clinical_input)
        if not validate_extracted_sections_against_source(clinical_input, anamnesis, therapy, lab_analysis):
            if deterministic is not None:
                if progress_callback is not None:
                    progress_callback(1.0)
                return deterministic
            if progress_callback is not None:
                progress_callback(1.0)
            return self._coarse_fallback_extract(clinical_input)
        if progress_callback is not None:
            progress_callback(1.0)
        return ClinicalSectionExtractionResult(
            source_text=clinical_input,
            anamnesis=anamnesis,
            drugs=therapy,
            laboratory_analysis=lab_analysis,
            line_ranges={},
            confidence=0.5,
        )
