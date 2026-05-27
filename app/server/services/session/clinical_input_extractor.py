from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from configurations.llm_configs import LLMRuntimeConfig
from configurations.startup import get_server_settings
from domain.clinical.entities import (
    ClinicalSectionExtractionResult,
    ClinicalSectionLineRange,
)
from services.llm.client_runtime import ensure_runtime_client
from services.llm.provider_factory import select_llm_provider
from services.session.clinical_section_parsers import (
    extract_required_dili_sections,
    missing_required_section_names,
    verify_verbatim_section_coherence,
)


class ClinicalInputExtractionError(RuntimeError):
    pass


def validate_extracted_sections_against_source(
    source_text: str,
    anamnesis: str,
    therapy: str,
    lab_analysis: str,
) -> bool:
    return all(section and section in source_text for section in (anamnesis, therapy, lab_analysis))


class ClinicalInputExtractor:
    def __init__(
        self,
        *,
        client: Any | None = None,
        timeout_s: float = get_server_settings().runtime.default_llm_timeout,
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

    def _deterministic_extract(self, clinical_input: str) -> ClinicalSectionExtractionResult:
        sections = extract_required_dili_sections(clinical_input)
        missing = missing_required_section_names(sections)
        if missing:
            raise ClinicalInputExtractionError(
                f"Missing required titled sections: {', '.join(missing)}"
            )
        for key, section in sections.items():
            if not section.text.strip():
                raise ClinicalInputExtractionError(f"Section '{key}' is empty.")
            if not verify_verbatim_section_coherence(clinical_input, section):
                raise ClinicalInputExtractionError(
                    f"Section '{key}' does not match a coherent verbatim span."
                )
        therapy = sections["therapy"]
        labs = sections["laboratory_history"]
        anamnesis = sections["anamnesis"]
        strict_verbatim = validate_extracted_sections_against_source(
            clinical_input,
            anamnesis=anamnesis.text,
            therapy=therapy.text,
            lab_analysis=labs.text,
        )
        if not strict_verbatim:
            raise ClinicalInputExtractionError("Deterministic section extraction failed source grounding.")
        return ClinicalSectionExtractionResult(
            source_text=clinical_input,
            anamnesis=anamnesis.text,
            drugs=therapy.text,
            laboratory_analysis=labs.text,
            line_ranges={
                "anamnesis": [
                    ClinicalSectionLineRange(
                        start_line=anamnesis.line_start,
                        end_line=anamnesis.line_end,
                    )
                ],
                "drugs": [
                    ClinicalSectionLineRange(
                        start_line=therapy.line_start,
                        end_line=therapy.line_end,
                    )
                ],
                "laboratory_analysis": [
                    ClinicalSectionLineRange(
                        start_line=labs.line_start,
                        end_line=labs.line_end,
                    )
                ],
            },
            confidence=0.95 if strict_verbatim else 0.7,
            metadata={
                "sections": {
                    key: {
                        "canonical_key": value.canonical_key,
                        "raw_heading": value.raw_heading,
                        "normalized_heading": value.normalized_heading,
                        "match_strategy": value.match_strategy,
                        "score": value.confidence_score,
                        "line_span": [value.line_start, value.line_end],
                        "char_span": [value.body_start, value.body_end],
                        "verbatim_coherent": value.verbatim_coherent,
                    }
                    for key, value in sections.items()
                }
            },
        )

    @staticmethod
    def _raise_extraction_failed(reason: str) -> None:
        raise ClinicalInputExtractionError(f"Unable to extract clinical sections: {reason}")

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

        deterministic: ClinicalSectionExtractionResult | None = None
        try:
            deterministic = self._deterministic_extract(clinical_input)
        except ValueError as exc:
            if progress_callback is not None:
                progress_callback(1.0)
            self._raise_extraction_failed(str(exc))
        except ClinicalInputExtractionError:
            if progress_callback is not None:
                progress_callback(1.0)
            raise
        if progress_callback is not None:
            progress_callback(1.0)
        if deterministic is None:
            self._raise_extraction_failed("deterministic extraction returned no result")
        assert deterministic is not None
        return deterministic
