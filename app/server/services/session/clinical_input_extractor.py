from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from services.llm.runtime_config import LLMRuntimeConfig
from configurations.startup import get_server_settings
from domain.clinical.entities import (
    ClinicalSectionExtractionResult,
)
from services.llm.client_runtime import ensure_runtime_client
from services.llm.provider_factory import select_llm_provider
from services.session.clinical_section_parsers import (
    extract_required_dili_sections,
    missing_required_section_names,
    verify_verbatim_section_coherence,
)
from services.session.text_section_parser import (
    build_section_extraction_from_initial_text,
    parse_initial_text_sections,
)


###############################################################################
class ClinicalInputExtractionError(RuntimeError):
    pass


###############################################################################
def validate_extracted_sections_against_source(
    source_text: str,
    anamnesis: str,
    therapy: str,
    lab_analysis: str,
) -> bool:
    return all(
        section and section in source_text
        for section in (anamnesis, therapy, lab_analysis)
    )


###############################################################################
class ClinicalInputExtractor:
    # -------------------------------------------------------------------------
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
    def _deterministic_extract(
        self, clinical_input: str
    ) -> ClinicalSectionExtractionResult:
        source_text = (clinical_input or "").replace("\r\n", "\n").replace("\r", "\n")
        sections = extract_required_dili_sections(source_text)
        missing = missing_required_section_names(sections)
        if missing:
            raise ClinicalInputExtractionError(
                f"Missing required titled sections: {', '.join(missing)}"
            )
        for key, section in sections.items():
            if not section.text.strip():
                raise ClinicalInputExtractionError(f"Section '{key}' is empty.")
            if not verify_verbatim_section_coherence(source_text, section):
                raise ClinicalInputExtractionError(
                    f"Section '{key}' does not match a coherent verbatim span."
                )
        therapy = sections["therapy"]
        labs = sections["laboratory_history"]
        anamnesis = sections["anamnesis"]
        strict_verbatim = validate_extracted_sections_against_source(
            source_text,
            anamnesis=anamnesis.text,
            therapy=therapy.text,
            lab_analysis=labs.text,
        )
        if not strict_verbatim:
            raise ClinicalInputExtractionError(
                "Deterministic section extraction failed source grounding."
            )
        parse_result = parse_initial_text_sections(source_text)
        extraction = build_section_extraction_from_initial_text(
            parse_result,
            source_text,
        )
        return extraction.model_copy(
            update={"confidence": 0.95 if strict_verbatim else 0.7}
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _raise_extraction_failed(reason: str) -> None:
        raise ClinicalInputExtractionError(
            f"Unable to extract clinical sections: {reason}"
        )

    # -------------------------------------------------------------------------
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
