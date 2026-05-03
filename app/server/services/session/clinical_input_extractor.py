from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Mapping
from typing import Any

from common.utils.logger import logger
from configurations.llm_configs import LLMRuntimeConfig
from configurations.startup import server_settings
from domain.clinical.entities import (
    ClinicalSectionExtractionResult,
    ClinicalSectionLineRange,
    LlmClinicalSectionExtractionDraft,
)
from domain.clinical.sections import ClinicalSectionKey, SECTION_FRAGMENT_JOINER, SECTION_KEYS
from services.llm.client_runtime import ensure_runtime_client
from services.llm.prompts import CLINICAL_SECTION_EXTRACTION_PROMPT
from services.llm.provider_factory import select_llm_provider
from services.session.clinical_section_parsers import (
    PlainTextSectionParser,
    SectionFragmentSlice,
)
from services.text.vocabulary import get_text_normalization_snapshot


class ClinicalInputExtractionError(RuntimeError):
    pass


class ClinicalInputExtractor:
    def __init__(
        self,
        *,
        client: Any | None = None,
        timeout_s: float = server_settings.external_data.default_llm_timeout,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.client: Any | None = client
        self.model: str = ""
        self.client_lock = asyncio.Lock()
        self.client_loop_id: int | None = None
        self.forced_provider: str | None = None
        self.forced_model: str | None = None

        snapshot = get_text_normalization_snapshot()
        if not any(snapshot.section_title_aliases.values()):
            logger.warning(
                "Section title alias vocabulary is unavailable; deterministic section parsing is disabled."
            )
        self._section_parsers = (
            PlainTextSectionParser(section_title_aliases=snapshot.section_title_aliases),
        )

        if client is None:
            self.client_provider: str | None = None
            self.runtime_revision = -1
        else:
            self.client_provider = "injected"
            self.runtime_revision = LLMRuntimeConfig.get_revision()

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
                )
            ),
        )

    @classmethod
    def _build_result_from_fragments(
        cls,
        *,
        source_text: str,
        fragments: list[SectionFragmentSlice],
        confidence: float,
    ) -> ClinicalSectionExtractionResult:
        if not fragments:
            raise ClinicalInputExtractionError("Clinical input does not contain structured sections.")

        section_fragments: dict[ClinicalSectionKey, list[SectionFragmentSlice]] = {
            section: [] for section in SECTION_KEYS
        }

        for fragment in fragments:
            if fragment.start < 0 or fragment.end > len(source_text) or fragment.start >= fragment.end:
                raise ClinicalInputExtractionError("Clinical input extraction produced invalid fragment offsets.")
            fragment_text = source_text[fragment.start : fragment.end]
            if not fragment_text.strip():
                raise ClinicalInputExtractionError("Clinical input extraction produced an empty section fragment.")
            section_fragments[fragment.section].append(fragment)

        missing_sections = [section for section, values in section_fragments.items() if not values]
        if missing_sections:
            raise ClinicalInputExtractionError(
                "Clinical input must contain anamnesis, current therapy, and laboratory analysis sections."
            )

        section_texts: dict[ClinicalSectionKey, str] = {}
        for section in SECTION_KEYS:
            ordered = sorted(section_fragments[section], key=lambda item: (item.start, item.end))
            texts = [source_text[fragment.start : fragment.end] for fragment in ordered]
            section_texts[section] = SECTION_FRAGMENT_JOINER.join(texts)

        return ClinicalSectionExtractionResult(
            source_text=source_text,
            anamnesis=section_texts["anamnesis"],
            drugs=section_texts["drugs"],
            laboratory_analysis=section_texts["laboratory_analysis"],
            line_ranges={},
            confidence=max(0.0, min(1.0, float(confidence))),
        )

    @staticmethod
    def _build_numbered_source(source_text: str) -> str:
        return "\n".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(source_text.splitlines(), start=1)
        )

    @staticmethod
    def _normalize_for_presence(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    @classmethod
    def _assert_sections_exist_in_source(
        cls,
        *,
        source_text: str,
        sections: Mapping[ClinicalSectionKey, str],
    ) -> None:
        normalized_source = cls._normalize_for_presence(source_text)
        for section, text in sections.items():
            normalized_text = cls._normalize_for_presence(text)
            if not normalized_text:
                raise ClinicalInputExtractionError(f"Clinical input extraction returned empty {section}.")
            if normalized_text not in normalized_source:
                raise ClinicalInputExtractionError(
                    f"Clinical input extraction returned {section} content not present in source text."
                )

    @classmethod
    def _build_result_from_line_ranges(
        cls,
        *,
        source_text: str,
        draft: LlmClinicalSectionExtractionDraft,
    ) -> ClinicalSectionExtractionResult:
        lines = source_text.splitlines()
        if not lines:
            raise ClinicalInputExtractionError("clinical_input is empty")

        section_texts: dict[ClinicalSectionKey, str] = {}
        ranges_by_section: dict[ClinicalSectionKey, list[ClinicalSectionLineRange]] = {}

        for section in SECTION_KEYS:
            ranges = getattr(draft, section)
            if not ranges:
                raise ClinicalInputExtractionError(
                    "Clinical input must contain anamnesis, current therapy, and laboratory analysis sections."
                )

            chunks: list[str] = []
            normalized_ranges: list[ClinicalSectionLineRange] = []
            for line_range in ranges:
                start_idx = line_range.start_line - 1
                end_idx = line_range.end_line - 1
                if start_idx < 0 or end_idx >= len(lines):
                    raise ClinicalInputExtractionError("Clinical input extraction returned out-of-range lines.")
                if start_idx > end_idx:
                    raise ClinicalInputExtractionError("Clinical input extraction returned an invalid line range.")

                chunk = "\n".join(lines[start_idx : end_idx + 1]).strip()
                if not chunk:
                    raise ClinicalInputExtractionError("Clinical input extraction returned an empty line range.")

                chunks.append(chunk)
                normalized_ranges.append(
                    ClinicalSectionLineRange(start_line=line_range.start_line, end_line=line_range.end_line)
                )

            section_texts[section] = SECTION_FRAGMENT_JOINER.join(chunks)
            ranges_by_section[section] = normalized_ranges

        cls._assert_sections_exist_in_source(source_text=source_text, sections=section_texts)

        return ClinicalSectionExtractionResult(
            source_text=source_text,
            anamnesis=section_texts["anamnesis"],
            drugs=section_texts["drugs"],
            laboratory_analysis=section_texts["laboratory_analysis"],
            line_ranges=ranges_by_section,
            confidence=max(0.0, min(1.0, float(draft.confidence))),
        )

    def _deterministic_extract(
        self,
        clinical_input: str,
    ) -> ClinicalSectionExtractionResult | None:
        for parser in self._section_parsers:
            fragments = parser(clinical_input)
            if fragments is None:
                continue
            return self._build_result_from_fragments(
                source_text=clinical_input,
                fragments=fragments,
                confidence=parser.confidence,
            )
        return None

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

        await self.ensure_client()

        if self.client is None:
            raise ClinicalInputExtractionError(
                "LLM client is not initialized for input extraction"
            )

        try:
            extraction = await self.client.llm_structured_call(
                model=self.model,
                system_prompt=CLINICAL_SECTION_EXTRACTION_PROMPT,
                user_prompt=self._build_numbered_source(clinical_input),
                schema=LlmClinicalSectionExtractionDraft,
                temperature=0.0,
                use_json_mode=True,
                max_repair_attempts=1,
            )
        except Exception as exc:  # noqa: BLE001
            raise ClinicalInputExtractionError(
                f"Clinical input extraction failed: {exc}"
            ) from exc

        draft = LlmClinicalSectionExtractionDraft.model_validate(extraction)
        validated = self._build_result_from_line_ranges(
            source_text=clinical_input,
            draft=draft,
        )

        if progress_callback is not None:
            progress_callback(1.0)

        return validated
