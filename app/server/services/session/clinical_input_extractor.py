from __future__ import annotations

import asyncio
from difflib import SequenceMatcher
from collections.abc import Callable
from typing import Any

from configurations.llm_configs import LLMRuntimeConfig
from configurations.startup import server_settings
from domain.clinical.entities import (
    ClinicalSectionExtractionResult,
    ClinicalSectionFragment,
)
from domain.clinical.sections import SECTION_FRAGMENT_JOINER, SECTION_KEYS
from services.llm.client_runtime import ensure_runtime_client
from services.llm.prompts import CLINICAL_SECTION_EXTRACTION_PROMPT
from services.llm.providers import select_llm_provider
from services.session.clinical_section_parsers import (
    DETERMINISTIC_SECTION_PARSERS,
    SectionFragmentSlice,
)


###############################################################################
class ClinicalInputExtractionError(RuntimeError):
    pass


###############################################################################
class ClinicalInputExtractor:
    MIN_SIMILARITY_RATIO = 0.96
    MAX_ABSOLUTE_CHAR_DRIFT = 12
    MAX_WORD_DELTA = 2
    MAX_ABSOLUTE_EDIT_DRIFT = 2

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
                )
            ),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _build_prompt(clinical_input: str) -> str:
        return (
            "Extract clinical section fragments from the source text.\n"
            "Sections may be non-contiguous and may appear multiple times.\n"
            "Use Python slicing offsets: start is inclusive, end is exclusive.\n"
            "For every fragment, source_text[start:end] must exactly equal fragment.text.\n"
            "Build anamnesis, drugs, and laboratory_analysis by joining fragments from the same section with two newline characters.\n"
            "Never paraphrase, summarize, translate, normalize whitespace, redact, repair typos, or reorder source text.\n"
            "Use this exact source text:\n\n"
            f"{clinical_input}"
        )

    # -------------------------------------------------------------------------
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

        section_fragments: dict[str, list[SectionFragmentSlice]] = {
            section: [] for section in SECTION_KEYS
        }

        occupied_spans: set[tuple[int, int]] = set()
        for fragment in fragments:
            if fragment.start < 0 or fragment.end > len(source_text) or fragment.start >= fragment.end:
                raise ClinicalInputExtractionError("Clinical input extraction produced invalid fragment offsets.")
            if (fragment.start, fragment.end) in occupied_spans:
                raise ClinicalInputExtractionError("Clinical input extraction produced duplicate fragments.")
            occupied_spans.add((fragment.start, fragment.end))

            fragment_text = source_text[fragment.start : fragment.end]
            if not fragment_text.strip():
                raise ClinicalInputExtractionError("Clinical input extraction produced an empty section fragment.")
            section_fragments[fragment.section].append(fragment)

        missing_sections = [
            section for section, values in section_fragments.items() if not values
        ]
        if missing_sections:
            raise ClinicalInputExtractionError(
                "Clinical input must contain anamnesis, current therapy, and laboratory analysis sections."
            )

        extracted_fragments: list[ClinicalSectionFragment] = []
        section_texts: dict[str, str] = {}

        for section in SECTION_KEYS:
            ordered = sorted(section_fragments[section], key=lambda item: (item.start, item.end))
            texts = []
            for fragment in ordered:
                text = source_text[fragment.start : fragment.end]
                texts.append(text)
                extracted_fragments.append(
                    ClinicalSectionFragment(
                        section=section,
                        start=fragment.start,
                        end=fragment.end,
                        text=text,
                    )
                )
            section_texts[section] = SECTION_FRAGMENT_JOINER.join(texts)

        return ClinicalSectionExtractionResult(
            source_text=source_text,
            anamnesis=section_texts["anamnesis"],
            drugs=section_texts["drugs"],
            laboratory_analysis=section_texts["laboratory_analysis"],
            fragments=sorted(extracted_fragments, key=lambda item: (item.start, item.end, item.section)),
            confidence=max(0.0, min(1.0, float(confidence))),
        )

    # -------------------------------------------------------------------------
    @classmethod
    def _validate_complete_preserved_result(
        cls,
        *,
        source_text: str,
        extraction: ClinicalSectionExtractionResult,
    ) -> ClinicalSectionExtractionResult:
        if not cls._is_near_match(extraction.source_text, source_text):
            raise ClinicalInputExtractionError(
                "Clinical input extraction returned mismatched source_text."
            )

        fragments = [
            SectionFragmentSlice(
                section=fragment.section,
                start=fragment.start,
                end=fragment.end,
            )
            for fragment in extraction.fragments
        ]

        rebuilt = cls._build_result_from_fragments(
            source_text=source_text,
            fragments=fragments,
            confidence=extraction.confidence,
        )

        for original_fragment, rebuilt_fragment in zip(extraction.fragments, rebuilt.fragments, strict=True):
            if not cls._is_near_match(
                source_text[original_fragment.start : original_fragment.end],
                original_fragment.text,
            ):
                raise ClinicalInputExtractionError(
                    "Clinical input extraction changed a source fragment."
                )
            if not cls._is_near_match(original_fragment.text, rebuilt_fragment.text):
                raise ClinicalInputExtractionError(
                    "Clinical input extraction returned inconsistent fragment text."
                )

        if not cls._is_near_match(extraction.anamnesis, rebuilt.anamnesis):
            raise ClinicalInputExtractionError("Clinical input extraction returned inconsistent anamnesis.")
        if not cls._is_near_match(extraction.drugs, rebuilt.drugs):
            raise ClinicalInputExtractionError("Clinical input extraction returned inconsistent current therapy.")
        if not cls._is_near_match(extraction.laboratory_analysis, rebuilt.laboratory_analysis):
            raise ClinicalInputExtractionError("Clinical input extraction returned inconsistent laboratory analysis.")

        return rebuilt

    # -------------------------------------------------------------------------
    @classmethod
    def _is_near_match(cls, expected: str | None, observed: str | None) -> bool:
        if expected is None or observed is None:
            return expected is observed
        if expected == observed:
            return True

        normalized_expected = expected.replace("\r\n", "\n").replace("\r", "\n")
        normalized_observed = observed.replace("\r\n", "\n").replace("\r", "\n")
        if normalized_expected == normalized_observed:
            return True

        stripped_expected = normalized_expected.rstrip()
        stripped_observed = normalized_observed.rstrip()
        if stripped_expected == stripped_observed:
            return True

        length_gap = abs(len(stripped_expected) - len(stripped_observed))
        if length_gap > cls.MAX_ABSOLUTE_CHAR_DRIFT:
            return False

        expected_words = stripped_expected.split()
        observed_words = stripped_observed.split()
        if abs(len(expected_words) - len(observed_words)) > cls.MAX_WORD_DELTA:
            return False

        estimated_edit_drift = cls._estimate_edit_drift(
            stripped_expected,
            stripped_observed,
        )
        if estimated_edit_drift <= cls.MAX_ABSOLUTE_EDIT_DRIFT:
            return True

        similarity = SequenceMatcher(a=stripped_expected, b=stripped_observed).ratio()
        return similarity >= cls.MIN_SIMILARITY_RATIO

    # -------------------------------------------------------------------------
    @staticmethod
    def _estimate_edit_drift(expected: str, observed: str) -> int:
        matcher = SequenceMatcher(a=expected, b=observed)
        drift = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            drift += max(i2 - i1, j2 - j1)
        return drift

    # -------------------------------------------------------------------------
    @classmethod
    def _deterministic_extract(
        cls,
        clinical_input: str,
    ) -> ClinicalSectionExtractionResult | None:
        for parser in DETERMINISTIC_SECTION_PARSERS:
            fragments = parser(clinical_input)
            if fragments is None:
                continue
            try:
                return cls._build_result_from_fragments(
                    source_text=clinical_input,
                    fragments=fragments,
                    confidence=parser.confidence,
                )
            except ClinicalInputExtractionError:
                continue
        return None

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
                user_prompt=self._build_prompt(clinical_input),
                schema=ClinicalSectionExtractionResult,
                temperature=0.0,
                use_json_mode=True,
                max_repair_attempts=1,
            )
        except Exception as exc:  # noqa: BLE001
            raise ClinicalInputExtractionError(
                f"Clinical input extraction failed: {exc}"
            ) from exc

        validated = self._validate_complete_preserved_result(
            source_text=clinical_input,
            extraction=extraction,
        )

        if progress_callback is not None:
            progress_callback(1.0)

        return validated
