from __future__ import annotations

import asyncio

import pytest

from domain.clinical.sections import SECTION_FRAGMENT_JOINER
from services.session.clinical_input_extractor import (
    ClinicalInputExtractionError,
    ClinicalInputExtractor,
)


def test_markdown_repeated_non_contiguous_sections_are_preserved() -> None:
    input_text = (
        "# Anamnesis\n"
        "first history block\n\n"
        "# Drugs\n"
        "drug A 10 mg daily\n\n"
        "# Anamnesis\n"
        "second history block\n\n"
        "# Lab analysis\n"
        "ALT 300 U/L"
    )

    extractor = ClinicalInputExtractor()
    result = asyncio.run(extractor.extract(clinical_input=input_text))

    assert result.anamnesis == "first history block" + SECTION_FRAGMENT_JOINER + "second history block"
    assert result.drugs == "drug A 10 mg daily"
    assert result.laboratory_analysis == "ALT 300 U/L"
    assert result.line_ranges == {}


def test_keyword_only_unstructured_text_does_not_pass_deterministic_parser() -> None:
    input_text = "Patient with jaundice and nausea. Drug A 10 mg daily. ALT 300 AST 250 bilirubin 4."
    assert ClinicalInputExtractor()._deterministic_extract(input_text) is None


def test_llm_requires_all_three_sections() -> None:
    input_text = "Unstructured text that forces fallback."

    class FakeClient:
        async def llm_structured_call(self, **_: object) -> object:
            return {
                "anamnesis": [{"start_line": 1, "end_line": 1}],
                "drugs": [{"start_line": 1, "end_line": 1}],
                "laboratory_analysis": [],
                "confidence": 0.4,
            }

    extractor = ClinicalInputExtractor(client=FakeClient())
    with pytest.raises(ClinicalInputExtractionError):
        asyncio.run(extractor.extract(clinical_input=input_text))


def test_llm_out_of_range_lines_rejected() -> None:
    input_text = "line one\nline two"

    class FakeClient:
        async def llm_structured_call(self, **_: object) -> object:
            return {
                "anamnesis": [{"start_line": 1, "end_line": 1}],
                "drugs": [{"start_line": 2, "end_line": 2}],
                "laboratory_analysis": [{"start_line": 99, "end_line": 99}],
                "confidence": 0.4,
            }

    extractor = ClinicalInputExtractor(client=FakeClient())
    with pytest.raises(ClinicalInputExtractionError):
        asyncio.run(extractor.extract(clinical_input=input_text))
