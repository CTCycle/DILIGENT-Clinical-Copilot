from __future__ import annotations

import asyncio

import pytest

from domain.clinical.entities import (
    ClinicalSectionExtractionResult,
    ClinicalSectionFragment,
)
from domain.clinical.sections import SECTION_FRAGMENT_JOINER
from services.session.clinical_input_extractor import (
    ClinicalInputExtractionError,
    ClinicalInputExtractor,
)


def test_markdown_repeated_non_contiguous_sections_are_preserved() -> None:
    input_text = (
        "# Anamnesis\n"
        "first history block\n\n"
        "# Current therapy\n"
        "drug A 10 mg daily\n\n"
        "# Anamnesis\n"
        "second history block\n\n"
        "# Laboratory analysis\n"
        "ALT 300 U/L"
    )

    extractor = ClinicalInputExtractor()
    result = asyncio.run(extractor.extract(clinical_input=input_text))

    assert (
        result.anamnesis
        == "first history block\n\n" + SECTION_FRAGMENT_JOINER + "second history block\n\n"
    )
    assert result.drugs == "drug A 10 mg daily\n\n"
    assert result.laboratory_analysis == "ALT 300 U/L"
    assert len([fragment for fragment in result.fragments if fragment.section == "anamnesis"]) == 2
    for fragment in result.fragments:
        assert input_text[fragment.start : fragment.end] == fragment.text


def test_xml_repeated_tags_are_preserved() -> None:
    input_text = (
        "<anamnesis>first history block</anamnesis>\n"
        "<current_therapy>drug A 10 mg daily</current_therapy>\n"
        "<anamnesis>second history block</anamnesis>\n"
        "<laboratory_analysis>ALT 300 U/L</laboratory_analysis>"
    )

    extractor = ClinicalInputExtractor()
    result = asyncio.run(extractor.extract(clinical_input=input_text))

    assert result.anamnesis == "first history block" + SECTION_FRAGMENT_JOINER + "second history block"
    assert result.drugs == "drug A 10 mg daily"
    assert result.laboratory_analysis == "ALT 300 U/L"
    for fragment in result.fragments:
        assert input_text[fragment.start : fragment.end] == fragment.text


def test_json_array_sections_are_preserved_when_values_are_literal() -> None:
    input_text = """{
  "anamnesis": ["first history block", "second history block"],
  "current_therapy": ["drug A 10 mg daily"],
  "laboratory_analysis": ["ALT 300 U/L"]
}"""

    extractor = ClinicalInputExtractor()
    result = asyncio.run(extractor.extract(clinical_input=input_text))

    assert result.anamnesis == "first history block" + SECTION_FRAGMENT_JOINER + "second history block"
    assert result.drugs == "drug A 10 mg daily"
    assert result.laboratory_analysis == "ALT 300 U/L"
    for fragment in result.fragments:
        assert input_text[fragment.start : fragment.end] == fragment.text


def test_json_escaped_values_do_not_pass_deterministic_parser() -> None:
    input_text = '{"anamnesis": "first\\nsecond", "current_therapy": "drug A", "laboratory_analysis": "ALT 300"}'
    assert ClinicalInputExtractor._deterministic_extract(input_text) is None


def test_indexed_repeated_section_requires_label() -> None:
    rejected = (
        "1.\n"
        "first history block\n\n"
        "2.\n"
        "drug A\n\n"
        "1.\n"
        "second history block\n\n"
        "3.\n"
        "ALT 300"
    )
    assert ClinicalInputExtractor._deterministic_extract(rejected) is None

    accepted = (
        "1. Anamnesis\n"
        "first history block\n\n"
        "2. Current therapy\n"
        "drug A\n\n"
        "1. Anamnesis\n"
        "second history block\n\n"
        "3. Laboratory analysis\n"
        "ALT 300"
    )
    extraction = ClinicalInputExtractor._deterministic_extract(accepted)
    assert extraction is not None
    assert len([fragment for fragment in extraction.fragments if fragment.section == "anamnesis"]) == 2


def test_keyword_only_unstructured_text_does_not_pass_deterministic_parser() -> None:
    input_text = "Patient with jaundice and nausea. Drug A 10 mg daily. ALT 300 AST 250 bilirubin 4."
    assert ClinicalInputExtractor._deterministic_extract(input_text) is None


def test_llm_fragment_must_match_source_slice() -> None:
    input_text = "Unstructured text that forces fallback."

    class FakeClient:
        async def llm_structured_call(self, **_: object) -> ClinicalSectionExtractionResult:
            return ClinicalSectionExtractionResult(
                source_text=input_text,
                anamnesis="Unstructured",
                drugs="text",
                laboratory_analysis="fallback",
                fragments=[
                    ClinicalSectionFragment(section="anamnesis", start=0, end=12, text="MismatchText"),
                    ClinicalSectionFragment(section="drugs", start=13, end=17, text="text"),
                    ClinicalSectionFragment(section="laboratory_analysis", start=18, end=26, text="that for"),
                ],
                confidence=0.3,
            )

    extractor = ClinicalInputExtractor(client=FakeClient())
    with pytest.raises(ClinicalInputExtractionError):
        asyncio.run(extractor.extract(clinical_input=input_text))


def test_llm_requires_all_three_sections() -> None:
    input_text = "Unstructured text that forces fallback."

    class FakeClient:
        async def llm_structured_call(self, **_: object) -> ClinicalSectionExtractionResult:
            return ClinicalSectionExtractionResult(
                source_text=input_text,
                anamnesis="anamnesis",
                drugs="drugs",
                laboratory_analysis=None,
                fragments=[
                    ClinicalSectionFragment(section="anamnesis", start=0, end=10, text="Unstructur"),
                    ClinicalSectionFragment(section="drugs", start=11, end=15, text="text"),
                ],
                confidence=0.4,
            )

    extractor = ClinicalInputExtractor(client=FakeClient())
    with pytest.raises(ClinicalInputExtractionError):
        asyncio.run(extractor.extract(clinical_input=input_text))
