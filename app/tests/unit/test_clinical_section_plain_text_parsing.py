from __future__ import annotations

import asyncio

import pytest

from domain.clinical.entities import LlmClinicalSectionExtractionDraft
from services.llm.prompts import CLINICAL_SECTION_EXTRACTION_PROMPT
from services.session.clinical_input_extractor import (
    ClinicalInputExtractionError,
    ClinicalInputExtractor,
)
from services.session.clinical_section_parsers import PlainTextSectionParser


def _aliases() -> dict[str, frozenset[str]]:
    return {
        "anamnesis": frozenset({"anamnesis", "anamnesi"}),
        "drugs": frozenset({"drugs", "farmaci"}),
        "laboratory_analysis": frozenset({"lab analysis", "analisi di laboratorio"}),
    }


def test_db_backed_aliases_are_required() -> None:
    parser = PlainTextSectionParser(section_title_aliases={})
    text = "# Anamnesis\nA\n# Drugs\nD\n# Lab analysis\nL"
    assert parser(text) is None


def test_markdown_headings_parse() -> None:
    parser = PlainTextSectionParser(section_title_aliases=_aliases())
    text = "# Anamnesis\nA\n# Drugs\nD\n# Lab analysis\nL"
    fragments = parser(text)
    assert fragments is not None


def test_numbered_headings_parse_only_with_db_titles() -> None:
    parser = PlainTextSectionParser(section_title_aliases=_aliases())
    text = "1. Anamnesis\nA\n2) Drugs\nD\n3 - Lab analysis\nL"
    fragments = parser(text)
    assert fragments is not None


def test_multilingual_aliases_parse() -> None:
    parser = PlainTextSectionParser(section_title_aliases=_aliases())
    text = "Anamnesi:\nA\nFarmaci:\nD\nAnalisi di laboratorio:\nL"
    fragments = parser(text)
    assert fragments is not None


def test_non_pooled_alias_is_rejected() -> None:
    parser = PlainTextSectionParser(
        section_title_aliases={
            "anamnesis": frozenset({"anamnesis"}),
            "drugs": frozenset({"drugs"}),
            "laboratory_analysis": frozenset({"lab analysis"}),
        }
    )
    text = "Anamnesis:\nA\nTherapy:\nD\nLab analysis:\nL"
    assert parser(text) is None


def test_json_like_text_is_not_deterministically_parsed() -> None:
    parser = PlainTextSectionParser(section_title_aliases=_aliases())
    text = '{"anamnesis":"A","drugs":"D","laboratory_analysis":"L"}'
    assert parser(text) is None


def test_xml_like_text_is_not_deterministically_parsed() -> None:
    parser = PlainTextSectionParser(section_title_aliases=_aliases())
    text = "<anamnesis>A</anamnesis><drugs>D</drugs><laboratory_analysis>L</laboratory_analysis>"
    assert parser(text) is None


def test_hallucination_validation_rejects_absent_content() -> None:
    with pytest.raises(ClinicalInputExtractionError):
        ClinicalInputExtractor._assert_sections_exist_in_source(
            source_text="Anamnesis: A\nDrugs: D\nLab analysis: L",
            sections={
                "anamnesis": "A",
                "drugs": "not present",
                "laboratory_analysis": "L",
            },
        )


def test_llm_fallback_uses_system_and_user_prompts() -> None:
    input_text = "Some intro\nAnamnesis details\nDrug details\nLab details"
    captured: dict[str, object] = {}

    class FakeClient:
        async def llm_structured_call(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return {
                "anamnesis": [{"start_line": 1, "end_line": 2}],
                "drugs": [{"start_line": 3, "end_line": 3}],
                "laboratory_analysis": [{"start_line": 4, "end_line": 4}],
                "confidence": 0.42,
            }

    extractor = ClinicalInputExtractor(client=FakeClient())
    result = asyncio.run(extractor.extract(clinical_input=input_text))

    assert captured["system_prompt"] == CLINICAL_SECTION_EXTRACTION_PROMPT
    assert "1:" in str(captured["user_prompt"])
    assert captured["schema"] is LlmClinicalSectionExtractionDraft
    assert result.anamnesis == "Some intro\nAnamnesis details"
    assert result.drugs == "Drug details"
    assert result.laboratory_analysis == "Lab details"
