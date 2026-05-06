from __future__ import annotations

from services.session.clinical_section_parsers import (
    extract_sections_from_markers,
    find_section_markers,
    validate_sections_against_source,
)


def _extract(text: str):
    return extract_sections_from_markers(text, find_section_markers(text))


def test_markdown_headings_parse() -> None:
    sections = _extract("# Anamnesis\nA\n## Therapy\nT\n### Laboratory Analysis\nL")
    assert sections is not None
    assert sections["anamnesis"] == "A"


def test_numbered_headings_parse() -> None:
    sections = _extract("1. Anamnesis\nA\n2) Therapy\nT\n3: Lab analysis\nL")
    assert sections is not None


def test_roman_headings_parse() -> None:
    sections = _extract("I. Anamnesis\nA\nII. Drugs\nT\nIII. Labs\nL")
    assert sections is not None


def test_plain_labeled_parse() -> None:
    sections = _extract("Anamnesis:\nA\nCurrent Drugs:\nT\nLaboratory Analysis:\nL")
    assert sections is not None


def test_missing_sections_fail() -> None:
    sections = _extract("Anamnesis:\nA\nDrugs:\nT")
    assert sections is None


def test_source_validation_rejects_fabrication() -> None:
    source = "Anamnesis: A\nTherapy: T\nLab analysis: L"
    assert validate_sections_against_source(
        source,
        {"anamnesis": "A", "drugs": "invented", "laboratory_analysis": "L"},
    ) is False
