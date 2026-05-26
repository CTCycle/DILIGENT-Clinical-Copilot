from __future__ import annotations

import pytest

from services.session.clinical_section_parsers import (
    extract_sections_from_markers,
    find_section_markers,
    validate_sections_against_source,
)


def _extract(text: str):
    return extract_sections_from_markers(text, find_section_markers(text))


def test_markdown_headings_parse() -> None:
    sections = _extract("## Anamnesis\nA\n## Therapy\nT\n## Laboratory history\nL")
    assert sections is not None
    assert sections["anamnesis"] == "A"


def test_variant_headings_parse() -> None:
    sections = _extract("Clinical history:\nA\nCurrent medications:\nT\nLaboratory tests:\nL")
    assert sections is not None


def test_missing_sections_fail() -> None:
    assert _extract("Anamnesis:\nA\nTherapy:\nT") is None


def test_untitled_prose_fails() -> None:
    assert _extract("Patient history with ALT and therapy in one paragraph.") is None


def test_duplicate_competing_headings_fail() -> None:
    with pytest.raises(ValueError):
        from services.session.clinical_section_parsers import extract_required_dili_sections

        extract_required_dili_sections(
            "Therapy:\nT1\nCurrent medications:\nT2\nAnamnesis:\nA\nLaboratory history:\nL"
        )


def test_source_validation_rejects_fabrication() -> None:
    source = "Anamnesis: A\nTherapy: T\nLaboratory history: L"
    assert (
        validate_sections_against_source(
            source,
            {"anamnesis": "A", "drugs": "invented", "laboratory_analysis": "L"},
        )
        is False
    )
