from __future__ import annotations

import pytest
from services.session.clinical_section_parsers import (
    extract_required_dili_sections,
    missing_required_section_names,
)

###############################################################################
def _extract(text: str):
    try:
        sections = extract_required_dili_sections(text)
    except ValueError:
        return None
    if missing_required_section_names(sections):
        return None
    return {
        "anamnesis": sections["anamnesis"].text,
        "drugs": sections["therapy"].text,
        "laboratory_analysis": sections["laboratory_history"].text,
    }

###############################################################################
def test_markdown_headings_parse() -> None:
    sections = _extract("## Anamnesis\nA\n## Therapy\nT\n## Laboratory history\nL")
    assert sections is not None
    assert sections["anamnesis"] == "A"

###############################################################################
def test_variant_headings_parse() -> None:
    sections = _extract(
        "Clinical history:\nA\nCurrent medications:\nT\nLaboratory tests:\nL"
    )
    assert sections is not None

###############################################################################
def test_missing_sections_fail() -> None:
    assert _extract("Anamnesis:\nA\nTherapy:\nT") is None

###############################################################################
def test_untitled_prose_fails() -> None:
    assert _extract("Patient history with ALT and therapy in one paragraph.") is None

###############################################################################
def test_duplicate_competing_headings_fail() -> None:
    with pytest.raises(ValueError):
        extract_required_dili_sections(
            "Therapy:\nT1\nCurrent medications:\nT2\nAnamnesis:\nA\nLaboratory history:\nL"
        )

###############################################################################
def test_duplicate_heading_error_reports_both_line_numbers() -> None:
    text = (
        "## Anamnesis\nhistory\n"
        "## Therapy\nfirst drug\n"
        "## Therapy\nsecond drug\n"
        "## Laboratory Analysis\nALT 200 U/L\n"
    )
    with pytest.raises(
        ValueError,
        match=r"Duplicate heading '## Therapy' found at lines 3 and 5",
    ):
        extract_required_dili_sections(text)

###############################################################################
def test_source_validation_rejects_fabrication() -> None:
    sections = _extract(
        "## Anamnesis\nReal history\n## Therapy\nReal drugs 400mg\n## Laboratory history\nALT 45"
    )
    assert sections is not None
    assert sections["anamnesis"] == "Real history"
    assert sections["drugs"] == "Real drugs 400mg"
    assert sections["laboratory_analysis"] == "ALT 45"
    assert "invented" not in sections["drugs"]
