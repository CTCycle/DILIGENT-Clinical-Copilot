from __future__ import annotations

import pytest

from services.session.clinical_section_parsers import extract_required_dili_sections, missing_required_section_names


def test_extract_preferred_markdown_headings() -> None:
    text = "## Anamnesis\nA details\n\n## Therapy\nT details\n\n## Laboratory history\nL details"
    sections = extract_required_dili_sections(text)
    assert missing_required_section_names(sections) == []
    assert sections["anamnesis"].text == "A details"
    assert sections["therapy"].text == "T details"
    assert sections["laboratory_history"].text == "L details"


def test_accepts_common_variants() -> None:
    text = "## Clinical history\nA\n\n## Current medications\nT\n\n## Laboratory tests\nL"
    sections = extract_required_dili_sections(text)
    assert missing_required_section_names(sections) == []


def test_rejects_missing_required_section() -> None:
    text = "## Anamnesis\nA\n\n## Therapy\nT"
    sections = extract_required_dili_sections(text)
    assert "laboratory_history" in missing_required_section_names(sections)


def test_rejects_untitled_prose_inference() -> None:
    text = "The patient has history and therapy and ALT/ALP values in one paragraph only."
    sections = extract_required_dili_sections(text)
    assert missing_required_section_names(sections) == ["anamnesis", "therapy", "laboratory_history"]


def test_duplicate_competing_headings_raise() -> None:
    text = "## Therapy\nT1\n\n## Current medications\nT2\n\n## Anamnesis\nA\n\n## Laboratory history\nL"
    with pytest.raises(ValueError):
        extract_required_dili_sections(text)
