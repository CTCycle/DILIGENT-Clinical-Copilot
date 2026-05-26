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


def test_markdown_sections_ignore_body_subheadings() -> None:
    text = (
        "## Anamnesis / Clinical History\n"
        "History details\n"
        "Terapia specialistica eseguita:\n"
        "Prior oncology treatment in the history narrative.\n\n"
        "## Therapy / Medication Exposure\n"
        "Treatment details\n"
        "Terapia farmacologica\n"
        "Current drug list.\n\n"
        "## Laboratory Analysis\n"
        "Laboratory details"
    )

    sections = extract_required_dili_sections(text)

    assert missing_required_section_names(sections) == []
    assert "Terapia specialistica eseguita:" in sections["anamnesis"].text
    assert "Terapia farmacologica" in sections["therapy"].text


def test_final_report_heading_is_not_anamnesis_typo() -> None:
    text = (
        "## Anamnesis / Clinical History\nA\n\n"
        "## Therapy / Medication Exposure\nT\n\n"
        "## Laboratory Analysis\nL\n\n"
        "## Final Physician Report / Medical Conclusion\nConclusion"
    )

    sections = extract_required_dili_sections(text)

    assert missing_required_section_names(sections) == []
    assert sections["laboratory_history"].text == "L"


def test_unclassified_markdown_headings_bound_sections_generically() -> None:
    text = (
        "# Source Document\nmetadata\n\n"
        "## Patient History\nA\n\n"
        "## Current Medications\nT\n\n"
        "## Blood Tests\nL\n\n"
        "## References\n1. Citation"
    )

    sections = extract_required_dili_sections(text)

    assert missing_required_section_names(sections) == []
    assert sections["laboratory_history"].text == "L"


def test_phrase_aware_typo_matching_accepts_heading_typos() -> None:
    text = "## Clinical History\nA\n\n## Medicatons\nT\n\n## Laboratroy tests\nL"
    sections = extract_required_dili_sections(text)
    assert missing_required_section_names(sections) == []


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
