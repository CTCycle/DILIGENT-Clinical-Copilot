from __future__ import annotations

import pytest
from services.session.clinical_section_parsers import (
    extract_required_dili_sections,
    find_dili_section_headings,
    missing_required_section_names,
)
from services.session.text_section_parser import (
    build_section_extraction_from_initial_text,
    parse_initial_text_sections,
)

###############################################################################
def test_extract_preferred_markdown_headings() -> None:
    text = "## Anamnesis\nA details\n\n## Therapy\nT details\n\n## Laboratory history\nL details"
    sections = extract_required_dili_sections(text)
    assert missing_required_section_names(sections) == []
    assert sections["anamnesis"].text == "A details"
    assert sections["therapy"].text == "T details"
    assert sections["laboratory_history"].text == "L details"

###############################################################################
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

###############################################################################
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

###############################################################################
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

###############################################################################
def test_phrase_aware_typo_matching_accepts_heading_typos() -> None:
    text = "## Clinical History\nA\n\n## Medicatons\nT\n\n## Laboratroy tests\nL"
    sections = extract_required_dili_sections(text)
    assert missing_required_section_names(sections) == []

###############################################################################
def test_accepts_common_variants() -> None:
    text = (
        "## Clinical history\nA\n\n## Current medications\nT\n\n## Laboratory tests\nL"
    )
    sections = extract_required_dili_sections(text)
    assert missing_required_section_names(sections) == []

###############################################################################
def test_mixed_language_therapy_heading_is_inferred_from_section_body() -> None:
    text = (
        "## Anamnesis\n"
        "Paziente con anamnesi oncologica complessa.\n\n"
        "## Terapia farmacologica\n"
        "Fortecortin 4 mg cpr 1-0-0-0\n"
        "De-Ursil 150 mg caps 1-0-1-0 per os\n\n"
        "## Laboratory Analysis\n"
        "ALT 730 U/L, AST 385 U/L, Bil tot 51.6 umol/L."
    )
    sections = extract_required_dili_sections(text)
    assert missing_required_section_names(sections) == []
    assert "Fortecortin" in sections["therapy"].text

###############################################################################
def test_rejects_missing_required_section() -> None:
    text = "## Anamnesis\nA\n\n## Therapy\nT"
    sections = extract_required_dili_sections(text)
    assert "laboratory_history" in missing_required_section_names(sections)

###############################################################################
def test_rejects_untitled_prose_inference() -> None:
    text = (
        "The patient has history and therapy and ALT/ALP values in one paragraph only."
    )
    sections = extract_required_dili_sections(text)
    assert missing_required_section_names(sections) == [
        "anamnesis",
        "therapy",
        "laboratory_history",
    ]

###############################################################################
def test_duplicate_competing_headings_raise() -> None:
    text = "## Therapy\nT1\n\n## Current medications\nT2\n\n## Anamnesis\nA\n\n## Laboratory history\nL"
    with pytest.raises(ValueError):
        extract_required_dili_sections(text)

###############################################################################
def test_blank_lines_do_not_create_markers() -> None:
    text = "line one\n\nline two\n\nline three"
    assert find_dili_section_headings(text) == []

###############################################################################
def test_section_parser_preserves_exact_body_slice_and_metadata() -> None:
    source = (
        "# Anamnesis\n"
        "Patient reports fatigue.\n\n"
        "# Therapy\n"
        "- Amoxicillin 500 mg bid\n\n"
        "# Laboratory history\n"
        "ALT 450 U/L\n"
    )
    parsed = parse_initial_text_sections(source)
    extraction = build_section_extraction_from_initial_text(parsed, source)
    therapy_meta = extraction.metadata["sections"]["drugs"]

    assert (
        extraction.drugs
        == source[therapy_meta["char_span"][0] : therapy_meta["char_span"][1]]
    )
    assert therapy_meta["canonical_key"] == "therapy"
    assert therapy_meta["body_line_span"][0] >= therapy_meta["heading_line_span"][1]
    assert therapy_meta["verbatim_coherent"] is True
    assert "source_hash" in extraction.metadata
    assert extraction.confidence == min(
        section["confidence_score"]
        for section in extraction.metadata["sections"].values()
    )
    assert extraction.metadata["requires_review"] is False

###############################################################################
def test_section_parser_marks_low_confidence_inferred_sections_for_review() -> None:
    source = (
        "# Anamnesis\n"
        "Patient reports fatigue.\n\n"
        "# Therapy\n"
        "- Amoxicillin 500 mg bid\n\n"
        "# Laboratory history\n"
        "ALT 450 U/L\n"
    )
    parsed = parse_initial_text_sections(source)
    parsed = parsed._replace(
        sections={
            **parsed.sections,
            "drugs": parsed.sections["drugs"]._replace(
                match_strategy="semantic_tokens",
                confidence_score=0.72,
                requires_review=True,
            ),
        }
    )
    extraction = build_section_extraction_from_initial_text(parsed, source)

    assert extraction.confidence < 0.85
    assert extraction.metadata["requires_review"] is True
    assert extraction.metadata["requires_review_sections"] == ["drugs"]
