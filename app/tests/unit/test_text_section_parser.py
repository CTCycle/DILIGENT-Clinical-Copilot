from __future__ import annotations

from services.session.text_section_parser import parse_initial_text_sections


def test_parses_required_sections_with_line_ranges() -> None:
    text = (
        "ANAMNESIS\n"
        "history text\n"
        "DRUGS:\n"
        "drug row\n"
        "LABORATORY ANALYSIS\n"
        "lab row\n"
    )
    result = parse_initial_text_sections(text)
    assert result.missing_required_sections == []
    assert result.malformed_sections == []
    assert result.sections["anamnesis"].start_line == 1
    assert result.sections["drugs"].start_line == 3
    assert result.sections["laboratory_analysis"].start_line == 5


def test_rejects_missing_anamnesis() -> None:
    text = "DRUGS\nx\nLABORATORY ANALYSIS\ny\n"
    result = parse_initial_text_sections(text)
    assert "anamnesis" in result.missing_required_sections


def test_rejects_empty_required_section() -> None:
    text = "ANAMNESIS\na\nDRUGS\n\nLABORATORY ANALYSIS\nx\n"
    result = parse_initial_text_sections(text)
    assert "empty:drugs" in result.malformed_sections


def test_rejects_duplicate_section_heading() -> None:
    text = "ANAMNESIS\na\nANAMNESIS\nb\nDRUGS\nd\nLABORATORY ANALYSIS\nl\n"
    result = parse_initial_text_sections(text)
    assert "duplicate:anamnesis" in result.malformed_sections


def test_parses_mixed_language_therapy_heading_from_live_preflight_path() -> None:
    text = (
        "## Anamnesis\n"
        "History text.\n\n"
        "## Terapia farmacologica\n"
        "Fortecortin 4 mg cpr 1-0-0-0\n"
        "De-Ursil 150 mg caps 1-0-1-0 per os\n\n"
        "## Laboratory Analysis\n"
        "ALT 730 U/L, AST 385 U/L, Bil tot 51.6 umol/L.\n"
    )
    result = parse_initial_text_sections(text)
    assert result.missing_required_sections == []
    assert result.malformed_sections == []
    assert "Fortecortin" in result.sections["drugs"].text
