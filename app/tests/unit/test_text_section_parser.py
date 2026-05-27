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
