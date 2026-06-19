from services.extraction_tools.registry import run_extraction_tool
from services.extraction_tools.schemas import RegexToolResult


###############################################################################
def test_regex_tools_return_structured_reproducible_matches() -> None:
    text = "Paracetamol 1000 mg oral bid from 2024-01-02\nALT 450 U/L"

    dosage = run_extraction_tool(
        "search_dosage_by_regex",
        {"text": text, "source_section": "therapy"},
    )
    lab = run_extraction_tool(
        "search_lab_value_by_regex",
        {"text": text, "source_section": "laboratory_history"},
    )

    assert isinstance(dosage, RegexToolResult)
    assert isinstance(lab, RegexToolResult)
    assert dosage.matches[0].match_text == "1000 mg"
    assert dosage.matches[0].start_char == text.index("1000 mg")
    assert lab.matches[0].line_number == 2
    assert lab.matches[0].pattern_id == "liver_lab_value"


###############################################################################
def test_regex_tools_cover_date_frequency_route_and_timeline() -> None:
    text = "Started amoxicillin 500 mg per os bid from 12/01/2024"
    for tool_name in (
        "search_date_by_regex",
        "search_frequency_by_regex",
        "search_route_by_regex",
        "search_timeline_by_regex",
    ):
        result = run_extraction_tool(
            tool_name,
            {"text": text, "source_section": "therapy"},
        )
        assert isinstance(result, RegexToolResult)
        assert result.matches
