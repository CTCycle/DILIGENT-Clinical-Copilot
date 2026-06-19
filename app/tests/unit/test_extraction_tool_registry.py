from services.extraction_tools.registry import (
    get_extraction_tool_definitions,
    get_openai_tool_schemas,
    run_extraction_tool,
)
from services.extraction_tools.schemas import ExtractionToolError, RegexToolResult


###############################################################################
def test_every_registered_tool_has_openai_compatible_schema() -> None:
    definitions = get_extraction_tool_definitions()
    schemas = get_openai_tool_schemas()

    assert {item.name for item in definitions} == {
        "search_drug_by_regex",
        "search_dosage_by_regex",
        "search_timeline_by_regex",
        "search_lab_value_by_regex",
        "search_date_by_regex",
        "search_frequency_by_regex",
        "search_route_by_regex",
    }
    for schema in schemas:
        assert schema["type"] == "function"
        assert schema["name"]
        assert schema["description"]
        assert schema["parameters"]["type"] == "object"
        assert "text" in schema["parameters"]["required"]


###############################################################################
def test_registry_returns_structured_errors_and_results() -> None:
    unknown = run_extraction_tool("missing", {})
    valid = run_extraction_tool(
        "search_date_by_regex",
        {"text": "15/01/2024", "source_section": "therapy"},
    )

    assert isinstance(unknown, ExtractionToolError)
    assert unknown.code == "unknown_tool"
    assert isinstance(valid, RegexToolResult)
    assert valid.matches[0].match_text == "15/01/2024"
