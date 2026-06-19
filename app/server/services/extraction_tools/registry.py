from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from common.paths import TOOLS_PATH
from services.extraction_tools.regex_tools import TOOL_PATTERNS, run_regex_tool
from services.extraction_tools.schemas import (
    ExtractionToolDefinition,
    ExtractionToolError,
    RegexToolRequest,
    RegexToolResult,
)

MANIFEST_PATH = TOOLS_PATH / "extraction_tools.json"


###############################################################################
@lru_cache(maxsize=1)
def get_extraction_tool_definitions() -> list[ExtractionToolDefinition]:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    definitions = [
        ExtractionToolDefinition.model_validate(item) for item in data.get("tools", [])
    ]
    names = {definition.name for definition in definitions}
    missing_runtime = sorted(names.difference(TOOL_PATTERNS))
    if missing_runtime:
        raise RuntimeError(
            "Extraction tool manifest references tools without runtime handlers: "
            + ", ".join(missing_runtime)
        )
    return definitions


###############################################################################
def get_openai_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": definition.name,
            "description": definition.description,
            "parameters": definition.parameters.model_dump(),
        }
        for definition in get_extraction_tool_definitions()
    ]


###############################################################################
def run_extraction_tool(
    name: str, arguments: dict[str, Any]
) -> RegexToolResult | ExtractionToolError:
    definitions = {
        definition.name: definition for definition in get_extraction_tool_definitions()
    }
    if name not in definitions:
        return ExtractionToolError(
            tool_name=name,
            code="unknown_tool",
            message=f"Extraction tool '{name}' is not registered.",
        )
    try:
        request = RegexToolRequest.model_validate(arguments)
    except Exception as exc:  # noqa: BLE001
        return ExtractionToolError(
            tool_name=name,
            code="invalid_arguments",
            message=str(exc),
        )
    return run_regex_tool(name, request)
