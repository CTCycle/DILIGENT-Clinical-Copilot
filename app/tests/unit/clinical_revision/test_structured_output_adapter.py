from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from services.llm.structured import StructuredOutputAdapter


###############################################################################
class StrictPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    value: int
    status: str


###############################################################################
def build_adapter() -> StructuredOutputAdapter:
    return StructuredOutputAdapter(
        provider="test-provider",
        model_name="test-model",
        supports_native_json_schema=False,
        supports_strict_schema=False,
        supports_tool_schema=False,
        supports_json_mode=True,
    )


###############################################################################
def test_structured_output_adapter_validates_strict_json_object() -> None:
    adapter = build_adapter()

    parsed = adapter.validate_or_fail('{"value": 7, "status": "ok"}', StrictPayload)

    assert parsed.value == 7
    assert parsed.status == "ok"


###############################################################################
def test_structured_output_adapter_rejects_wrong_schema() -> None:
    adapter = build_adapter()

    with pytest.raises(ValidationError):
        adapter.validate_or_fail('{"value": 7, "status": "ok", "extra": true}', StrictPayload)


###############################################################################
def test_structured_output_adapter_rejects_leading_or_trailing_prose() -> None:
    adapter = build_adapter()

    with pytest.raises(ValueError):
        adapter.validate_or_fail('Here is the JSON: {"value": 7, "status": "ok"}', StrictPayload)

    with pytest.raises(ValueError):
        adapter.validate_or_fail('{"value": 7, "status": "ok"} trailing text', StrictPayload)


###############################################################################
def test_structured_output_adapter_repairs_once_when_allowed() -> None:
    adapter = build_adapter()
    repair_calls: list[dict[str, object]] = []

    parsed = adapter.complete_with_schema(
        schema=StrictPayload,
        completion_callable=lambda _context: '{"value": "7", "status": "broken"}',
        repair_callable=lambda context: (
            repair_calls.append(context) or '{"value": 7, "status": "fixed"}'
        ),
        allow_repair=True,
    )

    assert parsed.value == 7
    assert parsed.status == "fixed"
    assert len(repair_calls) == 1
    assert repair_calls[0]["schema_name"] == "StrictPayload"


###############################################################################
def test_structured_output_adapter_does_not_repair_when_disabled() -> None:
    adapter = build_adapter()

    with pytest.raises(ValidationError):
        adapter.complete_with_schema(
            schema=StrictPayload,
            completion_callable=lambda _context: '{"value": "7", "status": "broken"}',
            repair_callable=lambda _context: '{"value": 7, "status": "fixed"}',
            allow_repair=False,
        )
