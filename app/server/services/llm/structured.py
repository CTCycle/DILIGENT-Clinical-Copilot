from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


###############################################################################
def extract_first_json_dict(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        start = match.start()
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


###############################################################################
def parse_json_dict(obj_or_text: dict[str, Any] | str) -> dict[str, Any] | None:
    if isinstance(obj_or_text, dict):
        return obj_or_text
    if not isinstance(obj_or_text, str) or not obj_or_text.strip():
        return None
    try:
        loaded = json.loads(obj_or_text)
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        return extract_first_json_dict(obj_or_text)


###############################################################################
class StructuredOutputParser(Generic[T]):
    def __init__(self, *, schema: type[T]) -> None:
        self.schema = schema

    def get_format_instructions(self) -> str:
        schema_json = json.dumps(
            self.schema.model_json_schema(),
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return (
            "Return ONLY a valid JSON object that conforms to this JSON schema.\n"
            "Do not include markdown, comments, or additional keys.\n"
            f"JSON schema:\n{schema_json}"
        )

    def parse(self, text: str) -> T:
        payload = parse_json_dict(text)
        if payload is None:
            raise ValueError("No JSON object found in model output")
        return self.schema.model_validate(payload)


def parse_json_object_strict(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty_input")
    if text.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped, count=1)
        text = stripped.strip()
    decoder = json.JSONDecoder()
    try:
        parsed, end_index = decoder.raw_decode(text)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid_json_object") from exc
    if not isinstance(parsed, dict):
        raise ValueError("top_level_object_required")
    prefix = text[: text.find("{")].strip()
    if prefix:
        raise ValueError("leading_prose_not_allowed")
    trailing = text[end_index:].strip()
    if trailing:
        raise ValueError("trailing_prose_not_allowed")
    return parsed


###############################################################################
class StructuredOutputAdapter:
    def __init__(
        self,
        *,
        provider: str,
        model_name: str,
        supports_native_json_schema: bool,
        supports_strict_schema: bool,
        supports_tool_schema: bool,
        supports_json_mode: bool,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.supports_native_json_schema = bool(supports_native_json_schema)
        self.supports_strict_schema = bool(supports_strict_schema)
        self.supports_tool_schema = bool(supports_tool_schema)
        self.supports_json_mode = bool(supports_json_mode)

    def validate_or_fail(self, raw_output: dict[str, Any] | str, schema: type[T]) -> T:
        if isinstance(raw_output, dict):
            payload = raw_output
        elif isinstance(raw_output, str):
            payload = parse_json_object_strict(raw_output)
        else:
            raise ValueError("unsupported_output_type")
        return schema.model_validate(payload)

    def repair_once_if_allowed(
        self,
        *,
        raw_output: dict[str, Any] | str,
        schema: type[T],
        repair_callable: Callable[[dict[str, Any]], dict[str, Any] | str],
        allow_repair: bool,
    ) -> T:
        try:
            return self.validate_or_fail(raw_output, schema)
        except (ValidationError, ValueError) as exc:
            if not allow_repair:
                raise
            repaired_output = repair_callable(
                {
                    "provider": self.provider,
                    "model_name": self.model_name,
                    "schema_name": schema.__name__,
                    "error": type(exc).__name__,
                    "raw_output": raw_output,
                }
            )
            return self.validate_or_fail(repaired_output, schema)

    def complete_with_schema(
        self,
        *,
        schema: type[T],
        completion_callable: Callable[[dict[str, Any]], dict[str, Any] | str],
        repair_callable: Callable[[dict[str, Any]], dict[str, Any] | str] | None = None,
        allow_repair: bool = False,
    ) -> T:
        raw_output = completion_callable(
            {
                "provider": self.provider,
                "model_name": self.model_name,
                "schema": schema,
                "supports_native_json_schema": self.supports_native_json_schema,
                "supports_strict_schema": self.supports_strict_schema,
                "supports_tool_schema": self.supports_tool_schema,
                "supports_json_mode": self.supports_json_mode,
            }
        )
        if repair_callable is None:
            return self.validate_or_fail(raw_output, schema)
        return self.repair_once_if_allowed(
            raw_output=raw_output,
            schema=schema,
            repair_callable=repair_callable,
            allow_repair=allow_repair,
        )
