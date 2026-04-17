from __future__ import annotations

import json
import re
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

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
