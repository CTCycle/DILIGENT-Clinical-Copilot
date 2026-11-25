# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import json
from typing import Any

from common.prompts.structured_output import (
    COMPACT_JSON_REPAIR_SYSTEM_PROMPT,
    build_compact_json_repair_user_prompt,
    build_json_repair_user_prompt,
    build_structured_system_prompt,
)
from common.utils.logger import logger
from services.llm.generation_policy import GenerationPurpose
from services.llm.ollama_runtime import OllamaError
from services.llm.structured import (
    StructuredOutputParser,
    T,
    parse_json_object_strict,
)


###############################################################################
async def llm_structured_call(
    self,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
    temperature: float = 0.0,
    use_json_mode: bool = True,
    purpose: GenerationPurpose = GenerationPurpose.STRUCTURED_EXTRACTION,
    timeline_complexity: str = "moderate",
    max_repair_attempts: int = 2,
) -> T:
    """
    Call your Ollama LLM and validate the response against a Pydantic schema
    using a local JSON-schema-guided parser.

    - Injects format instructions so the LLM knows to return the expected JSON.
    - Parses & validates. If invalid, makes up to `max_repair_attempts` repair calls.
    - Returns an instance of `schema` (a Pydantic model).

    This function is LLM-agnostic beyond the Ollama client; you can reuse it
    across parsers by supplying different prompts/schemas.

    """
    parser = StructuredOutputParser(schema=schema)
    format_instructions = parser.get_format_instructions()
    messages = self.build_structured_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        format_instructions=format_instructions,
    )
    active_model = (model or self.default_model or "").strip()
    if not active_model:
        raise RuntimeError("LLM call failed: no Ollama model is configured.")
    raw = await self._chat_structured_model(
        active_model=active_model,
        messages=messages,
        use_json_mode=use_json_mode,
        temperature=temperature,
        purpose=purpose,
        timeline_complexity=timeline_complexity,
    )
    return await self.parse_with_repairs(
        parser=parser,
        text=self._coerce_llm_text(raw),
        active_model=active_model,
        system_prompt=system_prompt,
        format_instructions=format_instructions,
        use_json_mode=use_json_mode,
        max_repair_attempts=max_repair_attempts,
    )


###############################################################################
def build_structured_messages(
    *,
    system_prompt: str,
    user_prompt: str,
    format_instructions: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": build_structured_system_prompt(
                system_prompt=system_prompt,
                format_instructions=format_instructions,
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


###############################################################################
def is_missing_model_error(err: OllamaError) -> bool:
    message = str(err).lower()
    return "not found" in message or "404" in message


###############################################################################
async def _chat_structured_model(
    self,
    *,
    active_model: str,
    messages: list[dict[str, str]],
    use_json_mode: bool,
    temperature: float,
    purpose: GenerationPurpose,
    timeline_complexity: str = "moderate",
) -> dict[str, Any] | str:
    try:
        return await self.chat(
            model=active_model,
            messages=messages,
            format="json" if use_json_mode else None,
            temperature=temperature,
            purpose=purpose,
            timeline_complexity=timeline_complexity,
        )
    except OllamaError as err:
        if is_missing_model_error(err):
            raise RuntimeError(
                f"LLM call failed: the Ollama model '{active_model}' is not "
                "installed. Pull it or select another model for this agent."
            ) from err
        raise RuntimeError(f"LLM call failed: {err}") from err


###############################################################################
def _coerce_llm_text(raw: dict[str, Any] | str) -> str:
    return json.dumps(raw) if isinstance(raw, dict) else str(raw)


###############################################################################
def build_repair_messages(
    *,
    system_prompt: str,
    format_instructions: str,
    text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": build_json_repair_user_prompt(
                format_instructions=format_instructions,
                previous_reply=text,
            ),
        },
    ]


###############################################################################
def build_compact_repair_messages(
    *,
    text: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": COMPACT_JSON_REPAIR_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_compact_json_repair_user_prompt(previous_reply=text),
        },
    ]


###############################################################################
def looks_like_schema_echo(text: str) -> bool:
    lowered = text.casefold()
    schema_markers = (
        '"$defs"',
        '"properties"',
        '"required"',
        '"title"',
        '"type"',
        '"$ref"',
    )
    return sum(1 for marker in schema_markers if marker in lowered) >= 3


###############################################################################
async def parse_with_repairs(
    self,
    *,
    parser: StructuredOutputParser[T],
    text: str,
    active_model: str,
    system_prompt: str,
    format_instructions: str,
    use_json_mode: bool,
    max_repair_attempts: int,
) -> T:
    for attempt in range(max_repair_attempts + 1):
        try:
            return parser.parse(text)
        except Exception as err:
            if attempt >= max_repair_attempts:
                logger.error(
                    "Structured parse failed after retries. Last text: %s",
                    text,
                )
                raise RuntimeError(f"Structured parsing failed: {err}") from err

            repair_messages = self.build_repair_messages(
                system_prompt=system_prompt,
                format_instructions=format_instructions,
                text=text,
            )
            if looks_like_schema_echo(text):
                repair_messages = self.build_compact_repair_messages(text=text)
            try:
                raw = await self.chat(
                    model=active_model,
                    messages=repair_messages,
                    format="json" if use_json_mode else None,
                    temperature=0.0,
                    purpose=GenerationPurpose.JSON_REPAIR,
                )
                text = self._coerce_llm_text(raw)

            except OllamaError as e:
                raise RuntimeError(f"Repair attempt failed: {e}") from e

    raise RuntimeError("No structured output produced by the model")


###############################################################################
def parse_json(obj_or_text: dict[str, Any] | str) -> dict[str, Any] | None:
    if isinstance(obj_or_text, dict):
        return obj_or_text
    try:
        return parse_json_object_strict(obj_or_text)
    except ValueError:
        return None
