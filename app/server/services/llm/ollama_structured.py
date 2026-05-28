from __future__ import annotations

import json
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any, Literal, NoReturn, TypeAlias

import httpx
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from common.constants import (
    TEXT_EXTRACTION_MODEL_CHOICES,
)
from common.utils.logger import logger
from configurations.llm_configs import LLMRuntimeConfig
from services.llm.structured import (
    StructuredOutputParser,
    T,
    parse_json_dict,
)

ProviderName = Literal["openai", "gemini"]
RuntimePurpose = Literal["clinical", "parser"]


###############################################################################
class OllamaError(RuntimeError):
    pass


###############################################################################
class OllamaTimeout(OllamaError):
    """Raised when requests to Ollama exceed the configured timeout."""


ProgressCb: TypeAlias = Callable[[dict[str, Any]], None | Awaitable[None]]


###############################################################################
def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except TypeError, ValueError:
        return default


###############################################################################
def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


###############################################################################
def _build_langchain_messages(messages: list[dict[str, str]]) -> list[BaseMessage]:
    output: list[BaseMessage] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        content = str(message.get("content", ""))
        if role == "system":
            output.append(SystemMessage(content=content))
        elif role in {"assistant", "model"}:
            output.append(AIMessage(content=content))
        else:
            output.append(HumanMessage(content=content))
    return output


###############################################################################
def _normalize_langchain_content(content: Any) -> dict[str, Any] | str:
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
                continue
            if isinstance(item, str):
                chunks.append(item)
                continue
            chunks.append(str(item))
        content = "".join(chunks)
    if isinstance(content, str):
        try:
            loaded = json.loads(content)
        except json.JSONDecodeError:
            return content
        return loaded if isinstance(loaded, dict) else content
    return str(content)


###############################################################################
def _map_ollama_langchain_exception(exc: Exception) -> OllamaError:
    if isinstance(exc, OllamaError):
        return exc
    if isinstance(exc, TimeoutError):
        return OllamaTimeout("Timed out waiting for Ollama response")
    if isinstance(exc, httpx.TimeoutException):
        return OllamaTimeout("Timed out waiting for Ollama response")
    error_name = exc.__class__.__name__.lower()
    if "timeout" in error_name:
        return OllamaTimeout("Timed out waiting for Ollama response")
    return OllamaError(f"Ollama request failed: {exc}")


###############################################################################

# Extracted from the facade module; functions intentionally accept the facade instance.


async def collect_structured_fallbacks(self, preferred: list[str]) -> list[str]:
    available: set[str] = set()
    try:
        available = await self.get_cached_models()
    except OllamaError as exc:
        logger.debug("Failed to list Ollama models for fallback: %s", exc)
        available = set()

    fallbacks: list[str] = []
    if available:
        for name in TEXT_EXTRACTION_MODEL_CHOICES:
            if name in available and name not in preferred and name not in fallbacks:
                fallbacks.append(name)
    else:
        for name in TEXT_EXTRACTION_MODEL_CHOICES:
            if name not in preferred and name not in fallbacks:
                fallbacks.append(name)

    return fallbacks


async def llm_structured_call(
    self,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
    temperature: float = 0.0,
    use_json_mode: bool = True,
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
    preferred = await self.resolve_text_extraction_models(model)
    return await self.call_with_structured_models(
        parser=parser,
        messages=messages,
        system_prompt=system_prompt,
        format_instructions=format_instructions,
        preferred=preferred,
        temperature=temperature,
        use_json_mode=use_json_mode,
        max_repair_attempts=max_repair_attempts,
    )


def build_structured_messages(
    *,
    system_prompt: str,
    user_prompt: str,
    format_instructions: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": f"{system_prompt.strip()}\n\n{format_instructions}",
        },
        {"role": "user", "content": user_prompt},
    ]


async def resolve_text_extraction_models(self, model: str) -> list[str]:
    preferred: list[str] = []
    for candidate in (
        (model or "").strip(),
        (self.default_model or "").strip(),
        (LLMRuntimeConfig.get_text_extraction_model() or "").strip(),
    ):
        if candidate and candidate not in preferred:
            preferred.append(candidate)
    if not preferred:
        preferred = await self.collect_structured_fallbacks([])
    return preferred


def is_missing_model_error(err: OllamaError) -> bool:
    message = str(err).lower()
    return "not found" in message or "404" in message


async def _chat_structured_model(
    self,
    *,
    active_model: str,
    messages: list[dict[str, str]],
    use_json_mode: bool,
    temperature: float,
) -> dict[str, Any] | str:
    try:
        return await self.chat(
            model=active_model,
            messages=messages,
            format="json" if use_json_mode else None,
            temperature=temperature,
        )
    except OllamaError as err:
        if self.is_missing_model_error(err):
            raise
        raise RuntimeError(f"LLM call failed: {err}") from err


async def _extend_structured_model_queue(
    self,
    *,
    queue: list[str],
    preferred_models: list[str],
    tried: set[str],
    fallbacks: list[str] | None,
) -> list[str]:
    if fallbacks is None:
        fallbacks = await self.collect_structured_fallbacks(preferred_models)
    for candidate in fallbacks:
        if candidate and candidate not in tried and candidate not in queue:
            queue.append(candidate)
    return fallbacks


def _coerce_llm_text(raw: dict[str, Any] | str) -> str:
    return json.dumps(raw) if isinstance(raw, dict) else str(raw)


def _raise_structured_models_exhausted(
    *,
    last_missing_error: Exception | None,
    missing: list[str],
) -> NoReturn:
    if last_missing_error:
        attempted = ", ".join(missing)
        raise RuntimeError(
            "LLM call failed: no local text extraction models were found. "
            f"Tried: {attempted}"
        ) from last_missing_error
    raise RuntimeError("LLM call failed: no text extraction model candidates available")


def build_repair_messages(
    *,
    system_prompt: str,
    format_instructions: str,
    text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt.strip()},
        {
            "role": "user",
            "content": (
                "The previous reply did not match the required JSON schema.\n"
                "Follow these format instructions exactly and return ONLY a valid JSON object:\n"
                f"{format_instructions}\n\n"
                f"Previous reply:\n{text}"
            ),
        },
    ]


async def call_with_structured_models(
    self,
    *,
    parser: StructuredOutputParser[T],
    messages: list[dict[str, str]],
    system_prompt: str,
    format_instructions: str,
    preferred: list[str],
    temperature: float,
    use_json_mode: bool,
    max_repair_attempts: int,
) -> T:
    queue = preferred.copy()
    tried: set[str] = set()
    missing: list[str] = []
    last_missing_error: Exception | None = None
    fallbacks: list[str] | None = None

    while queue:
        active_model = queue.pop(0)
        if not active_model or active_model in tried:
            continue
        tried.add(active_model)

        try:
            raw = await self._chat_structured_model(
                active_model=active_model,
                messages=messages,
                use_json_mode=use_json_mode,
                temperature=temperature,
            )
        except OllamaError as e:
            missing.append(active_model)
            last_missing_error = e
            fallbacks = await self._extend_structured_model_queue(
                queue=queue,
                preferred_models=preferred,
                tried=tried,
                fallbacks=fallbacks,
            )
            continue

        return await self.parse_with_repairs(
            parser=parser,
            text=self._coerce_llm_text(raw),
            active_model=active_model,
            system_prompt=system_prompt,
            format_instructions=format_instructions,
            use_json_mode=use_json_mode,
            max_repair_attempts=max_repair_attempts,
        )

    self._raise_structured_models_exhausted(
        last_missing_error=last_missing_error,
        missing=missing,
    )


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
            try:
                raw = await self.chat(
                    model=active_model,
                    messages=repair_messages,
                    format="json" if use_json_mode else None,
                    temperature=0.0,
                )
                text = self._coerce_llm_text(raw)

            except OllamaError as e:
                raise RuntimeError(f"Repair attempt failed: {e}") from e

    raise RuntimeError("No structured output produced by the model")


def extract_first_json_object(text: str) -> str | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        start = match.start()
        try:
            parsed, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return text[start : start + end]
    return None


def parse_json(obj_or_text: dict[str, Any] | str) -> dict[str, Any] | None:
    return parse_json_dict(obj_or_text)
