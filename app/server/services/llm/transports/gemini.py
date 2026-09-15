from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import types

from domain.llm.providers import CloudModelDescriptor
from domain.llm.transports import (
    ChatMessage,
    ChatRequest,
    ChatResult,
    ChatStreamEvent,
    ConnectivityResult,
    ToolCall,
    UsageMetadata,
)
from services.llm.transports.base import (
    StructuredTransportMixin,
    call_with_retries,
    normalize_finish_reason,
    request_output_token_limit,
    validate_request_capabilities,
)
from services.llm.transports.errors import TransportCancellation
from services.llm.model_capabilities import catalog_capability_metadata

###############################################################################
def gemini_model_requires_thinking(model: str) -> bool:
    normalized = (model or "").strip().lower()
    return normalized.startswith("gemini-3") or normalized.startswith(
        "gemini-2.5-pro"
    )

###############################################################################
class GeminiTransport(StructuredTransportMixin):

    # -------------------------------------------------------------------------
    def __init__(self, *, api_key: str, timeout: float, max_retries: int = 2) -> None:
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=max(1, int(timeout * 1000))),
        )
        self.max_retries = max(0, int(max_retries))

    # -------------------------------------------------------------------------
    @staticmethod
    def _value(item: object, name: str, default: object = None) -> object:
        if isinstance(item, dict):
            return item.get(name, default)
        return getattr(item, name, default)

    # -------------------------------------------------------------------------
    @classmethod
    def _contents(
        cls, messages: list[ChatMessage]
    ) -> tuple[str | None, list[types.Content]]:
        system_parts: list[str] = []
        contents: list[types.Content] = []
        for message in messages:
            if message.role == "system":
                if message.content:
                    system_parts.append(str(message.content))
                continue
            parts: list[types.Part] = []
            if message.content:
                parts.append(types.Part(text=str(message.content)))
            if message.role == "assistant" and message.tool_calls:
                parts.extend(
                    types.Part(
                        function_call=types.FunctionCall(
                            id=call.id,
                            name=call.name,
                            args=call.arguments,
                        )
                    )
                    for call in message.tool_calls
                )
            if message.role == "tool":
                parts = [
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=message.tool_call_id or "",
                            name=message.name or "",
                            response={"content": str(message.content or "")},
                        )
                    )
                ]
            if not parts:
                parts = [types.Part(text="")]
            contents.append(
                types.Content(
                    role="model" if message.role == "assistant" else "user",
                    parts=parts,
                )
            )
        if not contents:
            contents.append(types.Content(role="user", parts=[types.Part(text="")]))
        return "\n\n".join(system_parts) or None, contents

    # -------------------------------------------------------------------------
    @classmethod
    def _tool_config(cls, request: ChatRequest) -> types.ToolConfig | None:
        if request.tool_choice is None:
            return None
        mode = types.FunctionCallingConfigMode.AUTO
        allowed: list[str] | None = None
        if isinstance(request.tool_choice, str):
            mode = {
                "auto": types.FunctionCallingConfigMode.AUTO,
                "none": types.FunctionCallingConfigMode.NONE,
                "required": types.FunctionCallingConfigMode.ANY,
            }.get(
                request.tool_choice.lower(), types.FunctionCallingConfigMode.AUTO
            )
        elif isinstance(request.tool_choice, dict):
            mode = types.FunctionCallingConfigMode.ANY
            name = request.tool_choice.get("name")
            if name:
                allowed = [str(name)]
        return types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=mode,
                allowed_function_names=allowed,
            )
        )

    # -------------------------------------------------------------------------
    @classmethod
    def _config(cls, request: ChatRequest) -> types.GenerateContentConfig:
        system, _ = cls._contents(request.messages)
        kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": (
                None
                if request.reasoning_level and request.reasoning_level != "off"
                else request.temperature
                if request.temperature is not None
                else request.options.get("temperature")
            ),
            "top_p": request.top_p if request.top_p is not None else request.options.get("top_p"),
            "max_output_tokens": request_output_token_limit(request),
            "response_mime_type": "application/json" if request.json_mode else None,
            "thinking_config": cls._thinking_config(request),
        }
        if request.json_schema is not None:
            kwargs["response_json_schema"] = request.json_schema
        if request.tools:
            kwargs["tools"] = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description,
                            parameters_json_schema=tool.parameters,
                        )
                    ]
                )
                for tool in request.tools
            ]
            tool_config = cls._tool_config(request)
            if tool_config is not None:
                kwargs["tool_config"] = tool_config
        return types.GenerateContentConfig(**kwargs)

    # -------------------------------------------------------------------------
    @staticmethod
    def _thinking_config(request: ChatRequest) -> types.ThinkingConfig | None:
        if not request.reasoning_level or request.reasoning_parameter != "level":
            return None
        if request.reasoning_level == "off":
            if gemini_model_requires_thinking(request.model):
                return types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
            return types.ThinkingConfig(thinking_budget=0)
        sdk_level = (
            types.ThinkingLevel.LOW
            if request.reasoning_level in {"low", "medium"}
            else types.ThinkingLevel.HIGH
        )
        return types.ThinkingConfig(thinking_level=sdk_level)

    # -------------------------------------------------------------------------
    @staticmethod
    def _check_active(request: ChatRequest) -> None:
        if request.cancel_check is not None and request.cancel_check():
            raise TransportCancellation("LLM request cancelled")
        if request.deadline_at is not None and asyncio.get_running_loop().time() >= request.deadline_at:
            raise TimeoutError("LLM request deadline exceeded")

    # -------------------------------------------------------------------------
    @classmethod
    def _result(cls, response: object) -> ChatResult:
        candidates = cls._value(response, "candidates", [])
        candidate = candidates[0] if isinstance(candidates, list) and candidates else None
        content = cls._value(candidate, "content", None)
        parts = cls._value(content, "parts", [])
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: list[ToolCall] = []
        if isinstance(parts, list):
            for index, part in enumerate(parts):
                text = cls._value(part, "text", "")
                if text:
                    if bool(cls._value(part, "thought", False)):
                        reasoning_parts.append(str(text))
                    else:
                        text_parts.append(str(text))
                function_call = cls._value(part, "function_call", None)
                if function_call is not None:
                    name = cls._value(function_call, "name", "")
                    if name:
                        args = cls._value(function_call, "args", {})
                        calls.append(
                            ToolCall(
                                id=str(cls._value(function_call, "id") or f"call_{index}"),
                                name=str(name),
                                arguments=args if isinstance(args, dict) else {},
                            )
                        )
        usage_raw = cls._value(response, "usage_metadata")
        usage_values: dict[str, int] = {}
        for target, source in {
            "input_tokens": "prompt_token_count",
            "output_tokens": "candidates_token_count",
            "reasoning_tokens": "thoughts_token_count",
            "total_tokens": "total_token_count",
        }.items():
            value = cls._value(usage_raw, source)
            if isinstance(value, (int, float)) and value >= 0:
                usage_values[target] = int(value)
        return ChatResult(
            content="".join(text_parts),
            reasoning_content="".join(reasoning_parts) or None,
            tool_calls=calls,
            finish_reason=normalize_finish_reason(
                cls._value(candidate, "finish_reason")
            ),
            usage=UsageMetadata(**usage_values) if usage_values else None,
            provider_model=str(cls._value(response, "model_version", "") or "") or None,
            request_id=str(cls._value(response, "response_id", "") or "") or None,
            tool_call_mode="native" if calls else None,
        )

    # -------------------------------------------------------------------------
    async def chat(self, request: ChatRequest) -> ChatResult:
        self._check_active(request)
        _, contents = self._contents(request.messages)
        validate_request_capabilities(request.model_copy(update={"stream": False}))
        response = await call_with_retries(
            lambda: asyncio.to_thread(
                self.client.models.generate_content,
                model=request.model,
                contents=contents,
                config=self._config(request),
            ),
            max_retries=getattr(self, "max_retries", 0),
            cancel_check=request.cancel_check,
        )
        return self._result(response)

    # -------------------------------------------------------------------------
    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        self._check_active(request)
        validate_request_capabilities(request.model_copy(update={"stream": True}))
        _, contents = self._contents(request.messages)
        events: queue.Queue[object] = queue.Queue()
        sentinel = object()

        def produce() -> None:
            try:
                for item in self.client.models.generate_content_stream(
                    model=request.model,
                    contents=contents,
                    config=self._config(request),
                ):
                    events.put(item)
            except Exception as exc:  # noqa: BLE001
                events.put(exc)
            finally:
                events.put(sentinel)

        task = asyncio.create_task(asyncio.to_thread(produce))
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: dict[int, ToolCall] = {}
        last_response: object | None = None
        last_result: ChatResult | None = None
        completed = False
        try:
            while True:
                self._check_active(request)
                item = await asyncio.to_thread(events.get)
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                last_response = item
                result = self._result(item)
                last_result = result
                previous_text = "".join(text_parts)
                text = (
                    result.content[len(previous_text) :]
                    if result.content.startswith(previous_text)
                    else result.content
                )
                if text:
                    text_parts.append(text)
                    yield ChatStreamEvent(kind="text_delta", text=text)
                previous_reasoning = "".join(reasoning_parts)
                current_reasoning = result.reasoning_content or ""
                reasoning = (
                    current_reasoning[len(previous_reasoning) :]
                    if current_reasoning.startswith(previous_reasoning)
                    else current_reasoning
                )
                if reasoning:
                    reasoning_parts.append(reasoning)
                    yield ChatStreamEvent(kind="reasoning_delta", reasoning=reasoning)
                for index, call in enumerate(result.tool_calls):
                    if index not in calls or calls[index] != call:
                        calls[index] = call
                        yield ChatStreamEvent(
                            kind="tool_call_delta",
                            tool_call_index=index,
                            tool_call_id=call.id,
                            tool_name=call.name,
                            arguments_delta=call.raw_arguments or "",
                        )
                if result.usage:
                    yield ChatStreamEvent(kind="usage", usage=result.usage)
                if result.finish_reason:
                    completed = True
        finally:
            if not task.done():
                task.cancel()
        result = last_result or (
            self._result(last_response)
            if last_response is not None
            else ChatResult(
                content="",
                reasoning_content="".join(reasoning_parts) or None,
            )
        )
        result = result.model_copy(
            update={
                "content": "".join(text_parts),
                "reasoning_content": "".join(reasoning_parts) or None,
                "tool_calls": list(calls.values()),
                "partial": not completed,
            }
        )
        if not completed:
            yield ChatStreamEvent(
                kind="error",
                error_code="partial_response",
                error_message="Gemini stream ended without a finish reason",
            )
        yield ChatStreamEvent(kind="completed", finish_reason=result.finish_reason, usage=result.usage, result=result)

    # -------------------------------------------------------------------------
    async def list_models(
        self, *, force_refresh: bool = False
    ) -> list[CloudModelDescriptor]:
        del force_refresh
        page = await call_with_retries(
            lambda: asyncio.to_thread(lambda: list(self.client.models.list())),
            max_retries=getattr(self, "max_retries", 0),
        )
        descriptors: list[CloudModelDescriptor] = []
        for item in page:
            if "generateContent" not in (item.supported_actions or []):
                continue
            raw_item = (
                item.model_dump(mode="json")
                if hasattr(item, "model_dump")
                else item.__dict__
                if hasattr(item, "__dict__")
                else {}
            )
            if not isinstance(raw_item, dict):
                raw_item = {}
            raw_item.update(
                {
                    "input_token_limit": getattr(item, "input_token_limit", None),
                    "output_token_limit": getattr(item, "output_token_limit", None),
                    "supports_reasoning": getattr(item, "thinking", None),
                    "supports_temperature": getattr(item, "temperature", None),
                    "supports_chat": True,
                    "supports_streaming": True,
                }
            )
            descriptors.append(
                CloudModelDescriptor(
                    id=str(item.name).removeprefix("models/"),
                    display_name=str(item.display_name or item.name),
                    model_capabilities=catalog_capability_metadata(raw_item),
                )
            )
        return descriptors

    # -------------------------------------------------------------------------
    async def check_connectivity(self, model: str) -> ConnectivityResult:
        try:
            result = await self.chat(
                ChatRequest(
                    model=model,
                    messages=[ChatMessage(role="user", content="Reply with exactly: OK")],
                    output_token_limit=16,
                    operation="connectivity",
                )
            )
            return ConnectivityResult(ok=True, response_preview=result.content[:200])
        except Exception as exc:
            return ConnectivityResult(ok=False, error=str(exc))

    # -------------------------------------------------------------------------
    async def close(self) -> None:
        self.client.close()
