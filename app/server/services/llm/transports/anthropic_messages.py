from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam

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
    enter_async_context_with_retries,
    normalize_finish_reason,
    request_output_token_limit,
    validate_request_capabilities,
)
from services.llm.transports.errors import TransportCancellation
from services.llm.model_capabilities import catalog_capability_metadata

###############################################################################
class AnthropicMessagesTransport(StructuredTransportMixin):

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float,
        default_headers: Mapping[str, str] | None = None,
        max_retries: int = 2,
    ) -> None:
        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            default_headers=default_headers,
            max_retries=0,
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
    def _message_payload(cls, message: ChatMessage) -> dict[str, Any]:
        if message.role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id or "",
                        "content": str(message.content or ""),
                    }
                ],
            }
        if message.role == "assistant" and message.tool_calls:
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": str(message.content)})
            blocks.extend(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
                for call in message.tool_calls
            )
            return {"role": "assistant", "content": blocks}
        return {
            "role": "assistant" if message.role == "assistant" else "user",
            "content": message.content or "",
        }

    # -------------------------------------------------------------------------
    @classmethod
    def _request_kwargs(cls, request: ChatRequest, *, stream: bool = False) -> dict[str, Any]:
        validate_request_capabilities(request.model_copy(update={"stream": stream}))
        if request.json_mode and request.json_schema is None:
            raise ValueError("Anthropic JSON mode requires a JSON schema")
        system = "\n\n".join(
            str(item.content)
            for item in request.messages
            if item.role == "system" and item.content
        )
        messages = [
            cls._message_payload(item)
            for item in request.messages
            if item.role != "system"
        ]
        requested_max_tokens = request.options.get("max_tokens")
        output_token_limit = request_output_token_limit(request)
        max_tokens = int(output_token_limit or requested_max_tokens or 0)
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": cast(list[MessageParam], messages),
        }
        if system:
            kwargs["system"] = system
        if request.reasoning_level and request.reasoning_level != "off":
            budget_tokens = max(1024, int(request.reasoning_reserve or 0))
            max_tokens = max(max_tokens, int(request.output_token_limit or 0) + budget_tokens)
            if request.reasoning_parameter == "adaptive":
                kwargs["thinking"] = {"type": "adaptive"}
            else:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                }
        elif request.reasoning_parameter == "adaptive":
            kwargs["thinking"] = {"type": "disabled"}
        if max_tokens <= 0:
            raise ValueError("Anthropic requests require an output token limit")
        kwargs["max_tokens"] = max_tokens
        if request.json_schema is not None or request.reasoning_parameter == "adaptive":
            output_config: dict[str, Any] = {}
            if request.reasoning_parameter == "adaptive" and request.reasoning_level not in {
                None,
                "off",
            }:
                output_config["effort"] = request.reasoning_level
            if request.json_schema is not None:
                output_config["format"] = {
                    "type": "json_schema",
                    "schema": request.json_schema,
                }
            kwargs["output_config"] = output_config
        temperature = request.temperature
        if temperature is None:
            temperature = request.options.get("temperature")
        if (
            temperature is not None
            and request.reasoning_level in {None, "off"}
            and request.reasoning_parameter != "adaptive"
        ):
            kwargs["temperature"] = temperature
        top_p = request.top_p if request.top_p is not None else request.options.get("top_p")
        if top_p is not None and "temperature" not in kwargs:
            kwargs["top_p"] = top_p
        if request.tools and request.tool_choice != "none":
            kwargs["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.parameters,
                }
                for tool in request.tools
            ]
            if request.tool_choice is not None:
                kwargs["tool_choice"] = cls._tool_choice(request.tool_choice)
        if stream:
            kwargs["stream"] = True
        return kwargs

    # -------------------------------------------------------------------------
    @staticmethod
    def _tool_choice(value: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        normalized = value.strip().lower()
        if normalized == "auto":
            return {"type": "auto"}
        if normalized in {"required", "any"}:
            return {"type": "any"}
        if normalized == "none":
            return {"type": "auto"}
        raise ValueError(f"Unsupported Anthropic tool choice: {value}")

    # -------------------------------------------------------------------------
    @staticmethod
    def _check_active(request: ChatRequest) -> None:
        if request.cancel_check is not None and request.cancel_check():
            raise TransportCancellation("LLM request cancelled")
        if request.deadline_at is not None and asyncio.get_running_loop().time() >= request.deadline_at:
            raise TimeoutError("LLM request deadline exceeded")

    # -------------------------------------------------------------------------
    @classmethod
    def _result_from_response(cls, response: object) -> ChatResult:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: list[ToolCall] = []
        content = cls._value(response, "content", [])
        if isinstance(content, list):
            for index, block in enumerate(content):
                block_type = str(cls._value(block, "type", ""))
                if block_type == "text" or cls._value(block, "text") is not None:
                    text_parts.append(str(cls._value(block, "text", "")))
                elif block_type in {"thinking", "redacted_thinking"}:
                    thinking = cls._value(block, "thinking", "")
                    if thinking:
                        reasoning_parts.append(str(thinking))
                elif block_type == "tool_use":
                    name = cls._value(block, "name", "")
                    if name:
                        arguments = cls._value(block, "input", {})
                        calls.append(
                            ToolCall(
                                id=str(cls._value(block, "id") or f"call_{index}"),
                                name=str(name),
                                arguments=arguments if isinstance(arguments, dict) else {},
                            )
                        )
        usage_raw = cls._value(response, "usage")
        usage_values: dict[str, int] = {}
        for target, source in {
            "input_tokens": "input_tokens",
            "output_tokens": "output_tokens",
        }.items():
            value = cls._value(usage_raw, source)
            if isinstance(value, (int, float)) and value >= 0:
                usage_values[target] = int(value)
        if usage_values.get("input_tokens") is not None and usage_values.get("output_tokens") is not None:
            usage_values["total_tokens"] = usage_values["input_tokens"] + usage_values["output_tokens"]
        return ChatResult(
            content="".join(text_parts),
            reasoning_content="".join(reasoning_parts) or None,
            tool_calls=calls,
            finish_reason=normalize_finish_reason(cls._value(response, "stop_reason")),
            usage=UsageMetadata(**usage_values) if usage_values else None,
            provider_model=str(cls._value(response, "model", "") or "") or None,
            request_id=str(cls._value(response, "id", "") or "") or None,
            tool_call_mode="native" if calls else None,
        )

    # -------------------------------------------------------------------------
    async def chat(self, request: ChatRequest) -> ChatResult:
        self._check_active(request)
        response = await call_with_retries(
            lambda: cast(Any, self.client.messages).create(
                **self._request_kwargs(request)
            ),
            max_retries=getattr(self, "max_retries", 0),
            cancel_check=request.cancel_check,
        )
        return self._result_from_response(response)

    # -------------------------------------------------------------------------
    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        self._check_active(request)
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_state: dict[int, dict[str, Any]] = {}
        usage: UsageMetadata | None = None
        finish_reason: str | None = None
        provider_model: str | None = None
        request_id: str | None = None
        completed = False
        context, stream = await enter_async_context_with_retries(
            lambda: cast(Any, self.client.messages).stream(
                **self._request_kwargs(request, stream=True)
            ),
            max_retries=getattr(self, "max_retries", 0),
            cancel_check=request.cancel_check,
        )
        try:
            async for event in stream:
                self._check_active(request)
                event_type = str(self._value(event, "type", ""))
                if event_type == "message_start":
                    message = self._value(event, "message")
                    provider_model = str(self._value(message, "model", "") or "") or None
                    request_id = str(self._value(message, "id", "") or "") or None
                    usage_raw = self._value(message, "usage")
                    input_tokens = self._value(usage_raw, "input_tokens")
                    if isinstance(input_tokens, (int, float)):
                        usage = UsageMetadata(input_tokens=int(input_tokens))
                elif event_type == "content_block_start":
                    index = int(self._value(event, "index", 0) or 0)
                    block = self._value(event, "content_block")
                    block_type = str(self._value(block, "type", ""))
                    state = tool_state.setdefault(index, {})
                    if block_type == "tool_use":
                        state["id"] = str(self._value(block, "id") or f"call_{index}")
                        state["name"] = str(self._value(block, "name") or "")
                elif event_type == "content_block_delta":
                    index = int(self._value(event, "index", 0) or 0)
                    delta = self._value(event, "delta")
                    delta_type = str(self._value(delta, "type", ""))
                    if delta_type == "text_delta":
                        text = str(self._value(delta, "text", "") or "")
                        text_parts.append(text)
                        if text:
                            yield ChatStreamEvent(kind="text_delta", text=text)
                    elif delta_type in {"thinking_delta", "signature_delta"}:
                        thinking = str(
                            self._value(delta, "thinking", "")
                            or self._value(delta, "signature", "")
                            or ""
                        )
                        reasoning_parts.append(thinking)
                        if thinking:
                            yield ChatStreamEvent(kind="reasoning_delta", reasoning=thinking)
                    elif delta_type == "input_json_delta":
                        partial = str(self._value(delta, "partial_json", "") or "")
                        state = tool_state.setdefault(index, {})
                        state["arguments"] = state.get("arguments", "") + partial
                        yield ChatStreamEvent(
                            kind="tool_call_delta",
                            tool_call_index=index,
                            tool_call_id=state.get("id"),
                            tool_name=state.get("name"),
                            arguments_delta=partial,
                        )
                elif event_type == "message_delta":
                    delta = self._value(event, "delta")
                    finish_reason = normalize_finish_reason(
                        self._value(delta, "stop_reason") or finish_reason
                    )
                    usage_raw = self._value(event, "usage")
                    output_tokens = self._value(usage_raw, "output_tokens")
                    if isinstance(output_tokens, (int, float)):
                        usage = UsageMetadata(
                            input_tokens=usage.input_tokens if usage else None,
                            output_tokens=int(output_tokens),
                            total_tokens=(
                                (usage.input_tokens if usage else 0) + int(output_tokens)
                                if usage and usage.input_tokens is not None
                                else None
                            ),
                        )
                        yield ChatStreamEvent(kind="usage", usage=usage)
                elif event_type == "message_stop":
                    completed = True
        finally:
            await context.__aexit__(None, None, None)
        calls: list[ToolCall] = []
        for index, state in sorted(tool_state.items()):
            raw_arguments = str(state.get("arguments", "") or "")
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {}
            calls.append(
                ToolCall(
                    id=str(state.get("id") or f"call_{index}"),
                    name=str(state.get("name") or ""),
                    arguments=arguments if isinstance(arguments, dict) else {},
                    raw_arguments=raw_arguments,
                )
            )
        result = ChatResult(
            content="".join(text_parts),
            reasoning_content="".join(reasoning_parts) or None,
            tool_calls=[call for call in calls if call.name],
            finish_reason=finish_reason,
            usage=usage,
            provider_model=provider_model,
            request_id=request_id,
            partial=not completed,
            tool_call_mode="native" if calls else None,
        )
        if not completed:
            yield ChatStreamEvent(
                kind="error",
                error_code="partial_response",
                error_message="Anthropic stream ended before message_stop",
            )
        yield ChatStreamEvent(
            kind="completed",
            finish_reason=finish_reason,
            usage=usage,
            result=result,
        )

    # -------------------------------------------------------------------------
    async def list_models(
        self, *, force_refresh: bool = False
    ) -> list[CloudModelDescriptor]:
        del force_refresh
        models: list[CloudModelDescriptor] = []
        after_id: str | None = None
        while True:
            if after_id is None:
                page = await call_with_retries(
                    lambda: self.client.models.list(limit=100),
                    max_retries=getattr(self, "max_retries", 0),
                )
            else:
                page = await call_with_retries(
                    lambda: self.client.models.list(limit=100, after_id=after_id),
                    max_retries=getattr(self, "max_retries", 0),
                )
            for item in page.data:
                metadata = self._model_metadata(item)
                capabilities = metadata.get("capabilities")
                if not isinstance(capabilities, dict):
                    capabilities = {}
                thinking = capabilities.get("thinking")
                structured = capabilities.get("structured_outputs")
                if structured is None:
                    structured = capabilities.get("structured_output")
                models.append(
                    CloudModelDescriptor(
                        id=item.id,
                        display_name=item.display_name,
                        model_capabilities=catalog_capability_metadata(
                            {
                                **metadata,
                                "supports_reasoning": thinking,
                                "supports_json_mode": structured,
                                "supports_native_json_schema": structured,
                            }
                        ),
                    )
                )
            if not getattr(page, "has_more", False) or not page.data:
                return models
            after_id = page.data[-1].id

    # -------------------------------------------------------------------------
    @staticmethod
    def _model_metadata(item: object) -> dict[str, object]:
        if hasattr(item, "model_dump"):
            dumped = item.model_dump(mode="json")  # type: ignore[attr-defined]
            return dumped if isinstance(dumped, dict) else {}
        if hasattr(item, "dict"):
            dumped = item.dict()  # type: ignore[attr-defined]
            return dumped if isinstance(dumped, dict) else {}
        return {}

    # -------------------------------------------------------------------------
    async def check_connectivity(self, model: str) -> ConnectivityResult:
        try:
            result = await self.chat(
                ChatRequest(
                    model=model,
                    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                    options={"max_tokens": 16},
                    output_token_limit=16,
                    operation="connectivity",
                )
            )
            return ConnectivityResult(ok=True, response_preview=result.content[:200])
        except Exception as exc:
            return ConnectivityResult(ok=False, error=str(exc))

    # -------------------------------------------------------------------------
    async def close(self) -> None:
        await self.client.close()
