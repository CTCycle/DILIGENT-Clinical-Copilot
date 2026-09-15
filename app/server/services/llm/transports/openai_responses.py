from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

from openai import AsyncOpenAI

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
class OpenAIResponsesTransport(StructuredTransportMixin):

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
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            default_headers=default_headers,
            max_retries=0,
        )
        self.max_retries = max(0, int(max_retries))

    # -------------------------------------------------------------------------
    @staticmethod
    def _message_items(message: ChatMessage) -> list[dict[str, Any]]:
        if message.role == "tool":
            return [
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id or "",
                    "output": str(message.content or ""),
                }
            ]
        if message.role == "assistant" and message.tool_calls:
            return [
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": call.raw_arguments
                    or json.dumps(call.arguments, ensure_ascii=False, separators=(",", ":")),
                }
                for call in message.tool_calls
            ]
        return [{"role": message.role, "content": message.content or ""}]

    # -------------------------------------------------------------------------
    @classmethod
    def _request_kwargs(cls, request: ChatRequest, *, stream: bool = False) -> dict[str, Any]:
        validate_request_capabilities(request.model_copy(update={"stream": stream}))
        instructions = "\n\n".join(
            str(item.content)
            for item in request.messages
            if item.role == "system" and item.content
        )
        inputs: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role != "system":
                inputs.extend(cls._message_items(message))
        kwargs: dict[str, Any] = {
            "model": request.model,
            "input": inputs or [{"role": "user", "content": ""}],
            **{
                key: value
                for key, value in request.options.items()
                if key not in {"max_output_tokens", "temperature", "top_p", "stream"}
            },
        }
        if instructions:
            kwargs["instructions"] = instructions
        output_token_limit = request_output_token_limit(request)
        if output_token_limit is not None:
            kwargs["max_output_tokens"] = output_token_limit
        temperature = request.temperature
        if temperature is None:
            temperature = request.options.get("temperature")
        top_p = request.top_p
        if top_p is None:
            top_p = request.options.get("top_p")
        if request.reasoning_level and request.reasoning_level != "off":
            temperature = None
            if request.reasoning_parameter in {"effort", "level"}:
                kwargs["reasoning"] = {"effort": request.reasoning_level}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if request.json_mode:
            if request.json_schema is not None and (
                request.capabilities is not None
                and request.capabilities.supports_native_json_schema
            ):
                kwargs["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "structured_response",
                        "strict": True,
                        "schema": request.json_schema,
                    }
                }
            else:
                kwargs["text"] = {"format": {"type": "json_object"}}
        if request.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.parameters,
                }
                for tool in request.tools
            ]
            if request.tool_choice is not None:
                kwargs["tool_choice"] = request.tool_choice
        if stream:
            kwargs["stream"] = True
        return kwargs

    # -------------------------------------------------------------------------
    @staticmethod
    def _check_active(request: ChatRequest) -> None:
        if request.cancel_check is not None and request.cancel_check():
            raise TransportCancellation("LLM request cancelled")
        if request.deadline_at is not None and asyncio.get_running_loop().time() >= request.deadline_at:
            raise TimeoutError("LLM request deadline exceeded")

    # -------------------------------------------------------------------------
    @staticmethod
    def _value(item: object, name: str, default: object = None) -> object:
        if isinstance(item, dict):
            return item.get(name, default)
        return getattr(item, name, default)

    # -------------------------------------------------------------------------
    @classmethod
    def _usage(cls, raw: object) -> UsageMetadata | None:
        if raw is None:
            return None
        input_tokens = cls._value(raw, "input_tokens")
        output_tokens = cls._value(raw, "output_tokens")
        total_tokens = cls._value(raw, "total_tokens")
        details = cls._value(raw, "output_tokens_details")
        reasoning_tokens = cls._value(details, "reasoning_tokens") if details else None
        values = {
            name: int(value)
            for name, value in {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": total_tokens,
            }.items()
            if isinstance(value, (int, float)) and value >= 0
        }
        return UsageMetadata(**values) if values else None

    # -------------------------------------------------------------------------
    @classmethod
    def _result_from_response(cls, response: object) -> ChatResult:
        text = cls._value(response, "output_text", "")
        if not isinstance(text, str):
            text = ""
        calls: list[ToolCall] = []
        output = cls._value(response, "output", [])
        if isinstance(output, list):
            for index, item in enumerate(output):
                item_type = str(cls._value(item, "type", ""))
                if item_type != "function_call":
                    continue
                name = cls._value(item, "name")
                if not name:
                    continue
                raw_arguments = str(cls._value(item, "arguments", "") or "")
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments = {}
                calls.append(
                    ToolCall(
                        id=str(cls._value(item, "call_id") or f"call_{index}"),
                        name=str(name),
                        arguments=arguments if isinstance(arguments, dict) else {},
                        raw_arguments=raw_arguments,
                    )
                )
        return ChatResult(
            content=text,
            tool_calls=calls,
            finish_reason=normalize_finish_reason(cls._value(response, "status")),
            usage=cls._usage(cls._value(response, "usage")),
            provider_model=str(cls._value(response, "model", "") or "") or None,
            request_id=str(cls._value(response, "id", "") or "") or None,
            tool_call_mode="native" if calls else None,
        )

    # -------------------------------------------------------------------------
    async def chat(self, request: ChatRequest) -> ChatResult:
        self._check_active(request)
        response = await call_with_retries(
            lambda: self.client.responses.create(**self._request_kwargs(request)),
            max_retries=getattr(self, "max_retries", 0),
            cancel_check=request.cancel_check,
        )
        return self._result_from_response(response)

    # -------------------------------------------------------------------------
    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        self._check_active(request)
        stream = await call_with_retries(
            lambda: self.client.responses.create(
                **self._request_kwargs(request, stream=True)
            ),
            max_retries=getattr(self, "max_retries", 0),
            cancel_check=request.cancel_check,
        )
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_state: dict[int, dict[str, str]] = {}
        usage: UsageMetadata | None = None
        finish_reason: str | None = None
        response_id: str | None = None
        provider_model: str | None = None
        completed = False
        try:
            async for event in stream:
                self._check_active(request)
                event_type = str(self._value(event, "type", ""))
                response_id = str(self._value(event, "response_id", response_id or "") or "") or response_id
                provider_model = str(self._value(event, "model", provider_model or "") or "") or provider_model
                if event_type.endswith("output_text.delta"):
                    text = str(self._value(event, "delta", "") or "")
                    if text:
                        text_parts.append(text)
                        yield ChatStreamEvent(kind="text_delta", text=text)
                elif "reasoning" in event_type and event_type.endswith("delta"):
                    reasoning = str(self._value(event, "delta", "") or "")
                    if reasoning:
                        reasoning_parts.append(reasoning)
                        yield ChatStreamEvent(kind="reasoning_delta", reasoning=reasoning)
                elif event_type.endswith("function_call_arguments.delta"):
                    index = int(str(self._value(event, "output_index", 0) or 0))
                    state = tool_state.setdefault(index, {})
                    call_id = self._value(event, "call_id")
                    name = self._value(event, "name")
                    if call_id:
                        state["id"] = str(call_id)
                    if name:
                        state["name"] = str(name)
                    delta = str(self._value(event, "delta", "") or "")
                    state["arguments"] = state.get("arguments", "") + delta
                    yield ChatStreamEvent(
                        kind="tool_call_delta",
                        tool_call_index=index,
                        tool_call_id=state.get("id"),
                        tool_name=state.get("name"),
                        arguments_delta=delta,
                    )
                elif event_type == "response.completed":
                    completed = True
                    response = self._value(event, "response")
                    if response is not None:
                        parsed = self._result_from_response(response)
                        usage = parsed.usage
                        finish_reason = parsed.finish_reason
                        response_id = parsed.request_id or response_id
                        provider_model = parsed.provider_model or provider_model
                        if parsed.content and not text_parts:
                            text_parts.append(parsed.content)
                        if parsed.reasoning_content and not reasoning_parts:
                            reasoning_parts.append(parsed.reasoning_content)
                        for index, call in enumerate(parsed.tool_calls):
                            tool_state[index] = {
                                "id": call.id,
                                "name": call.name,
                                "arguments": call.raw_arguments or json.dumps(call.arguments),
                            }
        finally:
            close_stream = getattr(stream, "close", None)
            if close_stream is not None:
                close_result = close_stream()
                if hasattr(close_result, "__await__"):
                    await close_result
        calls: list[ToolCall] = []
        for index, state in sorted(tool_state.items()):
            raw_arguments = state.get("arguments", "")
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {}
            calls.append(
                ToolCall(
                    id=state.get("id") or f"call_{index}",
                    name=state.get("name") or "",
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
            request_id=response_id,
            partial=not completed,
            tool_call_mode="native" if calls else None,
        )
        if not completed:
            yield ChatStreamEvent(
                kind="error",
                error_code="partial_response",
                error_message="OpenAI Responses stream ended before response.completed",
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
        page = await call_with_retries(
            self.client.models.list,
            max_retries=getattr(self, "max_retries", 0),
        )
        descriptors: list[CloudModelDescriptor] = []
        for item in page.data:
            raw_item = (
                item.model_dump(mode="json")
                if hasattr(item, "model_dump")
                else item.__dict__
                if hasattr(item, "__dict__")
                else {"id": item.id}
            )
            if not isinstance(raw_item, dict):
                raw_item = {"id": item.id}
            descriptors.append(
                CloudModelDescriptor(
                    id=item.id,
                    display_name=getattr(item, "name", None) or item.id,
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
        await self.client.close()
