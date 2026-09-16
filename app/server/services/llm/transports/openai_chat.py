from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

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
    request_output_token_limit,
    normalize_finish_reason,
    validate_request_capabilities,
)
from services.llm.transports.errors import TransportCancellation
from services.llm.model_capabilities import catalog_capability_metadata

###############################################################################
class OpenAIChatTransport(StructuredTransportMixin):

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
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if default_headers:
            headers.update(default_headers)
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            trust_env=False,
            headers=headers,
        )
        self.max_retries = max(0, int(max_retries))

    # -------------------------------------------------------------------------
    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, str):
                    chunks.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
            return "".join(chunks)
        if isinstance(content, dict) and isinstance(content.get("text"), str):
            return content["text"]
        return str(content or "")

    # -------------------------------------------------------------------------
    @staticmethod
    def _message_payload(message: ChatMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": message.role}
        if message.content is not None:
            payload["content"] = message.content
        if message.name:
            payload["name"] = message.name
        if message.reasoning_content:
            payload["reasoning_content"] = message.reasoning_content
        if message.tool_calls:
            payload.setdefault("content", "")
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.raw_arguments
                        or json.dumps(
                            call.arguments,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                }
                for call in message.tool_calls
            ]
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        return payload

    # -------------------------------------------------------------------------
    @classmethod
    def _request_payload(cls, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        validate_request_capabilities(request.model_copy(update={"stream": stream}))
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [cls._message_payload(item) for item in request.messages],
            "stream": stream,
        }
        excluded = {
            "max_output_tokens",
            "temperature",
            "top_p",
            "stream",
            "tools",
            "tool_choice",
        }
        payload.update(
            {
                key: value
                for key, value in request.options.items()
                if key not in excluded
            }
        )
        temperature = (
            request.temperature
            if request.temperature is not None
            else request.options.get("temperature")
        )
        top_p = request.top_p if request.top_p is not None else request.options.get("top_p")
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        output_token_limit = request_output_token_limit(request)
        if output_token_limit is not None:
            payload["max_tokens"] = output_token_limit

        if request.reasoning_level and request.reasoning_level != "off":
            # DeepSeek thinking mode rejects temperature.  The same omission is
            # safe for other OpenAI-compatible reasoning endpoints.
            payload.pop("temperature", None)
            if request.capabilities is not None and (
                request.capabilities.reasoning_toggle_parameter == "thinking"
            ):
                payload["thinking"] = {"type": "enabled"}
            if request.reasoning_parameter == "boolean":
                payload["thinking"] = {"type": "enabled"}
            elif request.reasoning_parameter in {"effort", "level"}:
                payload["reasoning_effort"] = cls._reasoning_effort(request)
        elif request.reasoning_parameter == "boolean":
            payload["thinking"] = {"type": "disabled"}
        elif request.capabilities is not None and (
            request.capabilities.reasoning_toggle_parameter == "thinking"
        ):
            payload["thinking"] = {"type": "disabled"}

        if request.json_mode:
            if request.json_schema is not None and (
                request.capabilities is not None
                and request.capabilities.supports_native_json_schema
            ):
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_response",
                        "strict": True,
                        "schema": request.json_schema,
                    },
                }
            else:
                payload["response_format"] = {"type": "json_object"}
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]
            if request.tool_choice is not None:
                payload["tool_choice"] = request.tool_choice
        return payload

    # -------------------------------------------------------------------------
    @staticmethod
    def _reasoning_effort(request: ChatRequest) -> str:
        requested = request.reasoning_level or "high"
        supported = (
            request.capabilities.reasoning_levels
            if request.capabilities is not None
            else ()
        )
        if requested in supported:
            return requested
        if requested == "medium" and "high" in supported:
            return "high"
        if "high" in supported:
            return "high"
        if "low" in supported:
            return "low"
        return requested

    # -------------------------------------------------------------------------
    @staticmethod
    def _check_active(request: ChatRequest) -> None:
        if request.cancel_check is not None and request.cancel_check():
            raise TransportCancellation("LLM request cancelled")
        if request.deadline_at is not None:
            now = asyncio.get_running_loop().time()
            if now >= request.deadline_at:
                raise TimeoutError("LLM request deadline exceeded")

    # -------------------------------------------------------------------------
    @staticmethod
    def _parse_arguments(raw: object) -> tuple[dict[str, Any], str | None]:
        if isinstance(raw, dict):
            return dict(raw), None
        text = str(raw or "")
        if not text:
            return {}, text
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}, text
        return (dict(parsed), text) if isinstance(parsed, dict) else ({}, text)

    # -------------------------------------------------------------------------
    @classmethod
    def _parse_usage(cls, raw: object) -> UsageMetadata | None:
        if not isinstance(raw, dict):
            return None
        details = raw.get("completion_tokens_details")
        reasoning_tokens = (
            details.get("reasoning_tokens")
            if isinstance(details, dict)
            else raw.get("reasoning_tokens")
        )
        values: dict[str, int | None] = {}
        for target, keys in {
            "input_tokens": ("prompt_tokens", "input_tokens"),
            "output_tokens": ("completion_tokens", "output_tokens"),
            "total_tokens": ("total_tokens",),
            "reasoning_tokens": ("reasoning_tokens",),
        }.items():
            for key in keys:
                value = raw.get(key) if key != "reasoning_tokens" else reasoning_tokens
                if isinstance(value, (int, float)) and value >= 0:
                    values[target] = int(value)
                    break
        if not values:
            return None
        return UsageMetadata(**values)

    # -------------------------------------------------------------------------
    @classmethod
    def _parse_result(cls, payload: object, response: object | None = None) -> ChatResult:
        if not isinstance(payload, dict):
            raise ValueError("OpenAI-compatible response was not a JSON object")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("OpenAI-compatible response did not contain choices")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("OpenAI-compatible response did not contain a message")
        calls: list[ToolCall] = []
        raw_calls = message.get("tool_calls")
        if isinstance(raw_calls, list):
            for index, raw_call in enumerate(raw_calls):
                if not isinstance(raw_call, dict):
                    continue
                function = raw_call.get("function")
                if not isinstance(function, dict) or not function.get("name"):
                    continue
                arguments, raw_arguments = cls._parse_arguments(function.get("arguments"))
                calls.append(
                    ToolCall(
                        id=str(raw_call.get("id") or f"call_{index}"),
                        name=str(function["name"]),
                        arguments=arguments,
                        raw_arguments=raw_arguments,
                    )
                )
        headers = getattr(response, "headers", {}) if response is not None else {}
        request_id = next(
            (
                str(headers.get(name))
                for name in ("x-request-id", "request-id")
                if headers.get(name)
            ),
            None,
        )
        return ChatResult(
            content=cls._content_to_text(message.get("content")),
            reasoning_content=str(message.get("reasoning_content") or "") or None,
            tool_calls=calls,
            finish_reason=normalize_finish_reason(choice.get("finish_reason")),
            usage=cls._parse_usage(payload.get("usage")),
            provider_model=str(payload.get("model") or "") or None,
            request_id=request_id,
            tool_call_mode="native" if calls else None,
        )

    # -------------------------------------------------------------------------
    async def chat(self, request: ChatRequest) -> ChatResult:
        self._check_active(request)
        payload = self._request_payload(request, stream=False)

        async def post_checked() -> httpx.Response:
            response = await self.client.post("chat/completions", json=payload)
            response.raise_for_status()
            return response

        operation = asyncio.create_task(
            call_with_retries(
                post_checked,
                max_retries=getattr(self, "max_retries", 0),
                cancel_check=request.cancel_check,
            )
        )
        try:
            while not operation.done():
                await asyncio.wait({operation}, timeout=0.25)
                if not operation.done():
                    self._check_active(request)
            response = operation.result()
        except BaseException:
            if not operation.done():
                operation.cancel()
            try:
                await operation
            except BaseException:
                pass
            raise
        return self._parse_result(response.json(), response)

    # -------------------------------------------------------------------------
    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        self._check_active(request)
        payload = self._request_payload(request, stream=True)
        if (
            request.capabilities is None
            or request.capabilities.supports_usage_metadata is not False
        ):
            payload["stream_options"] = {"include_usage": True}
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_state: dict[int, dict[str, str]] = {}
        usage: UsageMetadata | None = None
        finish_reason: str | None = None
        provider_model: str | None = None
        request_id: str | None = None
        completed = False
        context, response = await enter_async_context_with_retries(
            lambda: self.client.stream("POST", "chat/completions", json=payload),
            max_retries=getattr(self, "max_retries", 0),
            cancel_check=request.cancel_check,
            validate=lambda item: item.raise_for_status(),
        )
        try:
            response.raise_for_status()
            headers = getattr(response, "headers", {})
            request_id = next(
                (
                    str(headers.get(name))
                    for name in ("x-request-id", "request-id")
                    if headers.get(name)
                ),
                None,
            )
            async for line in response.aiter_lines():
                self._check_active(request)
                if not line:
                    continue
                data_line = line[5:].strip() if line.startswith("data:") else line.strip()
                if data_line == "[DONE]":
                    completed = True
                    break
                try:
                    payload_chunk = json.loads(data_line)
                except json.JSONDecodeError:
                    yield ChatStreamEvent(
                        kind="error",
                        error_code="malformed_response",
                        error_message="Malformed OpenAI-compatible stream event",
                    )
                    return
                if not isinstance(payload_chunk, dict):
                    continue
                provider_model = str(payload_chunk.get("model") or provider_model or "") or None
                chunk_usage = self._parse_usage(payload_chunk.get("usage"))
                if chunk_usage is not None:
                    usage = chunk_usage
                    yield ChatStreamEvent(kind="usage", usage=chunk_usage)
                choices = payload_chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                finish_reason = normalize_finish_reason(
                    choice.get("finish_reason") or finish_reason
                )
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                text = self._content_to_text(delta.get("content"))
                if text:
                    text_parts.append(text)
                    yield ChatStreamEvent(kind="text_delta", text=text)
                reasoning = str(delta.get("reasoning_content") or "")
                if reasoning:
                    reasoning_parts.append(reasoning)
                    yield ChatStreamEvent(kind="reasoning_delta", reasoning=reasoning)
                raw_calls = delta.get("tool_calls")
                if isinstance(raw_calls, list):
                    for raw_call in raw_calls:
                        if not isinstance(raw_call, dict):
                            continue
                        index = int(raw_call.get("index", 0) or 0)
                        state = tool_state.setdefault(index, {})
                        if raw_call.get("id"):
                            state["id"] = str(raw_call["id"])
                        function = raw_call.get("function")
                        if isinstance(function, dict):
                            if function.get("name"):
                                state["name"] = str(function["name"])
                            arguments = str(function.get("arguments") or "")
                            if arguments:
                                state["arguments"] = state.get("arguments", "") + arguments
                                yield ChatStreamEvent(
                                    kind="tool_call_delta",
                                    tool_call_index=index,
                                    tool_call_id=state.get("id"),
                                    tool_name=state.get("name"),
                                    arguments_delta=arguments,
                                )
            calls: list[ToolCall] = []
            for index, state in sorted(tool_state.items()):
                arguments, raw_arguments = self._parse_arguments(state.get("arguments", ""))
                calls.append(
                    ToolCall(
                        id=state.get("id") or f"call_{index}",
                        name=state.get("name") or "",
                        arguments=arguments,
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
                    error_message="OpenAI-compatible stream ended before [DONE]",
                )
            yield ChatStreamEvent(
                kind="completed",
                finish_reason=finish_reason,
                usage=usage,
                result=result,
            )
        finally:
            await context.__aexit__(None, None, None)

    # -------------------------------------------------------------------------
    async def list_models(
        self, *, force_refresh: bool = False
    ) -> list[CloudModelDescriptor]:
        del force_refresh

        async def get_checked() -> httpx.Response:
            response = await self.client.get("models")
            response.raise_for_status()
            return response

        response = await call_with_retries(
            get_checked,
            max_retries=getattr(self, "max_retries", 0),
        )
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Provider model catalog was not a JSON object")
        return [
            CloudModelDescriptor(
                id=str(item["id"]),
                display_name=str(item.get("name") or item["id"]),
                model_capabilities=catalog_capability_metadata(item),
            )
            for item in data.get("data", [])
            if isinstance(item, dict) and item.get("id")
        ]

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
        await self.client.aclose()
