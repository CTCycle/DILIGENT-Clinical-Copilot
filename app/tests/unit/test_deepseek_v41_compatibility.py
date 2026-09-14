from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from domain.llm.providers import CloudModelDescriptor, ModelCapabilityMetadata
from domain.llm.transports import (
    ChatMessage,
    ChatRequest,
    ChatResult,
    ToolCall,
    ToolDefinition,
)
from services.llm.generation_policy import GenerationPurpose
from services.llm.model_capabilities import (
    capability_metadata,
    resolve_endpoint_family,
    resolve_model_capabilities,
)
from services.llm.tool_loop import ToolLoopExecutor
from services.llm.transports.anthropic_messages import AnthropicMessagesTransport
from services.llm.transports.base import validate_request_capabilities
from services.llm.transports.errors import TransportUnsupportedCapability
from services.llm.transports.gemini import GeminiTransport
from services.llm.transports.openai_chat import OpenAIChatTransport
from services.llm.transports.openai_responses import OpenAIResponsesTransport


###############################################################################
def _deepseek_request(**overrides: Any) -> ChatRequest:
    values: dict[str, Any] = {
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "hello"}],
        "capabilities": capability_metadata(
            provider="deepseek", model="deepseek-flash"
        ),
        "output_token_limit": 128,
        "reasoning_reserve": 64,
        "reasoning_level": "high",
        "reasoning_parameter": "effort",
    }
    values.update(overrides)
    return ChatRequest(**values)

###############################################################################
def test_deepseek_v41_flash_has_one_explicit_contract_per_exposing_provider() -> None:
    for provider, model in (
        ("deepseek", "deepseek-flash"),
        ("deepseek", "deepseek-v4-flash"),
        ("deepseek", "deepseek-v4.1-flash"),
        ("opencode_go", "deepseek-v4.1-flash"),
        ("opencode_go", "deepseek-flash"),
    ):
        capabilities = resolve_model_capabilities(provider=provider, model=model)

        assert capabilities.semantic_model_id == "deepseek-v4.1-flash"
        assert capabilities.endpoint_family == "chat/completions"
        assert capabilities.input_token_limit == 1_000_000
        assert capabilities.output_token_limit == 384_000
        assert capabilities.supports_chat is True
        assert capabilities.supports_streaming is True
        assert capabilities.supports_tools is True
        assert capabilities.tool_call_mode == "native"
        assert capabilities.supports_tool_choice is False
        assert capabilities.reasoning_toggle_parameter == "thinking"
        assert capabilities.requires_reasoning_content_for_tool_calls is True
        assert capabilities.requires_assistant_content_for_tool_calls is True

    assert resolve_endpoint_family(
        provider="opencode_zen", model="deepseek-v4.1-flash"
    ) is None

###############################################################################
def test_live_catalog_metadata_does_not_erase_exact_deepseek_contract_fields() -> None:
    descriptor = CloudModelDescriptor(
        id="deepseek-v4.1-flash",
        display_name="DeepSeek V4.1 Flash",
        endpoint_family="chat/completions",
        model_capabilities=ModelCapabilityMetadata(
            endpoint_family="chat/completions",
            supports_chat=True,
            supports_streaming=True,
            supports_tools=True,
            supports_reasoning=True,
            supports_temperature=True,
            supports_top_p=True,
            supports_json_mode=True,
            supports_usage_metadata=True,
            supports_finish_reason=True,
            evidence="catalog",
        ),
    )

    capabilities = resolve_model_capabilities(
        provider="opencode_go",
        model=descriptor.id,
        descriptor=descriptor,
    )

    assert capabilities.reasoning_parameter == "effort"
    assert capabilities.reasoning_toggle_parameter == "thinking"
    assert capabilities.tool_call_mode == "native"
    assert capabilities.reasoning_unsupported_parameters == (
        "temperature",
        "top_p",
        "tool_choice",
    )

###############################################################################
def test_deepseek_reasoning_mode_is_explicit_and_rejects_unsupported_parameters() -> None:
    payload = OpenAIChatTransport._request_payload(
        _deepseek_request(), stream=False
    )

    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"
    assert payload["max_tokens"] == 192
    assert "temperature" not in payload
    assert "top_p" not in payload

    with pytest.raises(TransportUnsupportedCapability, match="top_p"):
        OpenAIChatTransport._request_payload(
            _deepseek_request(top_p=0.9), stream=False
        )
    with pytest.raises(TransportUnsupportedCapability, match="tool_choice"):
        validate_request_capabilities(_deepseek_request(tool_choice="auto"))

###############################################################################
def test_deepseek_non_thinking_request_disables_provider_default_thinking() -> None:
    payload = OpenAIChatTransport._request_payload(
        _deepseek_request(
            reasoning_level="off",
            temperature=0.2,
            reasoning_reserve=0,
        ),
        stream=False,
    )

    assert payload["thinking"] == {"type": "disabled"}
    assert payload["temperature"] == 0.2
    assert "reasoning_effort" not in payload

###############################################################################
def test_openai_chat_parses_deepseek_tool_calls_usage_and_finish_reason() -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "id": "req-1",
                "model": "deepseek-v4.1-flash",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "selecting a lookup",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"term":"x"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "completion_tokens_details": {"reasoning_tokens": 12},
                },
            }

    class FakeClient:

        async def post(self, path: str, *, json: dict[str, Any]) -> FakeResponse:
            captured["path"] = path
            captured["payload"] = json
            return FakeResponse()

    transport = OpenAIChatTransport.__new__(OpenAIChatTransport)
    transport.client = FakeClient()
    transport.max_retries = 0
    result = asyncio.run(
        transport.chat(
            _deepseek_request(
                tools=[ToolDefinition(name="lookup", parameters={"type": "object"})]
            )
        )
    )

    assert captured["path"] == "chat/completions"
    assert result.content == ""
    assert result.reasoning_content == "selecting a lookup"
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls == [
        ToolCall(
            id="call-1",
            name="lookup",
            arguments={"term": "x"},
            raw_arguments='{"term":"x"}',
        )
    ]
    assert result.usage is not None
    assert result.usage.reasoning_tokens == 12
    assert captured["payload"]["tools"][0]["function"]["name"] == "lookup"
    assert "tool_choice" not in captured["payload"]

###############################################################################
def test_openai_chat_reinjects_reasoning_and_empty_assistant_content_for_tools() -> None:
    request = _deepseek_request(
        messages=[
            {"role": "user", "content": "lookup x"},
            ChatMessage(
                role="assistant",
                content="",
                reasoning_content="I need a lookup.",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="lookup",
                        arguments={"term": "x"},
                        raw_arguments='{"term":"x"}',
                    )
                ],
            ),
            ChatMessage(
                role="tool",
                name="lookup",
                tool_call_id="call-1",
                content='{"value":42}',
            ),
        ],
        tools=[ToolDefinition(name="lookup", parameters={"type": "object"})],
        reasoning_level="high",
    )
    payload = OpenAIChatTransport._request_payload(request, stream=False)

    assistant = payload["messages"][1]
    assert assistant["content"] == ""
    assert assistant["reasoning_content"] == "I need a lookup."
    assert assistant["tool_calls"][0]["id"] == "call-1"
    assert payload["messages"][2] == {
        "role": "tool",
        "content": '{"value":42}',
        "name": "lookup",
        "tool_call_id": "call-1",
    }

###############################################################################
def test_openai_chat_stream_aggregates_reasoning_tool_deltas_and_usage() -> None:
    class FakeResponse:

        headers: ClassVar[dict[str, str]] = {"x-request-id": "req-stream"}

        async def aiter_lines(self) -> AsyncIterator[str]:
            for line in (
                'data: {"model":"deepseek-v4.1-flash","choices":[{"delta":{"reasoning_content":"think"},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"lookup","arguments":"{\\"term\\":\\"x\\"}"}}]},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":8,"total_tokens":12}}',
                "data: [DONE]",
            ):
                yield line

        def raise_for_status(self) -> None:
            return None

    class FakeStreamContext:

        async def __aenter__(self) -> FakeResponse:
            return FakeResponse()

        async def __aexit__(self, *_: object) -> None:
            return None

    class FakeClient:

        def stream(self, *args: Any, **kwargs: Any) -> FakeStreamContext:
            return FakeStreamContext()

    transport = OpenAIChatTransport.__new__(OpenAIChatTransport)
    transport.client = FakeClient()
    transport.max_retries = 0
    events = asyncio.run(_collect(transport.stream(_deepseek_request())))

    completed = next(event for event in events if event.kind == "completed")
    assert completed.result is not None
    assert completed.result.partial is False
    assert completed.result.reasoning_content == "think"
    assert completed.result.finish_reason == "tool_calls"
    assert completed.result.tool_calls[0].id == "call-1"
    assert completed.result.usage is not None
    assert completed.result.usage.total_tokens == 12

###############################################################################
def test_openai_responses_and_anthropic_adapters_normalize_tool_results() -> None:
    response_items = OpenAIResponsesTransport._message_items(
        ChatMessage(
            role="tool",
            tool_call_id="call-1",
            content='{"value":42}',
        )
    )
    assert response_items == [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '{"value":42}',
        }
    ]
    response_result = OpenAIResponsesTransport._result_from_response(
        SimpleNamespace(
            output_text="done",
            output=[
                SimpleNamespace(
                    type="function_call",
                    call_id="call-1",
                    name="lookup",
                    arguments='{"term":"x"}',
                )
            ],
            status="completed",
            id="resp-1",
            usage=SimpleNamespace(input_tokens=4, output_tokens=8, total_tokens=12),
        )
    )
    assert response_result.finish_reason == "stop"
    assert response_result.tool_calls[0].name == "lookup"

    anthropic_payload = AnthropicMessagesTransport._message_payload(
        ChatMessage(role="tool", tool_call_id="call-1", content="42", name="lookup")
    )
    assert anthropic_payload == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "call-1", "content": "42"}
        ],
    }
    anthropic_result = AnthropicMessagesTransport._result_from_response(
        SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="done"),
                SimpleNamespace(
                    type="tool_use", id="call-1", name="lookup", input={"term": "x"}
                ),
            ],
            stop_reason="tool_use",
            id="msg-1",
            model="claude",
            usage=SimpleNamespace(input_tokens=4, output_tokens=8),
        )
    )
    assert anthropic_result.finish_reason == "tool_calls"
    assert anthropic_result.tool_calls[0].id == "call-1"

###############################################################################
def test_gemini_adapter_maps_tool_result_and_normalizes_finish_reason() -> None:
    _, contents = GeminiTransport._contents(
        [
            ChatMessage(role="user", content="lookup x"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(id="call-1", name="lookup", arguments={"term": "x"})
                ],
            ),
            ChatMessage(
                role="tool",
                name="lookup",
                tool_call_id="call-1",
                content="42",
            ),
        ]
    )

    assert contents[1].parts[0].function_call.name == "lookup"
    assert contents[2].parts[0].function_response.name == "lookup"
    result = GeminiTransport._result(
        SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[SimpleNamespace(text="done")]),
                    finish_reason="STOP",
                )
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=4,
                candidates_token_count=8,
                total_token_count=12,
            ),
            model_version="gemini",
            response_id="gem-1",
        )
    )
    assert result.finish_reason == "stop"
    assert result.usage is not None
    assert result.usage.total_tokens == 12

###############################################################################
def test_tool_loop_reinjects_multiple_tool_results_and_continues() -> None:
    calls: list[dict[str, Any]] = []
    responses = iter(
        [
            ChatResult(
                content="",
                reasoning_content="plan",
                tool_calls=[
                    ToolCall(id="call-1", name="one", arguments={"value": 1}),
                    ToolCall(id="call-2", name="two", arguments={"value": 2}),
                ],
                finish_reason="tool_calls",
            ),
            ChatResult(content="final", finish_reason="stop"),
        ]
    )

    async def chat(**kwargs: Any) -> ChatResult:
        calls.append(kwargs)
        return next(responses)

    def execute(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"tool": name, "value": arguments["value"] * 10}

    result = asyncio.run(
        ToolLoopExecutor(
            chat=chat,
            execute=execute,
            max_iterations=3,
            max_tool_calls=4,
        ).run(
            model="deepseek-flash",
            messages=[ChatMessage(role="user", content="run both")],
            tools=[
                ToolDefinition(name="one", parameters={"type": "object"}),
                ToolDefinition(name="two", parameters={"type": "object"}),
            ],
            purpose=GenerationPurpose.REVISION_TOOL_SELECTION,
        )
    )

    assert result.result.content == "final"
    assert result.iterations == 2
    assert result.tool_call_count == 2
    assert [entry["tool"] for entry in result.observations] == ["one", "two"]
    assert calls[0]["tool_choice"] is None
    assert result.messages[1].content == ""
    assert result.messages[1].reasoning_content == "plan"
    assert [message.tool_call_id for message in result.messages[2:] if message.role == "tool"] == [
        "call-1",
        "call-2",
    ]

###############################################################################
async def _collect(iterator: AsyncIterator[Any]) -> list[Any]:
    return [event async for event in iterator]
