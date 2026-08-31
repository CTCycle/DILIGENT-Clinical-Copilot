from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from domain.llm.transports import ChatRequest
from services.llm.transports.anthropic_messages import AnthropicMessagesTransport
from services.llm.transports.gemini import GeminiTransport
from services.llm.transports.openai_chat import OpenAIChatTransport
from services.llm.transports.openai_responses import OpenAIResponsesTransport


###############################################################################
class FakeOpenAIChatResponse:
    # -------------------------------------------------------------------------
    def raise_for_status(self) -> None:
        return None

    # -------------------------------------------------------------------------
    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "ok"}}]}


###############################################################################
def test_openai_chat_transport_normalizes_reasoning_and_output_options() -> None:
    captured: dict[str, Any] = {}

    ###############################################################################
    class FakeClient:
        # -------------------------------------------------------------------------
        async def post(
            self, path: str, *, json: dict[str, Any]
        ) -> FakeOpenAIChatResponse:
            captured["path"] = path
            captured["json"] = json
            return FakeOpenAIChatResponse()

    transport = OpenAIChatTransport.__new__(OpenAIChatTransport)
    transport.client = FakeClient()
    result = asyncio.run(
        transport.chat(
            ChatRequest(
                model="deepseek-v4",
                messages=[{"role": "user", "content": "hello"}],
                options={
                    "temperature": 0.2,
                    "max_output_tokens": 999,
                    "custom": "kept",
                },
                reasoning_level="high",
                reasoning_parameter="boolean",
                output_token_limit=128,
            )
        )
    )

    assert result.content == "ok"
    assert captured["path"] == "chat/completions"
    payload = captured["json"]
    assert payload["custom"] == "kept"
    assert payload["max_tokens"] == 128
    assert payload["stream"] is False
    assert payload["thinking"] == {"type": "enabled"}
    assert "temperature" not in payload
    assert "max_output_tokens" not in payload


###############################################################################
def test_openai_responses_transport_preserves_options_and_normalizes_limits() -> None:
    captured: dict[str, Any] = {}

    ###############################################################################
    class FakeResponses:
        # -------------------------------------------------------------------------
        async def create(self, **kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(output_text="ok")

    transport = OpenAIResponsesTransport.__new__(OpenAIResponsesTransport)
    transport.client = SimpleNamespace(responses=FakeResponses())
    result = asyncio.run(
        transport.chat(
            ChatRequest(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "hello"},
                ],
                options={
                    "temperature": 0.2,
                    "max_output_tokens": 999,
                    "custom": "kept",
                },
                reasoning_level="medium",
                reasoning_parameter="effort",
                output_token_limit=256,
            )
        )
    )

    assert result.content == "ok"
    assert captured["custom"] == "kept"
    assert captured["max_output_tokens"] == 256
    assert captured["reasoning"] == {"effort": "medium"}
    assert "temperature" not in captured


###############################################################################
def test_anthropic_transport_reserves_reasoning_budget_without_fixed_default() -> None:
    captured: dict[str, Any] = {}

    ###############################################################################
    class FakeMessages:
        # -------------------------------------------------------------------------
        async def create(self, **kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text="ok")])

    transport = AnthropicMessagesTransport.__new__(AnthropicMessagesTransport)
    transport.client = SimpleNamespace(messages=FakeMessages())
    result = asyncio.run(
        transport.chat(
            ChatRequest(
                model="claude-sonnet",
                messages=[{"role": "user", "content": "hello"}],
                options={"temperature": 0.2, "max_tokens": 999},
                reasoning_level="low",
                reasoning_parameter="budget_tokens",
                reasoning_reserve=256,
                output_token_limit=512,
            )
        )
    )

    assert result.content == "ok"
    assert captured["max_tokens"] == 1536
    assert captured["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert "temperature" not in captured


###############################################################################
def test_gemini_maps_medium_reasoning_to_low_sdk_level() -> None:
    request = ChatRequest(
        model="gemini-3",
        messages=[{"role": "user", "content": "hello"}],
        reasoning_level="medium",
        reasoning_parameter="level",
    )

    config = GeminiTransport._thinking_config(request)

    assert config is not None
    assert str(config.thinking_level).lower().endswith("low")
