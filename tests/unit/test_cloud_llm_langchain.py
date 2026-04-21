from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from DILIGENT.server.services.llm import cloud as cloud_module


###############################################################################
@dataclass
class FakeResponse:
    content: Any


###############################################################################
class ParsedPayload(BaseModel):
    value: int


# -----------------------------------------------------------------------------
def test_openai_chat_uses_langchain_and_normalizes_text(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        async def ainvoke(self, messages: Any) -> FakeResponse:
            captured["messages_count"] = len(messages)
            return FakeResponse(content="plain text response")

    monkeypatch.setattr(
        cloud_module.CloudLLMClient,
        "resolve_provider_access_key",
        lambda self, provider: "key-openai",
    )
    monkeypatch.setattr(cloud_module, "ChatOpenAI", FakeChatOpenAI)

    client = cloud_module.CloudLLMClient(
        provider="openai",
        timeout_s=12.0,
        default_model="gpt-4o-mini",
    )
    result = asyncio.run(
        client.chat(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            options={"temperature": 0.35, "top_p": 0.8},
        )
    )

    assert result == "plain text response"
    assert captured["model"] == "gpt-4o-mini"
    assert captured["timeout"] == 12.0
    assert captured["temperature"] == 0.35
    assert captured["top_p"] == 0.8
    assert captured["messages_count"] == 1


# -----------------------------------------------------------------------------
def test_gemini_chat_uses_langchain_and_normalizes_json(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeChatGemini:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        async def ainvoke(self, messages: Any) -> FakeResponse:
            captured["messages_count"] = len(messages)
            return FakeResponse(content='{"value": 7}')

    monkeypatch.setattr(
        cloud_module.CloudLLMClient,
        "resolve_provider_access_key",
        lambda self, provider: "key-gemini",
    )
    monkeypatch.setattr(cloud_module, "ChatGoogleGenerativeAI", FakeChatGemini)

    client = cloud_module.CloudLLMClient(
        provider="gemini",
        timeout_s=8.5,
        default_model="gemini-2.5-pro",
    )
    result = asyncio.run(
        client.chat(
            model="gemini-2.5-pro",
            messages=[{"role": "user", "content": "hello"}],
            options={"temperature": 0.6},
        )
    )

    assert result == {"value": 7}
    assert captured["model"] == "gemini-2.5-pro"
    assert captured["timeout"] == 8.5
    assert captured["temperature"] == 0.6
    assert captured["messages_count"] == 1


# -----------------------------------------------------------------------------
def test_llm_text_call_remains_compatibility_wrapper(monkeypatch) -> None:
    monkeypatch.setattr(
        cloud_module.CloudLLMClient,
        "resolve_provider_access_key",
        lambda self, provider: "key-openai",
    )

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            _ = kwargs

        async def ainvoke(self, messages: Any) -> FakeResponse:
            _ = messages
            return FakeResponse(content="wrapped text")

    monkeypatch.setattr(cloud_module, "ChatOpenAI", FakeChatOpenAI)
    client = cloud_module.CloudLLMClient(provider="openai", default_model="gpt-4o-mini")

    text = asyncio.run(
        client.llm_text_call(
            model="gpt-4o-mini",
            system_prompt="You are helpful.",
            user_prompt="Say hello.",
            temperature=0.2,
        )
    )
    assert text == "wrapped text"


# -----------------------------------------------------------------------------
def test_llm_structured_call_returns_schema_instance(monkeypatch) -> None:
    monkeypatch.setattr(
        cloud_module.CloudLLMClient,
        "resolve_provider_access_key",
        lambda self, provider: "key-openai",
    )

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            _ = kwargs

        async def ainvoke(self, messages: Any) -> FakeResponse:
            _ = messages
            return FakeResponse(content='{"value": 9}')

    monkeypatch.setattr(cloud_module, "ChatOpenAI", FakeChatOpenAI)
    client = cloud_module.CloudLLMClient(provider="openai", default_model="gpt-4o-mini")

    parsed = asyncio.run(
        client.llm_structured_call(
            model="gpt-4o-mini",
            system_prompt="Return strict JSON.",
            user_prompt="Value is 9.",
            schema=ParsedPayload,
        )
    )
    assert isinstance(parsed, ParsedPayload)
    assert parsed.value == 9


# -----------------------------------------------------------------------------
def test_provider_exception_maps_to_existing_error_types(monkeypatch) -> None:
    monkeypatch.setattr(
        cloud_module.CloudLLMClient,
        "resolve_provider_access_key",
        lambda self, provider: "key-openai",
    )

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            _ = kwargs

        async def ainvoke(self, messages: Any) -> FakeResponse:
            _ = messages
            raise TimeoutError("timed out")

    monkeypatch.setattr(cloud_module, "ChatOpenAI", FakeChatOpenAI)
    client = cloud_module.CloudLLMClient(provider="openai", default_model="gpt-4o-mini")

    try:
        asyncio.run(client.chat(model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}]))
        assert False, "Expected timeout mapping"
    except cloud_module.LLMTimeout:
        pass

