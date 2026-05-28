from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from services.llm import cloud as cloud_module


###############################################################################
@dataclass
class FakeOpenAIResponse:
    output_text: str
    output_parsed: Any | None = None


###############################################################################
@dataclass
class FakeGeminiResponse:
    text: str


###############################################################################
class ParsedPayload(BaseModel):
    value: int


###############################################################################
class FakeGenerateContentConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


# -----------------------------------------------------------------------------
def patch_access_key(monkeypatch, key: str = "provider-key") -> None:
    monkeypatch.setattr(
        cloud_module.CloudLLMClient,
        "resolve_provider_access_key",
        lambda self, provider: key,
    )


# -----------------------------------------------------------------------------
def test_openai_chat_uses_responses_api_and_normalizes_text(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResponses:
        async def create(self, **kwargs: Any) -> FakeOpenAIResponse:
            captured.update(kwargs)
            return FakeOpenAIResponse(output_text="plain text response")

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured["client"] = kwargs
            self.responses = FakeResponses()

        async def close(self) -> None:
            captured["closed"] = True

    patch_access_key(monkeypatch, "key-openai")
    monkeypatch.setattr(cloud_module, "AsyncOpenAI", FakeAsyncOpenAI)

    client = cloud_module.CloudLLMClient(
        provider="openai",
        timeout_s=12.0,
        default_model="gpt-4.1-mini",
    )
    result = asyncio.run(
        client.chat(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "System rule."},
                {"role": "user", "content": "hello"},
            ],
            options={"temperature": 0.35, "top_p": 0.8},
        )
    )

    assert result == "plain text response"
    assert captured["client"]["api_key"] == "key-openai"
    assert captured["client"]["timeout"] == 12.0
    assert captured["model"] == "gpt-4.1-mini"
    assert captured["instructions"] == "System rule."
    assert captured["input"] == [{"role": "user", "content": "hello"}]
    assert captured["temperature"] == 0.35
    assert captured["top_p"] == 0.8


# -----------------------------------------------------------------------------
def test_openai_gpt5_chat_omits_sampling_options(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResponses:
        async def create(self, **kwargs: Any) -> FakeOpenAIResponse:
            captured.update(kwargs)
            return FakeOpenAIResponse(output_text="ok")

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.responses = FakeResponses()

        async def close(self) -> None:
            pass

    patch_access_key(monkeypatch)
    monkeypatch.setattr(cloud_module, "AsyncOpenAI", FakeAsyncOpenAI)
    client = cloud_module.CloudLLMClient(provider="openai", default_model="gpt-5.4")

    result = asyncio.run(
        client.chat(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "x"}],
            options={"temperature": 0.9, "top_p": 0.8},
        )
    )

    assert result == "ok"
    assert "temperature" not in captured
    assert "top_p" not in captured


# -----------------------------------------------------------------------------
def test_gemini_chat_uses_generate_content_and_normalizes_json(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeModels:
        def generate_content(self, **kwargs: Any) -> FakeGeminiResponse:
            captured.update(kwargs)
            return FakeGeminiResponse(text='{"value": 7}')

    class FakeGeminiClient:
        def __init__(self, **kwargs: Any) -> None:
            captured["client"] = kwargs
            self.models = FakeModels()

    class FakeGenAI:
        Client = FakeGeminiClient

    class FakeTypes:
        GenerateContentConfig = FakeGenerateContentConfig

    patch_access_key(monkeypatch, "key-gemini")
    monkeypatch.setattr(cloud_module, "genai", FakeGenAI)
    monkeypatch.setattr(cloud_module, "genai_types", FakeTypes)

    client = cloud_module.CloudLLMClient(
        provider="gemini",
        timeout_s=8.5,
        default_model="gemini-2.5-pro",
    )
    result = asyncio.run(
        client.chat(
            model="gemini-2.5-pro",
            messages=[
                {"role": "system", "content": "System rule."},
                {"role": "user", "content": "hello"},
            ],
            options={"temperature": 3.0},
        )
    )

    assert result == {"value": 7}
    assert captured["client"]["api_key"] == "key-gemini"
    assert captured["model"] == "gemini-2.5-pro"
    assert captured["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert captured["config"].kwargs["system_instruction"] == "System rule."
    assert captured["config"].kwargs["temperature"] == 2.0


# -----------------------------------------------------------------------------
def test_llm_text_call_uses_openai_responses_api(monkeypatch) -> None:
    class FakeResponses:
        async def create(self, **kwargs: Any) -> FakeOpenAIResponse:
            _ = kwargs
            return FakeOpenAIResponse(output_text="wrapped text")

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.responses = FakeResponses()

        async def close(self) -> None:
            pass

    patch_access_key(monkeypatch, "key-openai")
    monkeypatch.setattr(cloud_module, "AsyncOpenAI", FakeAsyncOpenAI)
    client = cloud_module.CloudLLMClient(
        provider="openai", default_model="gpt-4.1-mini"
    )

    text = asyncio.run(
        client.llm_text_call(
            model="gpt-4.1-mini",
            system_prompt="You are helpful.",
            user_prompt="Say hello.",
            temperature=0.2,
        )
    )
    assert text == "wrapped text"


# -----------------------------------------------------------------------------
def test_openai_structured_call_uses_responses_parse(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResponses:
        async def parse(self, **kwargs: Any) -> FakeOpenAIResponse:
            captured.update(kwargs)
            return FakeOpenAIResponse(
                output_text='{"value": 9}',
                output_parsed=ParsedPayload(value=9),
            )

        async def create(self, **kwargs: Any) -> FakeOpenAIResponse:
            raise AssertionError(
                "structured OpenAI calls should prefer responses.parse"
            )

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.responses = FakeResponses()

        async def close(self) -> None:
            pass

    patch_access_key(monkeypatch, "key-openai")
    monkeypatch.setattr(cloud_module, "AsyncOpenAI", FakeAsyncOpenAI)
    client = cloud_module.CloudLLMClient(
        provider="openai", default_model="gpt-4.1-mini"
    )

    parsed = asyncio.run(
        client.llm_structured_call(
            model="gpt-4.1-mini",
            system_prompt="Return strict JSON.",
            user_prompt="Value is 9.",
            schema=ParsedPayload,
        )
    )

    assert isinstance(parsed, ParsedPayload)
    assert parsed.value == 9
    assert captured["model"] == "gpt-4.1-mini"
    assert captured["text_format"] is ParsedPayload
    assert captured["input"] == [{"role": "user", "content": "Value is 9."}]
    assert captured["instructions"] == "Return strict JSON."


# -----------------------------------------------------------------------------
def test_gemini_structured_call_passes_response_schema(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeModels:
        def generate_content(self, **kwargs: Any) -> FakeGeminiResponse:
            captured.update(kwargs)
            return FakeGeminiResponse(text='{"value": 11}')

    class FakeGeminiClient:
        def __init__(self, **kwargs: Any) -> None:
            self.models = FakeModels()

    class FakeGenAI:
        Client = FakeGeminiClient

    class FakeTypes:
        GenerateContentConfig = FakeGenerateContentConfig

    patch_access_key(monkeypatch, "key-gemini")
    monkeypatch.setattr(cloud_module, "genai", FakeGenAI)
    monkeypatch.setattr(cloud_module, "genai_types", FakeTypes)
    client = cloud_module.CloudLLMClient(
        provider="gemini", default_model="gemini-2.5-pro"
    )

    parsed = asyncio.run(
        client.llm_structured_call(
            model="gemini-2.5-pro",
            system_prompt="Return strict JSON.",
            user_prompt="Value is 11.",
            schema=ParsedPayload,
        )
    )

    assert isinstance(parsed, ParsedPayload)
    assert parsed.value == 11
    assert captured["model"] == "gemini-2.5-pro"
    assert captured["config"].kwargs["response_mime_type"] == "application/json"
    assert captured["config"].kwargs["response_json_schema"]["title"] == "ParsedPayload"


# -----------------------------------------------------------------------------
def test_provider_exception_maps_to_existing_error_types(monkeypatch) -> None:
    class FakeResponses:
        async def create(self, **kwargs: Any) -> FakeOpenAIResponse:
            _ = kwargs
            raise TimeoutError("timed out")

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.responses = FakeResponses()

        async def close(self) -> None:
            pass

    patch_access_key(monkeypatch, "key-openai")
    monkeypatch.setattr(cloud_module, "AsyncOpenAI", FakeAsyncOpenAI)
    client = cloud_module.CloudLLMClient(
        provider="openai", default_model="gpt-4.1-mini"
    )

    try:
        asyncio.run(
            client.chat(
                model="gpt-4.1-mini", messages=[{"role": "user", "content": "x"}]
            )
        )
        assert False, "Expected timeout mapping"
    except cloud_module.LLMTimeout:
        pass
