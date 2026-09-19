from __future__ import annotations

import asyncio

import httpx
import pytest
import services.llm.cloud as cloud_module
from openai import APIStatusError
from pydantic import BaseModel, ConfigDict
from services.llm.cloud import CloudLLMClient, LLMError, LLMTimeout
from services.llm.generation_policy import GenerationPurpose
from services.llm.transports.base import call_with_retries
from services.llm.transports.openai_chat import OpenAIChatTransport

###############################################################################
def _http_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://opencode.ai/zen/go/v1/models")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(
        "provider response", request=request, response=response
    )

###############################################################################
def test_provider_error_mapping_distinguishes_connection_failure() -> None:
    request = httpx.Request("GET", "https://opencode.ai/zen/go/v1/models")
    mapped = CloudLLMClient._map_provider_exception(
        httpx.ConnectError("All connection attempts failed", request=request)
    )

    assert isinstance(mapped, LLMError)
    assert mapped.error_code == "network_unavailable"
    assert mapped.retryable is True
    assert str(mapped) == "Cloud provider connection failed"

###############################################################################
def test_deepseek_structured_repair_handles_schema_echo_before_valid_json() -> None:

    ###############################################################################
    class Payload(BaseModel):
        model_config = ConfigDict(extra="forbid")

        ok: bool

    client = CloudLLMClient.__new__(CloudLLMClient)
    client.provider = "opencode_go"
    client.default_model = "deepseek-v4-flash"
    responses = iter(
        [
            '{"$defs":{"Payload":{"properties":{"ok":{"type":"boolean"}}}},"title":"Payload","type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}',
            "not json",
            '{"ok":true}',
        ]
    )
    calls: list[dict[str, object]] = []

    async def fake_chat(**kwargs: object) -> str:
        calls.append(kwargs)
        return next(responses)

    client.chat = fake_chat  # type: ignore[method-assign]

    result = asyncio.run(
        client.llm_structured_call(
            model="deepseek-v4-flash",
            system_prompt="Return the requested review object.",
            user_prompt="Review this synthetic revision.",
            schema=Payload,
            purpose=GenerationPurpose.REVISION_SCAN,
            max_repair_attempts=3,
        )
    )

    assert result.ok is True
    assert len(calls) == 3
    assert calls[1]["purpose"] is GenerationPurpose.JSON_REPAIR
    assert "schema or wrapper instead of the requested data" in calls[1]["messages"][1]["content"]  # type: ignore[index]
    assert "<format_instructions>" in calls[2]["messages"][1]["content"]  # type: ignore[index]
    assert calls[1]["json_schema"] == Payload.model_json_schema()

###############################################################################
def test_structured_repair_recovers_from_truncated_patient_drugs_payload() -> None:

    ###############################################################################
    class Payload(BaseModel):
        model_config = ConfigDict(extra="forbid")

        name: str

    client = CloudLLMClient.__new__(CloudLLMClient)
    client.provider = "opencode_go"
    client.default_model = "deepseek-v4-flash"
    responses = iter(
        [
            '{"name":"Drug A"',
            '{"name":"Drug A"}',
        ]
    )
    calls: list[dict[str, object]] = []

    async def fake_chat(**kwargs: object) -> str:
        calls.append(kwargs)
        return next(responses)

    client.chat = fake_chat  # type: ignore[method-assign]

    result = asyncio.run(
        client.llm_structured_call(
            model="deepseek-v4-flash",
            system_prompt="Return the requested object.",
            user_prompt="Extract the medication.",
            schema=Payload,
            purpose=GenerationPurpose.STRUCTURED_EXTRACTION,
            max_repair_attempts=1,
        )
    )

    assert result.name == "Drug A"
    assert [call["purpose"] for call in calls] == [
        GenerationPurpose.STRUCTURED_EXTRACTION,
        GenerationPurpose.JSON_REPAIR,
    ]

###############################################################################
def test_openai_chat_transport_extracts_json_from_content_parts() -> None:
    assert OpenAIChatTransport._content_to_text(
        [{"type": "text", "text": '{"ok":true}'}, {"type": "text", "text": "\n"}]
    ) == '{"ok":true}\n'

###############################################################################
def test_provider_error_mapping_distinguishes_timeout() -> None:
    mapped = CloudLLMClient._map_provider_exception(httpx.ReadTimeout("read timed out"))

    assert isinstance(mapped, LLMTimeout)
    assert mapped.error_code == "timeout"
    assert mapped.retryable is True

###############################################################################
def test_gemini_client_receives_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    ###############################################################################
    class FakeGeminiClient:

        # -------------------------------------------------------------------------
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        # -------------------------------------------------------------------------
        async def close(self) -> None:
            return None

    monkeypatch.setattr(cloud_module, "GeminiTransport", FakeGeminiClient)
    monkeypatch.setattr(
        CloudLLMClient,
        "resolve_provider_access_key",
        lambda _self, _provider: "gemini-test-key",
    )

    client = CloudLLMClient(provider="gemini", timeout_s=12.5)
    try:
        assert captured["timeout"] == 12.5
    finally:
        asyncio.run(client.close())

###############################################################################
def test_provider_error_mapping_classifies_http_statuses() -> None:
    authentication = CloudLLMClient._map_provider_exception(_http_error(401))
    rate_limited = CloudLLMClient._map_provider_exception(_http_error(429))
    upstream = CloudLLMClient._map_provider_exception(_http_error(503))
    gateway_timeout = CloudLLMClient._map_provider_exception(_http_error(530))
    missing_endpoint = CloudLLMClient._map_provider_exception(_http_error(404))

    assert authentication.error_code == "authentication"
    assert authentication.retryable is False
    assert rate_limited.error_code == "rate_limited"
    assert rate_limited.retryable is True
    assert upstream.error_code == "upstream_error"
    assert upstream.retryable is True
    assert gateway_timeout.error_code == "upstream_error"
    assert gateway_timeout.retryable is True
    assert gateway_timeout.status_code == 530
    assert missing_endpoint.error_code == "configuration"
    assert missing_endpoint.retryable is False

###############################################################################
def test_revision_retry_policy_retries_http_503_and_530_before_success() -> None:
    attempts = 0

    async def flaky_provider() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _http_error(503)
        if attempts == 2:
            raise _http_error(530)
        return "ok"

    assert asyncio.run(call_with_retries(flaky_provider, max_retries=2)) == "ok"
    assert attempts == 3

###############################################################################
def test_provider_error_mapping_classifies_openai_sdk_status_errors() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(429, request=request)
    mapped = CloudLLMClient._map_provider_exception(
        APIStatusError("provider response", response=response, body=None)
    )

    assert mapped.error_code == "rate_limited"
    assert mapped.retryable is True

###############################################################################
def test_provider_error_mapping_preserves_sanitized_contract_detail() -> None:
    request = httpx.Request("POST", "https://opencode.ai/zen/go/v1/chat/completions")
    response = httpx.Response(
        400,
        request=request,
        json={
            "error": {
                "message": "Model is unavailable; api_key=should-not-leak",
            }
        },
        headers={"x-request-id": "req-opencode-123"},
    )
    mapped = CloudLLMClient._map_provider_exception(
        httpx.HTTPStatusError("provider response", request=request, response=response),
        provider="opencode_go",
        model="deepseek-v4-flash",
        operation="structured_output",
    )

    assert mapped.status_code == 400
    assert mapped.request_id == "req-opencode-123"
    assert mapped.provider_detail == "Model is unavailable; api_key=<redacted>"
    assert mapped.user_message() == (
        "Check the provider connection, credentials, rate limits, or transient service status. "
        "Detail: opencode-go rejected deepseek-v4-flash during structured_output "
        "(HTTP 400): Model is unavailable; api_key=<redacted>"
    )

    quoted = LLMError(
        "provider failure",
        provider_detail='{"authorization":"Bearer secret-value"}',
    )
    assert quoted.provider_detail == '{"authorization":<redacted>}'
    assert (
        LLMError(
            "provider failure",
            provider_detail="authorization=Bearer secret-value",
        ).provider_detail
        == "authorization=<redacted>"
    )

###############################################################################
def test_openai_structured_provider_failure_is_not_downgraded_to_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    ###############################################################################
    class Payload(BaseModel):
        ok: bool

    client = CloudLLMClient.__new__(CloudLLMClient)
    client.provider = "openai"
    client.default_model = None

    async def fail_chat(**_: object) -> str:
        raise LLMError(
            "Cloud provider returned HTTP 400",
            status_code=400,
            provider="openai",
            model="gpt-5.6",
            operation="structured_output",
            provider_detail="unsupported schema",
        )

    monkeypatch.setattr(client, "chat", fail_chat)

    with pytest.raises(LLMError, match="Cloud provider returned HTTP 400"):
        asyncio.run(
            client.llm_structured_call(
                model="gpt-5.6",
                system_prompt="Return JSON",
                user_prompt="ok",
                schema=Payload,
            )
        )
