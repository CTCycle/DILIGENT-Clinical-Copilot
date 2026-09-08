from __future__ import annotations

import asyncio

import httpx
import pytest
from openai import APIStatusError
from pydantic import BaseModel
from services.llm.cloud import CloudLLMClient, LLMError, LLMTimeout


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
def test_provider_error_mapping_distinguishes_timeout() -> None:
    mapped = CloudLLMClient._map_provider_exception(httpx.ReadTimeout("read timed out"))

    assert isinstance(mapped, LLMTimeout)
    assert mapped.error_code == "timeout"
    assert mapped.retryable is True

###############################################################################
def test_provider_error_mapping_classifies_http_statuses() -> None:
    authentication = CloudLLMClient._map_provider_exception(_http_error(401))
    rate_limited = CloudLLMClient._map_provider_exception(_http_error(429))
    upstream = CloudLLMClient._map_provider_exception(_http_error(503))
    missing_endpoint = CloudLLMClient._map_provider_exception(_http_error(404))

    assert authentication.error_code == "authentication"
    assert authentication.retryable is False
    assert rate_limited.error_code == "rate_limited"
    assert rate_limited.retryable is True
    assert upstream.error_code == "upstream_error"
    assert upstream.retryable is True
    assert missing_endpoint.error_code == "configuration"
    assert missing_endpoint.retryable is False

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
    class Payload(BaseModel):
        ok: bool

    client = CloudLLMClient.__new__(CloudLLMClient)
    client.provider = "openai"
    client.default_model = None

    async def fail_native(**_: object) -> Payload:
        raise LLMError(
            "Cloud provider returned HTTP 400",
            status_code=400,
            provider="openai",
            model="gpt-5.6",
            operation="structured_output",
            provider_detail="unsupported schema",
        )

    async def fail_downgrade(**_: object) -> str:
        raise AssertionError("structured output must not downgrade to chat")

    monkeypatch.setattr(client, "_structured_openai", fail_native)
    monkeypatch.setattr(client, "chat", fail_downgrade)

    with pytest.raises(LLMError, match="Cloud provider returned HTTP 400"):
        asyncio.run(
            client.llm_structured_call(
                model="gpt-5.6",
                system_prompt="Return JSON",
                user_prompt="ok",
                schema=Payload,
            )
        )
