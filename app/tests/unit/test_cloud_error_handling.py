from __future__ import annotations

import httpx
from openai import APIStatusError

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
