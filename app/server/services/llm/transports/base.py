from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
import sys
from typing import Any, Protocol

import httpx

from domain.llm.providers import CloudModelDescriptor
from domain.llm.transports import (
    T,
    ChatRequest,
    ChatResult,
    ChatStreamEvent,
    ConnectivityResult,
    EmbeddingRequest,
    StructuredRequest,
)
from services.llm.transports.errors import TransportCancellation, TransportUnsupportedCapability

###############################################################################
def normalize_finish_reason(value: object) -> str | None:
    """Normalize provider finish/status enums into the shared result contract."""

    if value is None:
        return None
    raw = getattr(value, "value", value)
    normalized = str(raw or "").strip().casefold()
    if not normalized:
        return None
    aliases = {
        "stop": "stop",
        "end_turn": "stop",
        "end-turn": "stop",
        "completed": "stop",
        "complete": "stop",
        "done": "stop",
        "length": "length",
        "max_tokens": "length",
        "max-token": "length",
        "incomplete": "length",
        "tool_call": "tool_calls",
        "tool_calls": "tool_calls",
        "tool_use": "tool_calls",
        "function_call": "tool_calls",
        "function_calls": "tool_calls",
        "content_filter": "content_filter",
        "safety": "content_filter",
        "recitation": "content_filter",
        "malformed_function_call": "error",
        "failed": "error",
        "error": "error",
    }
    return aliases.get(normalized, normalized)

###############################################################################
def validate_request_capabilities(request: ChatRequest) -> None:
    """Reject unsupported features before a provider can receive bad input."""

    capabilities = request.capabilities
    if capabilities is None:
        return
    if capabilities.supports_chat is False:
        raise TransportUnsupportedCapability(
            f"Model '{request.model}' does not support chat requests."
        )
    checks = (
        (request.stream, capabilities.supports_streaming, "streaming"),
        (bool(request.tools), capabilities.supports_tools, "tool calling"),
        (request.json_mode, capabilities.supports_json_mode, "JSON mode"),
        (request.temperature is not None, capabilities.supports_temperature, "temperature"),
        (request.top_p is not None, capabilities.supports_top_p, "top_p"),
        (
            request.reasoning_level not in {None, "", "off"},
            capabilities.supports_reasoning,
            "reasoning",
        ),
    )
    for requested, supported, feature in checks:
        if requested and supported is False:
            raise TransportUnsupportedCapability(
                f"Model '{request.model}' does not support {feature}."
            )
    if request.tools and capabilities.tool_call_mode == "unsupported":
        raise TransportUnsupportedCapability(
            f"Model '{request.model}' does not expose native or structured tool calling."
        )
    if request.tool_choice is not None and capabilities.supports_tool_choice is False:
        raise TransportUnsupportedCapability(
            f"Model '{request.model}' does not support tool_choice."
        )
    if request.reasoning_level not in {None, "", "off"}:
        unsupported = set(capabilities.reasoning_unsupported_parameters)
        requested_parameters = {
            key
            for key, value in (
                ("temperature", request.temperature),
                ("top_p", request.top_p),
                ("temperature", request.options.get("temperature")),
                ("top_p", request.options.get("top_p")),
            )
            if value is not None
        }
        invalid = sorted(requested_parameters & unsupported)
        if invalid:
            raise TransportUnsupportedCapability(
                f"Model '{request.model}' does not support {', '.join(invalid)} "
                "while reasoning is enabled."
            )
    if request.tool_choice is not None and not request.tools:
        raise TransportUnsupportedCapability(
            "tool_choice requires at least one tool definition."
        )

###############################################################################
def request_output_token_limit(request: ChatRequest) -> int | None:
    """Return the provider limit needed for visible output plus reasoning."""

    limit = request.output_token_limit
    capabilities = request.capabilities
    if (
        limit is not None
        and request.reasoning_level not in {None, "", "off"}
        and request.reasoning_reserve
        and capabilities is not None
        and capabilities.reasoning_counts_toward_output_limit
    ):
        return limit + request.reasoning_reserve
    return limit

###############################################################################
def provider_exception_is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, TransportCancellation):
        return False
    if isinstance(exc, (TimeoutError, httpx.TimeoutException, httpx.NetworkError)):
        return True
    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int):
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code in {408, 409, 425, 429} or status_code >= 500
    name = exc.__class__.__name__.casefold()
    return any(
        marker in name
        for marker in (
            "timeout",
            "connection",
            "network",
            "rate_limit",
            "ratelimit",
            "serviceunavailable",
            "internalserver",
        )
    )

###############################################################################
async def call_with_retries(
    operation: Callable[[], Awaitable[Any]],
    *,
    max_retries: int,
    cancel_check: Callable[[], bool] | None = None,
) -> Any:
    """Retry only transient provider failures, never malformed requests."""

    attempts = max(0, int(max_retries)) + 1
    for attempt in range(attempts):
        if cancel_check is not None and cancel_check():
            raise TransportCancellation("LLM request cancelled")
        try:
            return await operation()
        except TransportCancellation:
            raise
        except Exception as exc:  # noqa: BLE001
            if attempt + 1 >= attempts or not provider_exception_is_retryable(exc):
                raise
            if cancel_check is not None and cancel_check():
                raise TransportCancellation("LLM request cancelled") from exc
            await asyncio.sleep(min(2.0, 0.25 * (2**attempt)))
    raise RuntimeError("Provider operation did not produce a result")

###############################################################################
async def enter_async_context_with_retries(
    context_factory: Callable[[], Any],
    *,
    max_retries: int,
    cancel_check: Callable[[], bool] | None = None,
    validate: Callable[[Any], None] | None = None,
) -> tuple[Any, Any]:
    """Enter an SDK/HTTP streaming context with the same retry policy as chat."""

    async def enter() -> tuple[Any, Any]:
        context = context_factory()
        try:
            value = await context.__aenter__()
            if validate is not None:
                validate(value)
            return context, value
        except Exception:
            exit_error = sys.exc_info()
            try:
                await context.__aexit__(*exit_error)
            except Exception:
                pass
            raise

    return await call_with_retries(
        enter,
        max_retries=max_retries,
        cancel_check=cancel_check,
    )

###############################################################################
class CloudTransport(Protocol):

    # -------------------------------------------------------------------------
    async def chat(self, request: ChatRequest) -> ChatResult: ...

    # -------------------------------------------------------------------------
    def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]: ...

    # -------------------------------------------------------------------------
    async def structured(self, request: StructuredRequest[T]) -> T: ...

    # -------------------------------------------------------------------------
    async def list_models(
        self, *, force_refresh: bool = False
    ) -> list[CloudModelDescriptor]: ...

    # -------------------------------------------------------------------------
    async def check_connectivity(self, model: str) -> ConnectivityResult: ...

    # -------------------------------------------------------------------------
    async def close(self) -> None: ...

###############################################################################
class EmbeddingTransport(Protocol):

    # -------------------------------------------------------------------------
    async def embed(self, request: EmbeddingRequest) -> list[list[float]]: ...

###############################################################################
class StructuredTransportMixin:

    # -------------------------------------------------------------------------
    async def chat(self, request: ChatRequest) -> ChatResult:
        raise NotImplementedError

    # -------------------------------------------------------------------------
    async def _fallback_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        result = await self.chat(request.model_copy(update={"stream": False}))
        if result.reasoning_content:
            yield ChatStreamEvent(kind="reasoning_delta", reasoning=result.reasoning_content)
        if result.content:
            yield ChatStreamEvent(kind="text_delta", text=result.content)
        yield ChatStreamEvent(
            kind="completed",
            finish_reason=result.finish_reason,
            usage=result.usage,
            result=result,
        )

    # -------------------------------------------------------------------------
    def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        return self._fallback_stream(request)

    # -------------------------------------------------------------------------
    async def structured(self, request: StructuredRequest[T]) -> T:
        result = await self.chat(
            ChatRequest(
                model=request.model,
                messages=request.messages,
                options=request.options,
                json_mode=True,
                operation="structured_output",
                json_schema=request.schema_type.model_json_schema(),
                reasoning_level=request.reasoning_level,
                reasoning_parameter=request.reasoning_parameter,
                reasoning_reserve=request.reasoning_reserve,
                output_token_limit=request.output_token_limit,
            )
        )
        return request.schema_type.model_validate_json(result.content)
