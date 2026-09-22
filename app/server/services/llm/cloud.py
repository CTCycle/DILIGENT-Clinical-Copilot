from __future__ import annotations

import hashlib
import asyncio
import json
import re
from typing import Any

import httpx
from google.genai import errors as genai_errors
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAIError,
)

from common.constants import GEMINI_API_BASE, OPENAI_API_BASE
from common.prompts.structured_output import (
    COMPACT_JSON_REPAIR_SYSTEM_PROMPT,
    build_compact_json_repair_user_prompt,
    build_json_repair_user_prompt,
)
from common.utils.logger import logger
from services.llm.runtime_config import LLMRuntimeConfig
from services.llm.generation_policy import GenerationPurpose
from configurations.startup import get_server_settings
from repositories.serialization.access_keys import AccessKeySerializer
from services.llm.structured import (
    StructuredOutputParser,
    T,
    parse_json_object_strict,
)
from domain.llm.providers import CloudModelDescriptor, CloudProviderId
from domain.llm.transports import (
    ChatMessage,
    ChatRequest,
    ChatResult,
    ChatStreamEvent,
    RequestOperation,
    ToolDefinition,
)
from services.llm.provider_registry import provider_registry
from services.llm.transports.anthropic_messages import AnthropicMessagesTransport
from services.llm.transports.base import CloudTransport
from services.llm.transports.gemini import GeminiTransport
from services.llm.transports.openai_chat import OpenAIChatTransport
from services.llm.transports.openai_responses import OpenAIResponsesTransport
from services.llm.transports.routed_gateway import RoutedGatewayTransport
from services.llm.transports.errors import (
    TransportCancellation,
    TransportPartialResponse,
    TransportUnsupportedCapability,
)
from services.llm.model_capabilities import (
    capability_metadata,
    enrich_model_descriptor,
)

ProviderName = CloudProviderId
_PROVIDER_FAILURE_HINT = (
    "Check the provider connection, credentials, rate limits, or transient service status."
)
_MAX_STRUCTURED_REPAIR_TEXT_CHARS = 30000

###############################################################################
class LLMError(RuntimeError):

    # -------------------------------------------------------------------------
    def __init__(
        self,
        message: str,
        *,
        error_code: str = "provider_error",
        retryable: bool = False,
        provider: str | None = None,
        model: str | None = None,
        operation: str | None = None,
        status_code: int | None = None,
        request_id: str | None = None,
        provider_detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.retryable = bool(retryable)
        self.provider = provider
        self.model = model
        self.operation = operation
        self.status_code = status_code
        self.request_id = request_id
        self.provider_detail = self._sanitize_provider_detail(provider_detail)

    # -------------------------------------------------------------------------
    @staticmethod
    def _sanitize_provider_detail(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, (dict, list, tuple)):
            try:
                value = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
            except (TypeError, ValueError):
                value = str(value)
        normalized = " ".join(str(value).split()).strip()
        normalized = re.sub(
            r"(?i)([\"']?(?:api[-_ ]?key|authorization|bearer|token|secret|password)[\"']?\s*[:=]\s*)(?:bearer\s+\S+|\"[^\"]*\"|'[^']*'|[^,;\s}]+)",
            r"\1<redacted>",
            normalized,
        )
        normalized = re.sub(r"(?i)\bbearer\s+\S+", "Bearer <redacted>", normalized)
        return normalized[:320] or None

    # -------------------------------------------------------------------------
    def with_context(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        operation: str | None = None,
    ) -> "LLMError":
        if self.provider is None:
            self.provider = provider
        if self.model is None:
            self.model = model
        if self.operation is None:
            self.operation = operation
        return self

    # -------------------------------------------------------------------------
    def user_message(self) -> str:
        if not self.provider or not self.model or not self.operation:
            return str(self)
        provider_label = self.provider.replace("_", "-")
        status = f" (HTTP {self.status_code})" if self.status_code else ""
        detail = f": {self.provider_detail}" if self.provider_detail else ""
        verb = "rejected" if self.status_code else "failed"
        return (
            f"{_PROVIDER_FAILURE_HINT} Detail: {provider_label} {verb} "
            f"{self.model} during {self.operation}{status}{detail}"
        )

###############################################################################
class LLMTimeout(LLMError):
    """Raised when requests exceed the configured timeout."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        message: str = "Timed out waiting for cloud chat response",
        *,
        error_code: str = "timeout",
        retryable: bool = True,
        provider: str | None = None,
        model: str | None = None,
        operation: str | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code,
            retryable=retryable,
            provider=provider,
            model=model,
            operation=operation,
        )

###############################################################################
def short_output_hash(output_text: str) -> str:
    return hashlib.sha256((output_text or "").encode("utf-8")).hexdigest()[:12]

###############################################################################
def clip_structured_repair_text(value: object) -> str:
    text = str(value or "")
    if len(text) <= _MAX_STRUCTURED_REPAIR_TEXT_CHARS:
        return text
    head_length = _MAX_STRUCTURED_REPAIR_TEXT_CHARS // 2
    tail_length = _MAX_STRUCTURED_REPAIR_TEXT_CHARS - head_length
    omitted = len(text) - head_length - tail_length
    return (
        f"{text[:head_length]}\n\n"
        f"[TRUNCATED: {omitted} characters omitted]\n\n"
        f"{text[-tail_length:]}"
    )

###############################################################################
def looks_like_schema_echo(text: str) -> bool:
    lowered = text.casefold()
    schema_markers = (
        '"$defs"',
        '"properties"',
        '"required"',
        '"title"',
        '"type"',
        '"$ref"',
    )
    return sum(1 for marker in schema_markers if marker in lowered) >= 3

###############################################################################
class CloudLLMClient:
    """
    Async client for hosted/proprietary LLMs (OpenAI, Gemini, etc.) that follows
    the app's shared LLM call shape.

    """

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        provider: ProviderName = "openai",
        base_url: str | None = None,
        timeout_s: float | None = None,
        keepalive_connections: int = 10,
        keepalive_max: int = 20,
        default_model: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self.provider: ProviderName = provider
        self.default_model = default_model
        runtime_timeout = get_server_settings().runtime.default_llm_timeout
        self.timeout_s = float(runtime_timeout if timeout_s is None else timeout_s)
        provider_access_key = self.resolve_provider_access_key(provider)
        self.provider_access_key = provider_access_key
        self.transport: CloudTransport | None = None

        if provider == "openai":
            if not provider_access_key:
                raise LLMError("No active OpenAI access key configured")
            self.base_url = (base_url or OPENAI_API_BASE).rstrip("/")
            headers = {
                "Authorization": f"Bearer {provider_access_key}",
                "Content-Type": "application/json",
            }
            self.transport = OpenAIResponsesTransport(
                api_key=provider_access_key,
                base_url=self.base_url,
                timeout=self.timeout_s,
                default_headers=headers,
                max_retries=max_retries,
            )
        elif provider == "gemini":
            if not provider_access_key:
                raise LLMError("No active Gemini access key configured")
            self.base_url = (base_url or GEMINI_API_BASE).rstrip("/")
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": provider_access_key,
            }
            self.transport = GeminiTransport(
                api_key=provider_access_key,
                timeout=self.timeout_s,
                max_retries=max_retries,
            )
        elif provider == "deepseek":
            if not provider_access_key:
                raise LLMError("No active DeepSeek access key configured")
            self.base_url = (base_url or "https://api.deepseek.com").rstrip("/")
            headers = {"Authorization": f"Bearer {provider_access_key}"}
            self.transport = OpenAIChatTransport(
                api_key=provider_access_key,
                base_url=self.base_url,
                timeout=self.timeout_s,
                max_retries=max_retries,
            )
        elif provider == "anthropic":
            if not provider_access_key:
                raise LLMError("No active Anthropic access key configured")
            self.base_url = (base_url or "https://api.anthropic.com").rstrip("/")
            headers = {"x-api-key": provider_access_key}
            self.transport = AnthropicMessagesTransport(
                api_key=provider_access_key,
                base_url=self.base_url,
                timeout=self.timeout_s,
                max_retries=max_retries,
            )
        elif provider in {"opencode_zen", "opencode_go"}:
            if not provider_access_key:
                raise LLMError("No active OpenCode access key configured")
            definition = provider_registry.get(provider)
            self.base_url = (base_url or "https://opencode.ai").rstrip("/")
            headers = {"Authorization": f"Bearer {provider_access_key}"}
            self.transport = RoutedGatewayTransport(
                api_key=provider_access_key,
                base_url=self.base_url,
                models_path=definition.models_endpoint or "",
                timeout=self.timeout_s,
                max_retries=max_retries,
            )
        else:
            raise LLMError(f"Unknown provider: {provider}")

        limits = httpx.Limits(
            max_keepalive_connections=keepalive_connections,
            max_connections=keepalive_max,
        )
        timeout = httpx.Timeout(self.timeout_s)
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            limits=limits,
            headers=headers,
            trust_env=False,
        )

    # -------------------------------------------------------------------------
    def resolve_provider_access_key(self, provider: ProviderName) -> str | None:
        credential_scope = provider_registry.get(provider).credential_scope
        access_key_serializer = AccessKeySerializer()
        try:
            return access_key_serializer.get_active_key_value(credential_scope)
        except Exception as exc:  # noqa: BLE001
            provider_label = provider_registry.get(provider).display_name
            raise LLMError(
                f"Failed to load active {provider_label} access key"
            ) from exc

    # -------------------------------------------------------------------------
    async def close(self) -> None:
        if self.transport is not None:
            await self.transport.close()
        await self.client.aclose()

    # -------------------------------------------------------------------------
    async def __aenter__(self) -> CloudLLMClient:
        return self

    # -------------------------------------------------------------------------
    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # -------------------------------------------------------------------------
    async def list_models(self) -> list[str]:
        return [item.id for item in await self.list_model_descriptors()]

    # -------------------------------------------------------------------------
    async def list_model_descriptors(
        self, *, force_refresh: bool = False
    ) -> list[CloudModelDescriptor]:
        if self.transport is not None:
            descriptors = await self.transport.list_models(force_refresh=force_refresh)
            return [
                enrich_model_descriptor(provider=self.provider, descriptor=item)
                for item in descriptors
            ]
        return []

    # -------------------------------------------------------------------------
    async def check_model_availability(self, name: str) -> None:
        descriptors = await self.list_model_descriptors()
        models = {item.id for item in descriptors}
        aliases = {
            alias
            for item in descriptors
            for alias in item.model_capabilities.aliases
        }
        if models and name not in models and name not in aliases:
            raise LLMError(f"Model '{name}' not found for provider {self.provider}")

    # -------------------------------------------------------------------------
    @staticmethod
    def is_gpt5_family_model(model: str | None) -> bool:
        normalized = (model or "").strip().lower()
        return normalized.startswith("gpt-5")

    # -------------------------------------------------------------------------
    @staticmethod
    def raise_for_status(resp: httpx.Response) -> None:
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            error_code, retryable = CloudLLMClient._http_status_error_code(
                resp.status_code
            )
            raise LLMError(
                f"Cloud provider returned HTTP {resp.status_code}",
                error_code=error_code,
                retryable=retryable,
                status_code=resp.status_code,
                request_id=CloudLLMClient._response_request_id(resp),
                provider_detail=CloudLLMClient._response_detail(resp),
            ) from e

    # -------------------------------------------------------------------------
    @staticmethod
    def _response_request_id(response: object) -> str | None:
        headers = getattr(response, "headers", None)
        if headers is None:
            return None
        for name in ("x-request-id", "request-id", "cf-ray"):
            value = headers.get(name)
            if value:
                return str(value)[:120]
        return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _response_detail(response: object) -> str | None:
        try:
            payload = response.json()  # type: ignore[attr-defined]
        except (AttributeError, TypeError, ValueError):
            payload = getattr(response, "text", None)
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                for key in ("message", "detail", "error", "code"):
                    if error.get(key):
                        return str(error[key])
            for key in ("message", "detail", "error"):
                if payload.get(key):
                    return str(payload[key])
        if payload is None:
            return None
        return str(payload)

    # -------------------------------------------------------------------------
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]] | list[ChatMessage],
        format: str | None = None,
        options: dict[str, Any] | None = None,
        purpose: GenerationPurpose = GenerationPurpose.CLINICAL_SYNTHESIS,
        timeline_complexity: str = "moderate",
        operation: RequestOperation = "chat",
        json_schema: dict[str, Any] | None = None,
        cancel_check: Any | None = None,
    ) -> dict[str, Any] | str:
        result = await self.chat_result(
            model=model,
            messages=messages,
            format=format,
            options=options,
            purpose=purpose,
            timeline_complexity=timeline_complexity,
            operation=operation,
            json_schema=json_schema,
            cancel_check=cancel_check,
        )
        return self._normalize_content(result.content)

    # -------------------------------------------------------------------------
    async def chat_result(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]] | list[ChatMessage],
        format: str | None = None,
        options: dict[str, Any] | None = None,
        purpose: GenerationPurpose = GenerationPurpose.CLINICAL_SYNTHESIS,
        timeline_complexity: str = "moderate",
        operation: RequestOperation = "chat",
        json_schema: dict[str, Any] | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        cancel_check: Any | None = None,
    ) -> ChatResult:
        resolved_model = model or self.default_model
        if not resolved_model:
            raise LLMError("Model is required")
        canonical_messages = [
            item if isinstance(item, ChatMessage) else ChatMessage.model_validate(item)
            for item in messages
        ]
        descriptor = LLMRuntimeConfig.get_model_descriptor(
            self.provider, resolved_model
        )
        effective = LLMRuntimeConfig.resolve_effective_inference_config(
            purpose=purpose,
            provider=self.provider,
            model=resolved_model,
            timeline_complexity=timeline_complexity,
            descriptor=descriptor,
        )
        options_payload = {
            key: value for key, value in (options or {}).items() if key != "temperature"
        }
        if effective.temperature is not None:
            options_payload["temperature"] = effective.temperature
        options_payload.setdefault("max_output_tokens", effective.output_token_limit)

        capability = capability_metadata(
            provider=self.provider,
            model=resolved_model,
            descriptor=descriptor,
        )
        if self.transport is None:
            raise LLMError(f"Provider '{self.provider}' does not support chat yet")
        request = ChatRequest(
            model=resolved_model,
            messages=canonical_messages,
            options=options_payload,
            json_mode=format == "json",
            operation=operation,
            json_schema=json_schema,
            reasoning_level=effective.effective_reasoning_level.value,
            reasoning_parameter=effective.reasoning_parameter,
            reasoning_reserve=effective.reasoning_reserve,
            output_token_limit=effective.output_token_limit,
            temperature=effective.temperature,
            top_p=options_payload.get("top_p"),
            tools=tools or [],
            tool_choice=tool_choice,
            stream=stream,
            capabilities=capability,
            deadline_at=asyncio.get_running_loop().time() + self.timeout_s,
            cancel_check=cancel_check,
        )
        try:
            if stream:
                return await self._collect_stream_result(request)
            return await self.transport.chat(request)
        except Exception as exc:  # noqa: BLE001
            raise self._map_provider_exception(
                exc,
                provider=self.provider,
                model=resolved_model,
                operation=operation,
            ) from exc

    # -------------------------------------------------------------------------
    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]] | list[ChatMessage],
        format: str | None = None,
        options: dict[str, Any] | None = None,
        purpose: GenerationPurpose = GenerationPurpose.CLINICAL_SYNTHESIS,
        timeline_complexity: str = "moderate",
        operation: RequestOperation = "chat",
        json_schema: dict[str, Any] | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        cancel_check: Any | None = None,
    ):
        if self.transport is None:
            raise LLMError(f"Provider '{self.provider}' does not support streaming")
        try:
            request_result = await self._build_chat_request(
                model=model,
                messages=messages,
                format=format,
                options=options,
                purpose=purpose,
                timeline_complexity=timeline_complexity,
                operation=operation,
                json_schema=json_schema,
                tools=tools,
                tool_choice=tool_choice,
                cancel_check=cancel_check,
            )
            async for event in self.transport.stream(request_result):
                yield event
        except Exception as exc:  # noqa: BLE001
            raise self._map_provider_exception(
                exc,
                provider=self.provider,
                model=model or self.default_model,
                operation=operation,
            ) from exc

    # -------------------------------------------------------------------------
    async def _collect_stream_result(self, request: ChatRequest) -> ChatResult:
        if self.transport is None:
            raise LLMError(f"Provider '{self.provider}' does not support streaming")
        partial_error: ChatStreamEvent | None = None
        try:
            async for event in self.transport.stream(request):
                if event.kind == "error":
                    partial_error = event
                if event.kind == "completed" and event.result is not None:
                    if event.result.partial or partial_error is not None:
                        raise LLMError(
                            partial_error.error_message
                            if partial_error is not None and partial_error.error_message
                            else "Provider stream ended before completion",
                            error_code=partial_error.error_code
                            if partial_error is not None and partial_error.error_code
                            else "partial_response",
                        )
                    return event.result
        except Exception as exc:  # noqa: BLE001
            if partial_error is not None:
                raise LLMError(
                    partial_error.error_message or "Provider stream failed",
                    error_code=partial_error.error_code or "partial_response",
                ) from exc
            raise
        raise LLMError(
            "Provider stream ended without a completion event",
            error_code="partial_response",
        )

    # -------------------------------------------------------------------------
    async def _build_chat_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]] | list[ChatMessage],
        format: str | None,
        options: dict[str, Any] | None,
        purpose: GenerationPurpose,
        timeline_complexity: str,
        operation: RequestOperation,
        json_schema: dict[str, Any] | None,
        tools: list[ToolDefinition] | None,
        tool_choice: str | dict[str, Any] | None,
        cancel_check: Any | None,
    ) -> ChatRequest:
        resolved_model = model or self.default_model
        if not resolved_model:
            raise LLMError("Model is required")
        canonical_messages = [
            item if isinstance(item, ChatMessage) else ChatMessage.model_validate(item)
            for item in messages
        ]
        descriptor = LLMRuntimeConfig.get_model_descriptor(
            self.provider, resolved_model
        )
        effective = LLMRuntimeConfig.resolve_effective_inference_config(
            purpose=purpose,
            provider=self.provider,
            model=resolved_model,
            timeline_complexity=timeline_complexity,
            descriptor=descriptor,
        )
        options_payload = {
            key: value for key, value in (options or {}).items() if key != "temperature"
        }
        if effective.temperature is not None:
            options_payload["temperature"] = effective.temperature
        options_payload.setdefault("max_output_tokens", effective.output_token_limit)
        return ChatRequest(
            model=resolved_model,
            messages=canonical_messages,
            options=options_payload,
            json_mode=format == "json",
            operation=operation,
            json_schema=json_schema,
            reasoning_level=effective.effective_reasoning_level.value,
            reasoning_parameter=effective.reasoning_parameter,
            reasoning_reserve=effective.reasoning_reserve,
            output_token_limit=effective.output_token_limit,
            temperature=effective.temperature,
            top_p=options_payload.get("top_p"),
            tools=tools or [],
            tool_choice=tool_choice,
            stream=True,
            capabilities=capability_metadata(
                provider=self.provider,
                model=resolved_model,
                descriptor=descriptor,
            ),
            deadline_at=asyncio.get_running_loop().time() + self.timeout_s,
            cancel_check=cancel_check,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def resolve_gemini_model_resource(model: str | None) -> str:
        model_name = (model or "").strip()
        if not model_name:
            raise LLMError("Gemini model is required")
        if model_name.startswith("models/"):
            return model_name
        return f"models/{model_name}"

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_content(content: Any) -> dict[str, Any] | str:
        if isinstance(content, dict):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
                    continue
                if isinstance(part, str):
                    chunks.append(part)
                    continue
                chunks.append(str(part))
            content = "".join(chunks)
        if isinstance(content, str):
            try:
                loaded = json.loads(content)
            except json.JSONDecodeError:
                return content
            return loaded if isinstance(loaded, dict) else content
        return str(content)

    # -------------------------------------------------------------------------
    @staticmethod
    def _http_status_error_code(status_code: int) -> tuple[str, bool]:
        if status_code in {401, 403}:
            return "authentication", False
        if status_code == 404:
            return "configuration", False
        if status_code == 408:
            return "timeout", True
        if status_code == 429:
            return "rate_limited", True
        if 500 <= status_code <= 599:
            return "upstream_error", True
        return "provider_error", False

    # -------------------------------------------------------------------------
    @staticmethod
    def _map_provider_exception(
        exc: Exception,
        *,
        provider: str | None = None,
        model: str | None = None,
        operation: str | None = None,
    ) -> LLMError:
        if isinstance(exc, TransportCancellation):
            return LLMError(
                "LLM request cancelled",
                error_code="cancelled",
                provider=provider,
                model=model,
                operation=operation,
            )
        if isinstance(exc, TransportUnsupportedCapability):
            return LLMError(
                str(exc),
                error_code="unsupported_capability",
                provider=provider,
                model=model,
                operation=operation,
            )
        if isinstance(exc, TransportPartialResponse):
            return LLMError(
                str(exc),
                error_code="partial_response",
                retryable=True,
                provider=provider,
                model=model,
                operation=operation,
            )
        if isinstance(exc, LLMError):
            return exc.with_context(
                provider=provider,
                model=model,
                operation=operation,
            )
        if isinstance(exc, (TimeoutError, APITimeoutError)):
            return LLMTimeout(
                "Timed out waiting for cloud chat response",
                provider=provider,
                model=model,
                operation=operation,
            )
        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeout(
                "Timed out waiting for cloud chat response",
                provider=provider,
                model=model,
                operation=operation,
            )
        if isinstance(exc, (httpx.NetworkError, APIConnectionError)):
            return LLMError(
                "Cloud provider connection failed",
                error_code="network_unavailable",
                retryable=True,
                provider=provider,
                model=model,
                operation=operation,
            )
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code
            error_code, retryable = CloudLLMClient._http_status_error_code(status_code)
            return LLMError(
                f"Cloud provider returned HTTP {status_code}",
                error_code=error_code,
                retryable=retryable,
                provider=provider,
                model=model,
                operation=operation,
                status_code=status_code,
                request_id=CloudLLMClient._response_request_id(exc.response),
                provider_detail=CloudLLMClient._response_detail(exc.response),
            )
        if isinstance(exc, APIStatusError):
            status_code = getattr(exc, "status_code", None)
            if not isinstance(status_code, int):
                response = getattr(exc, "response", None)
                status_code = getattr(response, "status_code", None)
            if isinstance(status_code, int):
                error_code, retryable = CloudLLMClient._http_status_error_code(
                    status_code
                )
                response = getattr(exc, "response", None)
                body = getattr(exc, "body", None)
                return LLMError(
                    f"Cloud provider returned HTTP {status_code}",
                    error_code=error_code,
                    retryable=retryable,
                    provider=provider,
                    model=model,
                    operation=operation,
                    status_code=status_code,
                    request_id=CloudLLMClient._response_request_id(response),
                    provider_detail=(
                        CloudLLMClient._response_detail(response)
                        if response is not None
                        else body
                    ),
                )
        timeout_error = getattr(genai_errors, "TimeoutError", None)
        if timeout_error is not None and isinstance(exc, timeout_error):
            return LLMTimeout(
                "Timed out waiting for cloud chat response",
                provider=provider,
                model=model,
                operation=operation,
            )
        if isinstance(exc, OpenAIError):
            return LLMError(
                f"Cloud LLM call failed: {exc}",
                provider=provider,
                model=model,
                operation=operation,
            )
        error_name = exc.__class__.__name__.lower()
        if "timeout" in error_name:
            return LLMTimeout(
                "Timed out waiting for cloud chat response",
                provider=provider,
                model=model,
                operation=operation,
            )
        return LLMError(
            f"Cloud LLM call failed: {exc}",
            provider=provider,
            model=model,
            operation=operation,
            provider_detail=str(exc),
        )

    # -------------------------------------------------------------------------
    async def llm_text_call(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        purpose: GenerationPurpose = GenerationPurpose.CLINICAL_SYNTHESIS,
    ) -> str:
        resolved_model = model or (self.default_model or "")
        raw = await self.chat(
            model=resolved_model,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt},
            ],
            purpose=purpose,
        )
        return json.dumps(raw) if isinstance(raw, dict) else str(raw)

    # -------------------------------------------------------------------------
    async def embed(
        self,
        *,
        model: str,
        input_texts: list[str],
    ) -> list[list[float]]:
        if not input_texts:
            return []

        if self.provider == "openai":
            return await self.embed_openai(model=model, input_texts=input_texts)
        if self.provider == "gemini":
            return await self.embed_gemini(model=model, input_texts=input_texts)
        raise LLMError(f"Provider '{self.provider}' does not support embeddings yet")

    # -------------------------------------------------------------------------
    async def embed_openai(
        self,
        *,
        model: str,
        input_texts: list[str],
    ) -> list[list[float]]:
        body = {"model": model or self.default_model, "input": input_texts}

        try:
            resp = await self.client.post("/embeddings", json=body)
        except httpx.TimeoutException as exc:
            raise LLMTimeout("Timed out waiting for OpenAI embeddings") from exc

        self.raise_for_status(resp)

        data = resp.json()
        entries = sorted(data.get("data", []), key=lambda entry: entry.get("index", 0))
        embeddings: list[list[float]] = []
        for item in entries:
            vector = item.get("embedding", [])
            try:
                embeddings.append([float(value) for value in vector])
            except (TypeError, ValueError) as exc:
                raise LLMError("Non-numeric values found in OpenAI embeddings") from exc

        if len(embeddings) != len(input_texts):
            raise LLMError("Mismatch between OpenAI embeddings and inputs")
        return embeddings

    # -------------------------------------------------------------------------
    async def embed_gemini(
        self,
        *,
        model: str,
        input_texts: list[str],
    ) -> list[list[float]]:
        resolved_model = model or self.default_model
        model_resource = self.resolve_gemini_model_resource(resolved_model)
        requests_payload = [
            {
                "model": model_resource,
                "content": {"parts": [{"text": text}]},
            }
            for text in input_texts
        ]
        body = {"requests": requests_payload}
        path = f"/{model_resource}:batchEmbedContents"

        try:
            resp = await self.client.post(path, json=body)
        except httpx.TimeoutException as exc:
            raise LLMTimeout("Timed out waiting for Gemini embeddings") from exc

        self.raise_for_status(resp)

        data = resp.json()
        embeddings: list[list[float]] = []
        for item in data.get("embeddings", []):
            values = item.get("values") or item.get("embedding") or []
            try:
                embeddings.append([float(value) for value in values])
            except (TypeError, ValueError) as exc:
                raise LLMError("Non-numeric values found in Gemini embeddings") from exc

        if len(embeddings) != len(input_texts):
            raise LLMError("Mismatch between Gemini embeddings and inputs")
        return embeddings

    # -------------------------------------------------------------------------
    async def llm_structured_call(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        purpose: GenerationPurpose = GenerationPurpose.STRUCTURED_EXTRACTION,
        use_json_mode: bool = True,
        max_repair_attempts: int = 2,
        timeline_complexity: str = "moderate",
        cancel_check: Any | None = None,
    ) -> T:
        parser = StructuredOutputParser(schema=schema)
        format_instructions = parser.get_format_instructions()
        resolved_model = model or (self.default_model or "")
        messages = self.build_structured_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            format_instructions=format_instructions,
        )
        raw = await self.chat(
            model=resolved_model,
            messages=messages,
            format="json" if use_json_mode else None,
            options=None,
            purpose=purpose,
            timeline_complexity=timeline_complexity,
            operation="structured_output",
            json_schema=schema.model_json_schema() if use_json_mode else None,
            cancel_check=cancel_check,
        )
        text = json.dumps(raw) if isinstance(raw, dict) else str(raw)
        return await self.parse_with_repairs(
            parser=parser,
            text=text,
            model=resolved_model,
            system_prompt=system_prompt,
            format_instructions=format_instructions,
            use_json_mode=use_json_mode,
            max_repair_attempts=max_repair_attempts,
            cancel_check=cancel_check,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def build_structured_messages(
        *,
        system_prompt: str,
        user_prompt: str,
        format_instructions: str,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": f"{system_prompt.strip()}\n\n{format_instructions}",
            },
            {"role": "user", "content": user_prompt},
        ]

    # -------------------------------------------------------------------------
    @staticmethod
    def build_repair_messages(
        *,
        system_prompt: str,
        format_instructions: str,
        text: str,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt.strip()},
            {
                "role": "user",
                "content": build_json_repair_user_prompt(
                    format_instructions=format_instructions,
                    previous_reply=clip_structured_repair_text(text),
                ),
            },
        ]

    # -------------------------------------------------------------------------
    @staticmethod
    def build_compact_repair_messages(*, text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": COMPACT_JSON_REPAIR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_compact_json_repair_user_prompt(
                    previous_reply=clip_structured_repair_text(text),
                ),
            },
        ]

    # -------------------------------------------------------------------------
    async def parse_with_repairs(
        self,
        *,
        parser: StructuredOutputParser[T],
        text: str,
        model: str,
        system_prompt: str,
        format_instructions: str,
        use_json_mode: bool,
        max_repair_attempts: int,
        cancel_check: Any | None = None,
    ) -> T:
        for attempt in range(max_repair_attempts + 1):
            try:
                return parser.parse(text)
            except Exception as err:
                if attempt >= max_repair_attempts:
                    logger.error(
                        "Structured parse failed after retries: schema=%s attempts=%s output_length=%s output_hash=%s error=%s",
                        parser.schema.__name__,
                        attempt + 1,
                        len(text or ""),
                        short_output_hash(text or ""),
                        type(err).__name__,
                    )
                    raise RuntimeError(f"Structured parsing failed: {err}") from err

                repair_messages = self.build_repair_messages(
                    system_prompt=system_prompt,
                    format_instructions=format_instructions,
                    text=text,
                )
                if looks_like_schema_echo(text):
                    repair_messages = self.build_compact_repair_messages(text=text)
                raw = await self.chat(
                    model=model,
                    messages=repair_messages,
                    format="json" if use_json_mode else None,
                    purpose=GenerationPurpose.JSON_REPAIR,
                    operation="json_repair",
                    json_schema=parser.schema.model_json_schema()
                    if use_json_mode
                    else None,
                    cancel_check=cancel_check,
                )
                text = json.dumps(raw) if isinstance(raw, dict) else str(raw)

        raise RuntimeError("No structured output produced by the model")

    # -------------------------------------------------------------------------
    @staticmethod
    def parse_json(obj_or_text: dict[str, Any] | str) -> dict[str, Any] | None:
        if isinstance(obj_or_text, dict):
            return obj_or_text
        try:
            return parse_json_object_strict(obj_or_text)
        except ValueError:
            return None
