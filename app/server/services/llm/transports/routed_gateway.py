from __future__ import annotations

import hashlib
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import httpx

from common.utils.logger import logger
from common.version import resolve_application_version
from domain.llm.providers import CloudModelDescriptor
from domain.llm.transports import (
    ChatMessage,
    ChatRequest,
    ChatResult,
    ChatStreamEvent,
    ConnectivityResult,
)
from services.llm.transports.anthropic_messages import AnthropicMessagesTransport
from services.llm.transports.base import (
    CloudTransport,
    StructuredTransportMixin,
    call_with_retries,
)
from services.llm.transports.openai_chat import OpenAIChatTransport
from services.llm.transports.openai_responses import OpenAIResponsesTransport
from services.llm.model_capabilities import (
    catalog_capability_metadata,
    resolve_endpoint_family,
)

###############################################################################
class RoutedGatewayTransport(StructuredTransportMixin):
    _opencode_go_models_path = "/zen/go/v1/models"
    _opencode_go_route_map_version = "2026-09-14"
    _cache: dict[str, tuple[datetime, list[CloudModelDescriptor]]] = {}
    _cache_ttl = timedelta(minutes=15)
    _opencode_go_route_maps: dict[str, dict[str, frozenset[str]]] = {
        "opencode_go": {
        "responses": frozenset(
            {
                "grok-4.6",
                "gpt-5.6-luna",
                "muse-spark-1.3-contributor",
                "muse-spark-1.2-contributor",
            }
        ),
        "chat/completions": frozenset(
            {
                "glm-5.3-flash",
                "glm-5.3",
                "glm-5.2",
                "glm-5.1",
                "kimi-k3",
                "kimi-k2.7-code",
                "kimi-k2.6",
                "longcat-2.0",
                "deepseek-v4-pro",
                "deepseek-v4-flash",
                "deepseek-v4.1-flash",
                "deepseek-flash",
                "deepseek-v4-flash-vision-exp",
                "mimo-v2.5",
                "mimo-v2.5-pro",
                "hy4-preview",
                "hy3",
                "omen-alpha",
            }
        ),
        "messages": frozenset(
            {
                "minimax-m3",
                "minimax-m2.7",
                "minimax-m2.5",
                "qwen3.8-max",
                "qwen3.8-flash",
                "qwen3.7-max",
                "qwen3.7-plus",
                "qwen3.6-plus",
            }
        ),
        },
        # Zen currently documents the older DeepSeek route.  It is retained as
        # a provider route, but it is not treated as V4.1 without explicit
        # catalog or response attestation.
        "opencode_zen": {
            "chat/completions": frozenset(
                {"deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"}
            ),
        },
    }

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        models_path: str,
        timeout: float,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.models_path = models_path
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.provider_id = (
            "opencode_go" if models_path == self._opencode_go_models_path else "opencode_zen"
        )
        self._session_headers = {
            "x-opencode-session": uuid.uuid4().hex,
            "User-Agent": f"diligent-clinical-copilot/{resolve_application_version()}",
        }
        self._models: dict[str, CloudModelDescriptor] = {}
        self._transports: list[CloudTransport] = []

    # -------------------------------------------------------------------------
    async def list_models(
        self, *, force_refresh: bool = False
    ) -> list[CloudModelDescriptor]:
        key_fingerprint = hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()[:16]
        cache_key = f"{self.base_url}{self.models_path}:{key_fingerprint}"
        cached = self._cache.get(cache_key)
        if not force_refresh and cached and cached[0] > datetime.now(UTC):
            self._models = {item.id: item for item in cached[1]}
            return list(cached[1])
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                trust_env=False,
                headers={"Authorization": f"Bearer {self.api_key}"},
            ) as client:

                async def get_checked() -> httpx.Response:
                    response = await client.get(self.models_path)
                    response.raise_for_status()
                    return response

                response = await call_with_retries(
                    get_checked,
                    max_retries=self.max_retries,
                )
        except Exception:
            if cached and not force_refresh:
                self._models = {item.id: item for item in cached[1]}
                return list(cached[1])
            raise
        models: list[CloudModelDescriptor] = []
        for item in response.json().get("data", []):
            if not isinstance(item, dict) or not item.get("id"):
                continue
            endpoint = self._normalize_endpoint(
                str(item.get("endpoint") or item.get("endpoint_family") or "")
            )
            descriptor = CloudModelDescriptor(
                id=str(item["id"]),
                display_name=str(item.get("name") or item["id"]),
                endpoint_family=endpoint,
                model_capabilities=catalog_capability_metadata(
                    item, endpoint_family=endpoint
                ),
            )
            descriptor = self._annotate_descriptor(descriptor)
            models.append(descriptor)
        self._models = {item.id: item for item in models}
        self._cache[cache_key] = (datetime.now(UTC) + self._cache_ttl, models)
        return models

    # -------------------------------------------------------------------------
    async def _resolve_descriptor(
        self, model: str
    ) -> tuple[CloudModelDescriptor, str]:
        descriptor = self._models.get(model)
        route_source = "catalog"
        catalog_error: Exception | None = None
        if descriptor is None:
            try:
                await self.list_models()
            except Exception as exc:  # noqa: BLE001
                catalog_error = exc
            descriptor = self._models.get(model)
        if descriptor is None:
            fallback_endpoint = self._static_route_for_model(model)
            if fallback_endpoint and self.provider_id == "opencode_go":
                descriptor = CloudModelDescriptor(
                    id=model,
                    display_name=model,
                    endpoint_family=fallback_endpoint,
                    supports_json_mode=True,
                )
                descriptor = self._annotate_descriptor(descriptor)
                route_source = (
                    "documented_opencode_go_route@"
                    f"{self._opencode_go_route_map_version}"
                )
        if descriptor is None:
            if catalog_error is not None:
                raise ValueError(
                    f"Provider model metadata is unavailable for {model}"
                ) from catalog_error
            raise ValueError(
                "Provider model metadata does not include the requested model"
            )
        return descriptor, route_source

    # -------------------------------------------------------------------------
    def _build_transport(
        self, descriptor: CloudModelDescriptor
    ) -> tuple[CloudTransport, str]:
        endpoint = self._resolve_transport_endpoint(descriptor)
        if not endpoint:
            raise ValueError(
                "Provider model metadata does not declare a transport endpoint"
            )
        route_prefix = self.models_path.removesuffix("/models")
        transport_base_url = f"{self.base_url}{route_prefix}"
        if endpoint.startswith("http"):
            transport_base_url = endpoint.rsplit("/", maxsplit=1)[0]
        elif endpoint.startswith("/"):
            transport_base_url = f"{self.base_url}{endpoint.rsplit('/', maxsplit=1)[0]}"
        if "responses" in endpoint:
            transport = OpenAIResponsesTransport(
                api_key=self.api_key,
                base_url=transport_base_url,
                timeout=self.timeout,
                default_headers=self._session_headers,
                max_retries=self.max_retries,
            )
        elif "messages" in endpoint:
            anthropic_base = transport_base_url.removesuffix("/v1")
            transport = AnthropicMessagesTransport(
                api_key=self.api_key,
                base_url=anthropic_base,
                timeout=self.timeout,
                default_headers=self._session_headers,
                max_retries=self.max_retries,
            )
        elif "chat/completions" in endpoint:
            transport = OpenAIChatTransport(
                api_key=self.api_key,
                base_url=transport_base_url,
                timeout=self.timeout,
                default_headers=self._session_headers,
                max_retries=self.max_retries,
            )
        else:
            raise ValueError(
                f"Unsupported provider model transport: {descriptor.endpoint_family}"
            )
        return transport, endpoint

    # -------------------------------------------------------------------------
    async def _transport_for_request(
        self, request: ChatRequest
    ) -> tuple[CloudTransport, str, str]:
        descriptor, route_source = await self._resolve_descriptor(request.model)
        transport, endpoint = self._build_transport(descriptor)
        message_chars = sum(
            len(str(message.content or "")) for message in request.messages
        )
        logger.info(
            "Cloud chat request attempted: gateway_path=%s model=%s endpoint=%s "
            "route_source=%s operation=%s message_count=%d message_chars=%d",
            self.models_path,
            request.model,
            endpoint,
            route_source,
            request.operation,
            len(request.messages),
            message_chars,
        )
        self._transports.append(transport)
        return transport, endpoint, route_source

    # -------------------------------------------------------------------------
    async def chat(self, request: ChatRequest) -> ChatResult:
        transport, _, _ = await self._transport_for_request(request)
        return await transport.chat(request)

    # -------------------------------------------------------------------------
    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        transport, _, _ = await self._transport_for_request(request)
        async for event in transport.stream(request):
            yield event

    # -------------------------------------------------------------------------
    def _resolve_transport_endpoint(self, descriptor: CloudModelDescriptor) -> str:
        if descriptor.endpoint_family:
            normalized = self._normalize_endpoint(descriptor.endpoint_family)
            if normalized:
                return normalized
        return self._static_route_for_model(descriptor.id)

    # -------------------------------------------------------------------------
    def _annotate_descriptor(self, descriptor: CloudModelDescriptor) -> CloudModelDescriptor:
        metadata = descriptor.model_capabilities
        route = descriptor.endpoint_family or metadata.endpoint_family
        static_route = self._static_route_for_model(descriptor.id)
        route = route or static_route
        is_chat = route == "chat/completions"
        is_responses = route == "responses"
        is_messages = route == "messages"
        semantic_model_id = metadata.semantic_model_id
        aliases = metadata.aliases
        if descriptor.id in {"deepseek-flash", "deepseek-v4.1-flash"}:
            semantic_model_id = "deepseek-v4.1-flash"
            aliases = tuple(sorted(set(aliases) | {"deepseek-v4-flash"}))
        annotated = metadata.model_copy(
            update={
                "endpoint_family": route or metadata.endpoint_family,
                "semantic_model_id": semantic_model_id,
                "aliases": aliases,
                "supports_chat": (
                    is_chat if metadata.supports_chat is None else metadata.supports_chat
                ),
                "supports_streaming": (
                    is_chat or is_responses or is_messages
                    if metadata.supports_streaming is None
                    else metadata.supports_streaming
                ),
                "supports_tools": metadata.supports_tools,
                "supports_structured_output": (
                    True
                    if metadata.supports_structured_output is None and route
                    else metadata.supports_structured_output
                ),
                "supports_json_mode": (
                    True
                    if metadata.supports_json_mode is None and route
                    else metadata.supports_json_mode
                ),
                "supports_temperature": (
                    True
                    if metadata.supports_temperature is None and route == "chat/completions"
                    else metadata.supports_temperature
                ),
                "supports_usage_metadata": metadata.supports_usage_metadata,
                "supports_finish_reason": metadata.supports_finish_reason,
                "evidence": (
                    metadata.evidence
                    if metadata.evidence != "fallback"
                    else ("catalog" if descriptor.endpoint_family else "documented")
                ),
            }
        )
        return descriptor.model_copy(
            update={"model_capabilities": annotated, "endpoint_family": route}
        )

    # -------------------------------------------------------------------------
    def _static_route_for_model(self, model: str) -> str:
        catalog_route = resolve_endpoint_family(
            provider=self.provider_id,
            model=model,
        )
        if catalog_route:
            return catalog_route
        normalized = (model or "").strip().casefold()
        routes = self._opencode_go_route_maps.get(self.provider_id, {})
        for endpoint, models in routes.items():
            if normalized in {item.casefold() for item in models}:
                return endpoint
        return ""

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_endpoint(value: str) -> str:
        normalized = value.strip().lower()
        if "responses" in normalized:
            return "responses"
        if "messages" in normalized:
            return "messages"
        if "chat/completions" in normalized or normalized in {
            "chat",
            "chat_completions",
        }:
            return "chat/completions"
        return ""

    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_positive_int(value: object) -> int | None:
        if value is None:
            return None
        try:
            parsed = int(str(value))
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_optional_bool(value: object) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1", "enabled", "supported"}:
                return True
            if normalized in {"false", "no", "0", "disabled", "unsupported"}:
                return False
        return bool(value) if isinstance(value, (dict, list, tuple, set)) else None

    # -------------------------------------------------------------------------
    async def check_connectivity(self, model: str) -> ConnectivityResult:
        try:
            result = await self.chat(
                ChatRequest(
                    model=model,
                    messages=[ChatMessage(role="user", content="Reply with exactly: OK")],
                    operation="connectivity",
                )
            )
            return ConnectivityResult(ok=True, response_preview=result.content[:200])
        except Exception as exc:
            return ConnectivityResult(ok=False, error=str(exc))

    # -------------------------------------------------------------------------
    async def close(self) -> None:
        for transport in self._transports:
            await transport.close()
