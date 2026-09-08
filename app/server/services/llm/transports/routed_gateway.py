from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime, timedelta

import httpx

from common.utils.logger import logger
from domain.llm.providers import CloudModelDescriptor
from domain.llm.transports import ChatRequest, ChatResult, ConnectivityResult
from services.llm.transports.anthropic_messages import AnthropicMessagesTransport
from services.llm.transports.base import CloudTransport, StructuredTransportMixin
from services.llm.transports.openai_chat import OpenAIChatTransport
from services.llm.transports.openai_responses import OpenAIResponsesTransport

###############################################################################
class RoutedGatewayTransport(StructuredTransportMixin):
    _opencode_go_models_path = "/zen/go/v1/models"
    _opencode_go_route_map_version = "2026-09-07"
    _cache: dict[str, tuple[datetime, list[CloudModelDescriptor]]] = {}
    _cache_ttl = timedelta(minutes=15)
    _opencode_go_route_map: dict[str, frozenset[str]] = {
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
    }

    # -------------------------------------------------------------------------
    def __init__(
        self, *, api_key: str, base_url: str, models_path: str, timeout: float
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.models_path = models_path
        self.timeout = timeout
        self._session_headers = {
            "x-opencode-session": uuid.uuid4().hex,
            "User-Agent": "diligent-clinical-copilot/3.3",
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
                response = await client.get(self.models_path)
                response.raise_for_status()
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
            raw_capabilities = item.get("capabilities")
            if not isinstance(raw_capabilities, dict):
                raw_capabilities = {}
            descriptor = CloudModelDescriptor(
                id=str(item["id"]),
                display_name=str(item.get("name") or item["id"]),
                endpoint_family=endpoint,
                input_token_limit=self._coerce_positive_int(
                    item.get("input_token_limit") or item.get("max_input_tokens")
                ),
                output_token_limit=self._coerce_positive_int(
                    item.get("output_token_limit") or item.get("max_output_tokens")
                ),
                supports_thinking=self._coerce_optional_bool(
                    item.get("supports_thinking", raw_capabilities.get("thinking"))
                ),
                supports_temperature=self._coerce_optional_bool(
                    item.get(
                        "supports_temperature", raw_capabilities.get("temperature")
                    )
                ),
                supports_json_mode=self._coerce_optional_bool(
                    item.get(
                        "supports_json_mode",
                        raw_capabilities.get("structured_output")
                        or raw_capabilities.get("structured_outputs"),
                    )
                ),
                supports_native_json_schema=self._coerce_optional_bool(
                    item.get(
                        "supports_native_json_schema",
                        raw_capabilities.get("native_json_schema"),
                    )
                ),
            )
            models.append(descriptor)
        self._models = {item.id: item for item in models}
        self._cache[cache_key] = (datetime.now(UTC) + self._cache_ttl, models)
        return models

    # -------------------------------------------------------------------------
    async def chat(self, request: ChatRequest) -> ChatResult:
        descriptor = self._models.get(request.model)
        route_source = "catalog"
        catalog_error: Exception | None = None
        if descriptor is None:
            try:
                await self.list_models()
            except Exception as exc:  # noqa: BLE001
                catalog_error = exc
            descriptor = self._models.get(request.model)
        if descriptor is None and self.models_path == self._opencode_go_models_path:
            fallback_endpoint = self._fallback_endpoint_for_model(request.model)
            if fallback_endpoint:
                descriptor = CloudModelDescriptor(
                    id=request.model,
                    display_name=request.model,
                    endpoint_family=fallback_endpoint,
                    supports_json_mode=True,
                )
                route_source = (
                    "documented_opencode_go_route@"
                    f"{self._opencode_go_route_map_version}"
                )
        if descriptor is None:
            if catalog_error is not None:
                raise ValueError(
                    f"Provider model metadata is unavailable for {request.model}"
                ) from catalog_error
            raise ValueError(
                "Provider model metadata does not include the requested model"
            )
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
            )
        elif "messages" in endpoint:
            anthropic_base = transport_base_url.removesuffix("/v1")
            transport = AnthropicMessagesTransport(
                api_key=self.api_key,
                base_url=anthropic_base,
                timeout=self.timeout,
                default_headers=self._session_headers,
            )
        elif "chat/completions" in endpoint:
            transport = OpenAIChatTransport(
                api_key=self.api_key,
                base_url=transport_base_url,
                timeout=self.timeout,
                default_headers=self._session_headers,
            )
        else:
            raise ValueError(
                f"Unsupported provider model transport: {descriptor.endpoint_family}"
            )
        message_chars = sum(
            len(str(message.get("content") or "")) for message in request.messages
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
        return await transport.chat(request)

    # -------------------------------------------------------------------------
    def _resolve_transport_endpoint(self, descriptor: CloudModelDescriptor) -> str:
        if descriptor.endpoint_family:
            normalized = self._normalize_endpoint(descriptor.endpoint_family)
            if normalized:
                return normalized
        if self.models_path != self._opencode_go_models_path:
            return ""
        return self._fallback_endpoint_for_model(descriptor.id)

    # -------------------------------------------------------------------------
    @classmethod
    def _fallback_endpoint_for_model(cls, model: str) -> str:
        normalized = (model or "").strip().casefold()
        for endpoint, models in cls._opencode_go_route_map.items():
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
                    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
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
