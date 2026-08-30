from __future__ import annotations

import hashlib
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
    _cache: dict[str, tuple[datetime, list[CloudModelDescriptor]]] = {}
    _cache_ttl = timedelta(minutes=15)
    _opencode_go_anthropic_models = frozenset(
        {
            "minimax-m3",
            "minimax-m2.7",
            "minimax-m2.5",
            "qwen3.7-max",
            "qwen3.7-plus",
            "qwen3.6-plus",
        }
    )

    # -------------------------------------------------------------------------
    def __init__(
        self, *, api_key: str, base_url: str, models_path: str, timeout: float
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.models_path = models_path
        self.timeout = timeout
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
        models = []
        for item in response.json().get("data", []):
            endpoint = str(item.get("endpoint") or item.get("endpoint_family") or "")
            descriptor = CloudModelDescriptor(
                id=item["id"],
                display_name=item.get("name") or item["id"],
                endpoint_family=endpoint,
            )
            models.append(descriptor)
        self._models = {item.id: item for item in models}
        self._cache[cache_key] = (datetime.now(UTC) + self._cache_ttl, models)
        return models

    # -------------------------------------------------------------------------
    async def chat(self, request: ChatRequest) -> ChatResult:
        descriptor = self._models.get(request.model)
        route_source = "catalog"
        if descriptor is None and self.models_path == self._opencode_go_models_path:
            # OpenCode Go has a documented route family even when its catalog is
            # temporarily unavailable or does not contain the selected model.
            # Keep the explicit model selection and route it directly instead of
            # turning a catalog outage into a timeline fallback.
            descriptor = CloudModelDescriptor(
                id=request.model,
                display_name=request.model,
            )
            route_source = "known_opencode_go_route"
        elif descriptor is None:
            await self.list_models()
            descriptor = self._models.get(request.model)
        if descriptor is None:
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
                api_key=self.api_key, base_url=transport_base_url, timeout=self.timeout
            )
        elif "messages" in endpoint:
            anthropic_base = transport_base_url.removesuffix("/v1")
            transport = AnthropicMessagesTransport(
                api_key=self.api_key, base_url=anthropic_base, timeout=self.timeout
            )
        elif "chat/completions" in endpoint:
            transport = OpenAIChatTransport(
                api_key=self.api_key, base_url=transport_base_url, timeout=self.timeout
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
            "route_source=%s message_count=%d message_chars=%d",
            self.models_path,
            request.model,
            endpoint,
            route_source,
            len(request.messages),
            message_chars,
        )
        self._transports.append(transport)
        return await transport.chat(request)

    # -------------------------------------------------------------------------
    def _resolve_transport_endpoint(self, descriptor: CloudModelDescriptor) -> str:
        if descriptor.endpoint_family:
            return descriptor.endpoint_family.lower()
        if self.models_path != self._opencode_go_models_path:
            return ""
        if descriptor.id in self._opencode_go_anthropic_models:
            return "messages"
        return "chat/completions"

    # -------------------------------------------------------------------------
    async def check_connectivity(self, model: str) -> ConnectivityResult:
        try:
            result = await self.chat(
                ChatRequest(
                    model=model,
                    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                )
            )
            return ConnectivityResult(ok=True, response_preview=result.content[:200])
        except Exception as exc:
            return ConnectivityResult(ok=False, error=str(exc))

    # -------------------------------------------------------------------------
    async def close(self) -> None:
        for transport in self._transports:
            await transport.close()
