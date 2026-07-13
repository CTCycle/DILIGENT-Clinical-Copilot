from __future__ import annotations

import httpx
from datetime import UTC, datetime, timedelta

from domain.llm.providers import CloudModelDescriptor
from services.llm.transports.anthropic_messages import AnthropicMessagesTransport
from services.llm.transports.base import (
    ChatRequest,
    ChatResult,
    ConnectivityResult,
    StructuredTransportMixin,
)
from services.llm.transports.openai_chat import OpenAIChatTransport
from services.llm.transports.openai_responses import OpenAIResponsesTransport


###############################################################################
class RoutedGatewayTransport(StructuredTransportMixin):
    _cache: dict[str, tuple[datetime, list[CloudModelDescriptor]]] = {}
    _cache_ttl = timedelta(minutes=15)

    # -------------------------------------------------------------------------
    def __init__(
        self, *, api_key: str, base_url: str, models_path: str, timeout: float
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.models_path = models_path
        self.timeout = timeout
        self._models: dict[str, CloudModelDescriptor] = {}
        self._transports: list[object] = []

    # -------------------------------------------------------------------------
    async def list_models(self) -> list[CloudModelDescriptor]:
        cache_key = f"{self.base_url}{self.models_path}"
        cached = self._cache.get(cache_key)
        if cached and cached[0] > datetime.now(UTC):
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
            if cached:
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
        if request.model not in self._models:
            await self.list_models()
        descriptor = self._models.get(request.model)
        if descriptor is None or not descriptor.endpoint_family:
            raise ValueError(
                "Provider model metadata does not declare a transport endpoint"
            )
        endpoint = descriptor.endpoint_family.lower()
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
        self._transports.append(transport)
        return await transport.chat(request)

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
