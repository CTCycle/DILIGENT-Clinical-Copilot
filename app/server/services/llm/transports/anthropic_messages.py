from __future__ import annotations

from typing import cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam

from domain.llm.providers import CloudModelDescriptor
from domain.llm.transports import ChatRequest, ChatResult, ConnectivityResult
from services.llm.transports.base import StructuredTransportMixin

###############################################################################
class AnthropicMessagesTransport(StructuredTransportMixin):

    # -------------------------------------------------------------------------
    def __init__(self, *, api_key: str, base_url: str, timeout: float) -> None:
        self.client = AsyncAnthropic(
            api_key=api_key, base_url=base_url, timeout=timeout
        )

    # -------------------------------------------------------------------------
    async def chat(self, request: ChatRequest) -> ChatResult:
        system = "\n\n".join(
            item["content"] for item in request.messages if item.get("role") == "system"
        )
        messages = cast(
            list[MessageParam],
            [item for item in request.messages if item.get("role") != "system"],
        )
        requested_max_tokens = request.options.get("max_tokens")
        max_tokens = int(request.output_token_limit or requested_max_tokens or 0)
        kwargs: dict[str, object] = {
            "model": request.model,
            "system": system,
            "messages": messages,
        }
        if request.reasoning_level and request.reasoning_level != "off":
            budget_tokens = max(1024, int(request.reasoning_reserve or 0))
            max_tokens = max(max_tokens, int(request.output_token_limit or 0) + budget_tokens)
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget_tokens,
            }
        if max_tokens <= 0:
            raise ValueError("Anthropic requests require an output token limit")
        kwargs["max_tokens"] = max_tokens
        if "temperature" in request.options and request.reasoning_level in {None, "off"}:
            kwargs["temperature"] = request.options["temperature"]
        response = await self.client.messages.create(
            **kwargs,
        )
        text = "".join(getattr(block, "text", "") for block in response.content)
        return ChatResult(content=text)

    # -------------------------------------------------------------------------
    async def list_models(
        self, *, force_refresh: bool = False
    ) -> list[CloudModelDescriptor]:
        del force_refresh
        models: list[CloudModelDescriptor] = []
        after_id: str | None = None
        while True:
            if after_id is None:
                page = await self.client.models.list(limit=100)
            else:
                page = await self.client.models.list(limit=100, after_id=after_id)
            models.extend(
                CloudModelDescriptor(id=item.id, display_name=item.display_name)
                for item in page.data
            )
            if not getattr(page, "has_more", False) or not page.data:
                return models
            after_id = page.data[-1].id

    # -------------------------------------------------------------------------
    async def check_connectivity(self, model: str) -> ConnectivityResult:
        try:
            result = await self.chat(
                ChatRequest(
                    model=model,
                    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                    options={"max_tokens": 16},
                )
            )
            return ConnectivityResult(ok=True, response_preview=result.content[:200])
        except Exception as exc:
            return ConnectivityResult(ok=False, error=str(exc))

    # -------------------------------------------------------------------------
    async def close(self) -> None:
        await self.client.close()
