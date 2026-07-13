from __future__ import annotations

from anthropic import AsyncAnthropic

from domain.llm.providers import CloudModelDescriptor
from services.llm.transports.base import (
    ChatRequest,
    ChatResult,
    ConnectivityResult,
    StructuredTransportMixin,
)


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
        messages = [item for item in request.messages if item.get("role") != "system"]
        response = await self.client.messages.create(
            model=request.model,
            system=system,
            messages=messages,
            max_tokens=int(request.options.get("max_tokens", 4096)),
        )
        text = "".join(getattr(block, "text", "") for block in response.content)
        return ChatResult(content=text)

    # -------------------------------------------------------------------------
    async def list_models(self) -> list[CloudModelDescriptor]:
        page = await self.client.models.list()
        return [
            CloudModelDescriptor(id=item.id, display_name=item.display_name)
            for item in page.data
        ]

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
