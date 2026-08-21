from __future__ import annotations

from openai import AsyncOpenAI

from domain.llm.providers import CloudModelDescriptor
from domain.llm.transports import ChatRequest, ChatResult, ConnectivityResult
from services.llm.transports.base import StructuredTransportMixin

###############################################################################
class OpenAIResponsesTransport(StructuredTransportMixin):

    # -------------------------------------------------------------------------
    def __init__(self, *, api_key: str, base_url: str, timeout: float) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    # -------------------------------------------------------------------------
    async def chat(self, request: ChatRequest) -> ChatResult:
        instructions = "\n\n".join(
            item["content"] for item in request.messages if item.get("role") == "system"
        )
        inputs = [item for item in request.messages if item.get("role") != "system"]
        kwargs = {"model": request.model, "input": inputs, **request.options}
        if instructions:
            kwargs["instructions"] = instructions
        if request.output_token_limit is not None:
            kwargs["max_output_tokens"] = request.output_token_limit
        if request.reasoning_level and request.reasoning_level != "off":
            kwargs.pop("temperature", None)
            if request.reasoning_parameter in {"effort", "level"}:
                kwargs["reasoning"] = {"effort": request.reasoning_level}
        response = await self.client.responses.create(**kwargs)
        return ChatResult(content=response.output_text)

    # -------------------------------------------------------------------------
    async def list_models(
        self, *, force_refresh: bool = False
    ) -> list[CloudModelDescriptor]:
        del force_refresh
        page = await self.client.models.list()
        return [
            CloudModelDescriptor(id=item.id, display_name=item.id) for item in page.data
        ]

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
        await self.client.close()
