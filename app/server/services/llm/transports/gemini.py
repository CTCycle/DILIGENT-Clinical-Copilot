from __future__ import annotations

import asyncio
from typing import Any

from google import genai
from google.genai import types

from domain.llm.providers import CloudModelDescriptor
from domain.llm.transports import ChatRequest, ChatResult, ConnectivityResult
from services.llm.transports.base import StructuredTransportMixin


###############################################################################
class GeminiTransport(StructuredTransportMixin):
    # -------------------------------------------------------------------------
    def __init__(self, *, api_key: str, timeout: float) -> None:
        self.client = genai.Client(api_key=api_key)

    # -------------------------------------------------------------------------
    async def chat(self, request: ChatRequest) -> ChatResult:
        system = "\n\n".join(
            item["content"] for item in request.messages if item.get("role") == "system"
        )
        contents: list[dict[str, Any]] = []
        for item in request.messages:
            if item.get("role") == "system":
                continue
            role = "model" if item.get("role") == "assistant" else "user"
            contents.append(
                {"role": role, "parts": [{"text": item.get("content", "")}]}
            )
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=(
                None
                if request.reasoning_level and request.reasoning_level != "off"
                else request.options.get("temperature")
            ),
            max_output_tokens=request.output_token_limit,
            response_mime_type="application/json" if request.json_mode else None,
            thinking_config=self._thinking_config(request),
        )
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=request.model,
            contents=contents,
            config=config,
        )
        return ChatResult(content=str(response.text or ""))

    # -------------------------------------------------------------------------
    @staticmethod
    def _thinking_config(request: ChatRequest) -> types.ThinkingConfig | None:
        if not request.reasoning_level or request.reasoning_parameter != "level":
            return None
        if request.reasoning_level == "off":
            return types.ThinkingConfig(thinking_budget=0)
        sdk_level = (
            types.ThinkingLevel.LOW
            if request.reasoning_level in {"low", "medium"}
            else types.ThinkingLevel.HIGH
        )
        return types.ThinkingConfig(thinking_level=sdk_level)

    # -------------------------------------------------------------------------
    async def list_models(
        self, *, force_refresh: bool = False
    ) -> list[CloudModelDescriptor]:
        del force_refresh
        page = await asyncio.to_thread(lambda: list(self.client.models.list()))
        return [
            CloudModelDescriptor(
                id=str(item.name).removeprefix("models/"),
                display_name=str(item.display_name or item.name),
                input_token_limit=getattr(item, "input_token_limit", None),
                output_token_limit=getattr(item, "output_token_limit", None),
                supports_thinking=(
                    bool(getattr(item, "thinking", None))
                    if getattr(item, "thinking", None) is not None
                    else None
                ),
                supports_temperature=(
                    bool(getattr(item, "temperature", None))
                    if getattr(item, "temperature", None) is not None
                    else None
                ),
            )
            for item in page
            if "generateContent" in (item.supported_actions or [])
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
        self.client.close()
