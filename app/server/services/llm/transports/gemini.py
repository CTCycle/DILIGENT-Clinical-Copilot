from __future__ import annotations

import asyncio
from typing import Any

from google import genai
from google.genai import types

from domain.llm.providers import CloudModelDescriptor
from services.llm.transports.base import (
    ChatRequest,
    ChatResult,
    ConnectivityResult,
    StructuredTransportMixin,
)


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
            temperature=request.options.get("temperature"),
            response_mime_type="application/json" if request.json_mode else None,
        )
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=request.model,
            contents=contents,
            config=config,
        )
        return ChatResult(content=str(response.text or ""))

    # -------------------------------------------------------------------------
    async def list_models(self) -> list[CloudModelDescriptor]:
        page = await asyncio.to_thread(lambda: list(self.client.models.list()))
        return [
            CloudModelDescriptor(
                id=str(item.name).removeprefix("models/"),
                display_name=str(item.display_name or item.name),
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
