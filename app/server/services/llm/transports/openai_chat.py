from __future__ import annotations

import httpx
from collections.abc import Mapping
from typing import Any

from domain.llm.providers import CloudModelDescriptor
from domain.llm.transports import ChatRequest, ChatResult, ConnectivityResult
from services.llm.transports.base import StructuredTransportMixin

###############################################################################
class OpenAIChatTransport(StructuredTransportMixin):

    # -------------------------------------------------------------------------
    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, str):
                    chunks.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
            return "".join(chunks)
        if isinstance(content, dict) and isinstance(content.get("text"), str):
            return content["text"]
        return str(content or "")

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if default_headers:
            headers.update(default_headers)
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            trust_env=False,
            headers=headers,
        )

    # -------------------------------------------------------------------------
    async def chat(self, request: ChatRequest) -> ChatResult:
        payload = {
            "model": request.model,
            "messages": request.messages,
            "stream": False,
            **{
                key: value
                for key, value in request.options.items()
                if key != "max_output_tokens"
            },
        }
        if request.output_token_limit is not None:
            payload["max_tokens"] = request.output_token_limit
        if request.reasoning_level and request.reasoning_level != "off":
            payload.pop("temperature", None)
            if request.reasoning_parameter == "boolean":
                payload["thinking"] = {"type": "enabled"}
            elif request.reasoning_parameter in {"effort", "level"}:
                payload["reasoning_effort"] = request.reasoning_level
        elif request.reasoning_parameter == "boolean":
            payload["thinking"] = {"type": "disabled"}
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = await self.client.post("chat/completions", json=payload)
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        return ChatResult(
            content=self._content_to_text(message.get("content")),
            reasoning_content=str(message.get("reasoning_content") or "") or None,
        )

    # -------------------------------------------------------------------------
    async def list_models(
        self, *, force_refresh: bool = False
    ) -> list[CloudModelDescriptor]:
        del force_refresh
        response = await self.client.get("models")
        response.raise_for_status()
        return [
            CloudModelDescriptor(
                id=item["id"], display_name=item.get("name") or item["id"]
            )
            for item in response.json().get("data", [])
        ]

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
        await self.client.aclose()
