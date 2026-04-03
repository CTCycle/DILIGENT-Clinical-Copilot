from __future__ import annotations

import json
from typing import Any, Literal

import httpx

from DILIGENT.server.common.constants import GEMINI_API_BASE, OPENAI_API_BASE
from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.configurations import LLMRuntimeConfig, server_settings
from DILIGENT.server.models.structured import StructuredOutputParser, parse_json_dict, T
from DILIGENT.server.repositories.serialization.access_keys import AccessKeySerializer
from DILIGENT.server.services.cryptography import (
    decrypt as decrypt_access_key,
)

ProviderName = Literal["openai", "azure-openai", "anthropic", "gemini"]


###############################################################################
class LLMError(RuntimeError):
    pass


class LLMTimeout(LLMError):
    """Raised when requests exceed the configured timeout."""


###############################################################################
class CloudLLMClient:
    """
    Async client for hosted/proprietary LLMs (OpenAI, Gemini, etc.) with a
    compatible interface to `OllamaClient` for easy swapping.

    """

    def __init__(
        self,
        *,
        provider: ProviderName = "openai",
        base_url: str | None = None,
        timeout_s: float = server_settings.external_data.default_llm_timeout,
        keepalive_connections: int = 10,
        keepalive_max: int = 20,
        default_model: str | None = None,
    ) -> None:
        self.provider: ProviderName = provider
        self.default_model = default_model
        provider_access_key = self.resolve_provider_access_key(provider)

        if provider == "openai":
            if not provider_access_key:
                raise LLMError("No active OpenAI access key configured")
            self.base_url = (base_url or OPENAI_API_BASE).rstrip("/")
            headers = {
                "Authorization": f"Bearer {provider_access_key}",
                "Content-Type": "application/json",
            }
        elif provider == "gemini":
            if not provider_access_key:
                raise LLMError("No active Gemini access key configured")
            self.base_url = (base_url or GEMINI_API_BASE).rstrip("/")
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": provider_access_key,
            }
        elif provider in ("azure-openai", "anthropic"):
            raise LLMError(f"Provider '{provider}' not yet configured")
        else:
            raise LLMError(f"Unknown provider: {provider}")

        limits = httpx.Limits(
            max_keepalive_connections=keepalive_connections,
            max_connections=keepalive_max,
        )
        timeout = httpx.Timeout(timeout_s)
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            limits=limits,
            headers=headers,
        )

    # ---------------------------------------------------------------------
    def resolve_provider_access_key(self, provider: ProviderName) -> str | None:
        if provider not in {"openai", "gemini"}:
            return None

        access_key_serializer = AccessKeySerializer()
        try:
            row = access_key_serializer.get_active_key(provider, mark_used=True)
        except Exception as exc:  # noqa: BLE001
            provider_label = "OpenAI" if provider == "openai" else "Gemini"
            raise LLMError(f"Failed to load active {provider_label} access key") from exc
        if row is None:
            return None
        try:
            return decrypt_access_key(row.encrypted_value)
        except Exception as exc:  # noqa: BLE001
            provider_label = "OpenAI" if provider == "openai" else "Gemini"
            raise LLMError(f"Failed to decrypt active {provider_label} access key") from exc

    # ---------------------------------------------------------------------
    async def close(self) -> None:
        await self.client.aclose()

    # ---------------------------------------------------------------------
    async def __aenter__(self) -> CloudLLMClient:
        return self

    # ---------------------------------------------------------------------
    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ---------------------------------------------------------------------
    async def list_models(self) -> list[str]:
        if self.provider == "openai":
            try:
                resp = await self.client.get("/models")
            except httpx.TimeoutException as e:
                raise LLMTimeout("Timed out listing OpenAI models") from e
            self.raise_for_status(resp)
            data = resp.json()
            return [m["id"] for m in data.get("data", []) if "id" in m]
        return []

    # ---------------------------------------------------------------------
    async def check_model_availability(self, name: str) -> None:
        models = set(await self.list_models())
        if models and name not in models:
            raise LLMError(f"Model '{name}' not found for provider {self.provider}")

    # ---------------------------------------------------------------------
    @staticmethod
    def is_gpt5_family_model(model: str | None) -> bool:
        normalized = (model or "").strip().lower()
        return normalized.startswith("gpt-5")

    # ---------------------------------------------------------------------
    @staticmethod
    def raise_for_status(resp: httpx.Response) -> None:
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = resp.text
            raise LLMError(f"HTTP {resp.status_code}: {detail}") from e

    # ---------------------------------------------------------------------
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        format: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any] | str:
        options_payload = dict(options) if options else {}
        if "temperature" not in options_payload:
            options_payload["temperature"] = LLMRuntimeConfig.get_cloud_temperature()
        if self.provider == "openai":
            return await self.chat_openai(
                model=model,
                messages=messages,
                format=format,
                options=options_payload,
            )
        if self.provider == "gemini":
            return await self.chat_gemini(
                model=model,
                messages=messages,
                options=options_payload,
            )
        raise LLMError(f"Provider '{self.provider}' does not support chat yet")

    # ---------------------------------------------------------------------
    async def embed(
        self,
        *,
        model: str,
        input_texts: list[str],
    ) -> list[list[float]]:
        if not input_texts:
            return []

        if self.provider == "openai":
            return await self.embed_openai(model=model, input_texts=input_texts)
        if self.provider == "gemini":
            return await self.embed_gemini(model=model, input_texts=input_texts)
        raise LLMError(f"Provider '{self.provider}' does not support embeddings yet")

    # ---------------------------------------------------------------------
    async def chat_openai(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        format: str | None,
        options: dict[str, Any] | None,
    ) -> dict[str, Any] | str:
        resolved_model = model or self.default_model
        body: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "stream": False,
        }
        if options:
            supports_sampling = not self.is_gpt5_family_model(resolved_model)
            if supports_sampling and "temperature" in options:
                body["temperature"] = options["temperature"]
            if supports_sampling and "top_p" in options:
                body["top_p"] = options["top_p"]
        if format == "json":
            body["response_format"] = {"type": "json_object"}

        try:
            resp = await self.client.post("/chat/completions", json=body)
        except httpx.TimeoutException as e:
            raise LLMTimeout("Timed out waiting for OpenAI chat response") from e
        self.raise_for_status(resp)

        data = resp.json()
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get(
            "content",
            "",
        )
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
        return str(content)

    # ---------------------------------------------------------------------
    @staticmethod
    def to_gemini_contents(
        messages: list[dict[str, str]],
    ) -> tuple[list[dict[str, Any]], str | None]:
        contents: list[dict[str, Any]] = []
        system_text: str | None = None
        for message in messages:
            role = message.get("role", "user")
            text = message.get("content", "")
            if role == "system":
                if text:
                    if system_text:
                        system_text = f"{system_text}\n{text}"
                    else:
                        system_text = text
                continue
            gem_role = "user" if role == "user" else "model"
            contents.append({"role": gem_role, "parts": [{"text": text}]})
        return contents, system_text

    # ---------------------------------------------------------------------
    @staticmethod
    def resolve_gemini_model_resource(model: str | None) -> str:
        model_name = (model or "").strip()
        if not model_name:
            raise LLMError("Gemini model is required")
        if model_name.startswith("models/"):
            return model_name
        return f"models/{model_name}"

    # ---------------------------------------------------------------------
    async def chat_gemini(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any] | str:
        resolved_model = model or self.default_model
        model_resource = self.resolve_gemini_model_resource(resolved_model)
        contents, system_text = self.to_gemini_contents(messages)
        path = f"/{model_resource}:generateContent"

        body: dict[str, Any] = {"contents": contents}
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        if options and "temperature" in options:
            try:
                temperature = float(options["temperature"])
            except (TypeError, ValueError):
                temperature = 0.0
            body["generationConfig"] = {
                "temperature": max(0.0, min(2.0, temperature))
            }

        try:
            resp = await self.client.post(path, json=body)
        except httpx.TimeoutException as e:
            raise LLMTimeout("Timed out waiting for Gemini chat response") from e
        self.raise_for_status(resp)

        data = resp.json()
        try:
            content = (
                ((data.get("candidates") or [{}])[0].get("content") or {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
        except Exception:
            content = ""

        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
        return str(content)

    # ---------------------------------------------------------------------
    async def embed_openai(
        self,
        *,
        model: str,
        input_texts: list[str],
    ) -> list[list[float]]:
        body = {"model": model or self.default_model, "input": input_texts}

        try:
            resp = await self.client.post("/embeddings", json=body)
        except httpx.TimeoutException as exc:
            raise LLMTimeout("Timed out waiting for OpenAI embeddings") from exc

        self.raise_for_status(resp)

        data = resp.json()
        entries = sorted(data.get("data", []), key=lambda entry: entry.get("index", 0))
        embeddings: list[list[float]] = []
        for item in entries:
            vector = item.get("embedding", [])
            try:
                embeddings.append([float(value) for value in vector])
            except (TypeError, ValueError) as exc:
                raise LLMError("Non-numeric values found in OpenAI embeddings") from exc

        if len(embeddings) != len(input_texts):
            raise LLMError("Mismatch between OpenAI embeddings and inputs")
        return embeddings

    # ---------------------------------------------------------------------
    async def embed_gemini(
        self,
        *,
        model: str,
        input_texts: list[str],
    ) -> list[list[float]]:
        resolved_model = model or self.default_model
        model_resource = self.resolve_gemini_model_resource(resolved_model)
        requests_payload = [
            {
                "model": model_resource,
                "content": {"parts": [{"text": text}]},
            }
            for text in input_texts
        ]
        body = {"requests": requests_payload}
        path = f"/{model_resource}:batchEmbedContents"

        try:
            resp = await self.client.post(path, json=body)
        except httpx.TimeoutException as exc:
            raise LLMTimeout("Timed out waiting for Gemini embeddings") from exc

        self.raise_for_status(resp)

        data = resp.json()
        embeddings: list[list[float]] = []
        for item in data.get("embeddings", []):
            values = item.get("values") or item.get("embedding") or []
            try:
                embeddings.append([float(value) for value in values])
            except (TypeError, ValueError) as exc:
                raise LLMError("Non-numeric values found in Gemini embeddings") from exc

        if len(embeddings) != len(input_texts):
            raise LLMError("Mismatch between Gemini embeddings and inputs")
        return embeddings

    # ---------------------------------------------------------------------
    async def llm_structured_call(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        temperature: float = 0.0,
        use_json_mode: bool = True,
        max_repair_attempts: int = 2,
    ) -> T:
        parser = StructuredOutputParser(schema=schema)
        format_instructions = parser.get_format_instructions()
        resolved_model = model or (self.default_model or "")
        messages = self.build_structured_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            format_instructions=format_instructions,
        )

        raw = await self.chat(
            model=resolved_model,
            messages=messages,
            format="json" if use_json_mode else None,
            options=(
                None
                if self.is_gpt5_family_model(resolved_model)
                else {"temperature": temperature}
            ),
        )
        text = json.dumps(raw) if isinstance(raw, dict) else str(raw)
        return await self.parse_with_repairs(
            parser=parser,
            text=text,
            model=resolved_model,
            system_prompt=system_prompt,
            format_instructions=format_instructions,
            use_json_mode=use_json_mode,
            max_repair_attempts=max_repair_attempts,
        )

    # ---------------------------------------------------------------------
    @staticmethod
    def build_structured_messages(
        *,
        system_prompt: str,
        user_prompt: str,
        format_instructions: str,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": f"{system_prompt.strip()}\n\n{format_instructions}",
            },
            {"role": "user", "content": user_prompt},
        ]

    # ---------------------------------------------------------------------
    @staticmethod
    def build_repair_messages(
        *,
        system_prompt: str,
        format_instructions: str,
        text: str,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt.strip()},
            {
                "role": "user",
                "content": (
                    "The previous reply did not match the required JSON schema.\n"
                    "Follow these format instructions exactly and return ONLY a valid JSON object:\n"
                    f"{format_instructions}\n\n"
                    f"Previous reply:\n{text}"
                ),
            },
        ]

    # ---------------------------------------------------------------------
    async def parse_with_repairs(
        self,
        *,
        parser: StructuredOutputParser[T],
        text: str,
        model: str,
        system_prompt: str,
        format_instructions: str,
        use_json_mode: bool,
        max_repair_attempts: int,
    ) -> T:
        for attempt in range(max_repair_attempts + 1):
            try:
                return parser.parse(text)
            except Exception as err:
                if attempt >= max_repair_attempts:
                    logger.error(
                        "Structured parse failed after retries. Last text: %s",
                        text,
                    )
                    raise RuntimeError(f"Structured parsing failed: {err}") from err

                repair_messages = self.build_repair_messages(
                    system_prompt=system_prompt,
                    format_instructions=format_instructions,
                    text=text,
                )
                raw = await self.chat(
                    model=model,
                    messages=repair_messages,
                    format="json" if use_json_mode else None,
                    options=(
                        None
                        if self.is_gpt5_family_model(model)
                        else {"temperature": 0.0}
                    ),
                )
                text = json.dumps(raw) if isinstance(raw, dict) else str(raw)

        raise RuntimeError("No structured output produced by the model")

    # ---------------------------------------------------------------------
    @staticmethod
    def parse_json(obj_or_text: dict[str, Any] | str) -> dict[str, Any] | None:
        return parse_json_dict(obj_or_text)
