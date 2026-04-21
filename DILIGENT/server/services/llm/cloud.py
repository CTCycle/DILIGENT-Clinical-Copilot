from __future__ import annotations

import json
from typing import Any, Literal

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from DILIGENT.server.common.constants import GEMINI_API_BASE, OPENAI_API_BASE
from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.configurations.startup import server_settings
from DILIGENT.server.configurations.llm_configs import LLMRuntimeConfig
from DILIGENT.server.services.llm.structured import StructuredOutputParser, parse_json_dict, T
from DILIGENT.server.repositories.serialization.access_keys import AccessKeySerializer

ProviderName = Literal["openai", "azure-openai", "anthropic", "gemini"]


###############################################################################
class LLMError(RuntimeError):
    pass

###############################################################################
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
        self.timeout_s = float(timeout_s)
        provider_access_key = self.resolve_provider_access_key(provider)
        self.provider_access_key = provider_access_key

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
            return access_key_serializer.decrypt_key_row(row)
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
        resolved_model = model or self.default_model
        if not resolved_model:
            raise LLMError("Model is required")

        lc_messages = self._build_langchain_messages(messages)
        try:
            if self.provider == "openai":
                chat_model = self._build_openai_chat_model(
                    resolved_model=resolved_model,
                    format=format,
                    options=options_payload,
                )
            elif self.provider == "gemini":
                chat_model = self._build_gemini_chat_model(
                    resolved_model=resolved_model,
                    options=options_payload,
                )
            else:
                raise LLMError(f"Provider '{self.provider}' does not support chat yet")
            response = await chat_model.ainvoke(lc_messages)
        except Exception as exc:  # noqa: BLE001
            raise self._map_langchain_exception(exc) from exc
        return self._normalize_langchain_content(response.content)

    # ---------------------------------------------------------------------
    def _build_openai_chat_model(
        self,
        *,
        resolved_model: str,
        format: str | None,
        options: dict[str, Any] | None,
    ) -> ChatOpenAI:
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "api_key": self.provider_access_key,
            "base_url": self.base_url,
            "timeout": self.timeout_s,
        }
        supports_sampling = not self.is_gpt5_family_model(resolved_model)
        if supports_sampling and options and "temperature" in options:
            kwargs["temperature"] = float(options["temperature"])
        if supports_sampling and options and "top_p" in options:
            kwargs["top_p"] = float(options["top_p"])
        if format == "json":
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kwargs)

    # ---------------------------------------------------------------------
    def _build_gemini_chat_model(
        self,
        *,
        resolved_model: str,
        options: dict[str, Any] | None,
    ) -> ChatGoogleGenerativeAI:
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "google_api_key": self.provider_access_key,
            "timeout": self.timeout_s,
        }
        if options and "temperature" in options:
            kwargs["temperature"] = max(0.0, min(2.0, float(options["temperature"])))
        return ChatGoogleGenerativeAI(**kwargs)

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
    @staticmethod
    def _build_langchain_messages(
        messages: list[dict[str, str]],
    ) -> list[BaseMessage]:
        output: list[BaseMessage] = []
        for item in messages:
            role = str(item.get("role", "user")).strip().lower()
            content = str(item.get("content", ""))
            if role == "system":
                output.append(SystemMessage(content=content))
            elif role in {"assistant", "model"}:
                output.append(AIMessage(content=content))
            else:
                output.append(HumanMessage(content=content))
        return output

    # ---------------------------------------------------------------------
    @staticmethod
    def _normalize_langchain_content(content: Any) -> dict[str, Any] | str:
        if isinstance(content, dict):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
                    continue
                if isinstance(part, str):
                    chunks.append(part)
                    continue
                chunks.append(str(part))
            content = "".join(chunks)
        if isinstance(content, str):
            try:
                loaded = json.loads(content)
            except json.JSONDecodeError:
                return content
            return loaded if isinstance(loaded, dict) else content
        return str(content)

    # ---------------------------------------------------------------------
    @staticmethod
    def _map_langchain_exception(exc: Exception) -> LLMError:
        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, TimeoutError):
            return LLMTimeout("Timed out waiting for cloud chat response")
        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeout("Timed out waiting for cloud chat response")
        error_name = exc.__class__.__name__.lower()
        if "timeout" in error_name:
            return LLMTimeout("Timed out waiting for cloud chat response")
        return LLMError(f"Cloud LLM call failed: {exc}")

    # ---------------------------------------------------------------------
    async def llm_text_call(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        resolved_model = model or (self.default_model or "")
        raw = await self.chat(
            model=resolved_model,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt},
            ],
            options=(
                None
                if self.is_gpt5_family_model(resolved_model)
                else {"temperature": float(temperature)}
            ),
        )
        return json.dumps(raw) if isinstance(raw, dict) else str(raw)

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

