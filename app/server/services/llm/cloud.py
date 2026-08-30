from __future__ import annotations

import hashlib
import asyncio
import json
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAIError,
)

from common.constants import GEMINI_API_BASE, OPENAI_API_BASE
from common.utils.logger import logger
from services.llm.runtime_config import LLMRuntimeConfig
from services.llm.generation_policy import GenerationPurpose
from configurations.startup import get_server_settings
from repositories.serialization.access_keys import AccessKeySerializer
from services.llm.structured import (
    StructuredOutputParser,
    T,
    parse_json_object_strict,
)
from domain.llm.providers import CloudModelDescriptor, CloudProviderId
from domain.llm.transports import ChatRequest
from services.llm.provider_registry import provider_registry
from services.llm.transports.anthropic_messages import AnthropicMessagesTransport
from services.llm.transports.base import CloudTransport
from services.llm.transports.openai_chat import OpenAIChatTransport
from services.llm.transports.routed_gateway import RoutedGatewayTransport
from services.llm.model_capabilities import EffectiveInferenceConfig

ProviderName = CloudProviderId


###############################################################################
def _list_gemini_models_sync(client: genai.Client) -> list[Any]:
    return list(client.models.list())


###############################################################################
class LLMError(RuntimeError):
    # -------------------------------------------------------------------------
    def __init__(
        self,
        message: str,
        *,
        error_code: str = "provider_error",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.retryable = bool(retryable)


###############################################################################
class LLMTimeout(LLMError):
    """Raised when requests exceed the configured timeout."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        message: str = "Timed out waiting for cloud chat response",
        *,
        error_code: str = "timeout",
        retryable: bool = True,
    ) -> None:
        super().__init__(message, error_code=error_code, retryable=retryable)


###############################################################################
def short_output_hash(output_text: str) -> str:
    return hashlib.sha256((output_text or "").encode("utf-8")).hexdigest()[:12]


###############################################################################
class CloudLLMClient:
    """
    Async client for hosted/proprietary LLMs (OpenAI, Gemini, etc.) that follows
    the app's shared LLM call shape.

    """

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        provider: ProviderName = "openai",
        base_url: str | None = None,
        timeout_s: float | None = None,
        keepalive_connections: int = 10,
        keepalive_max: int = 20,
        default_model: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self.provider: ProviderName = provider
        self.default_model = default_model
        runtime_timeout = get_server_settings().runtime.default_llm_timeout
        self.timeout_s = float(runtime_timeout if timeout_s is None else timeout_s)
        provider_access_key = self.resolve_provider_access_key(provider)
        self.provider_access_key = provider_access_key
        self.openai_client: AsyncOpenAI | None = None
        self.gemini_client: Any | None = None
        self.transport: CloudTransport | None = None

        if provider == "openai":
            if not provider_access_key:
                raise LLMError("No active OpenAI access key configured")
            self.base_url = (base_url or OPENAI_API_BASE).rstrip("/")
            headers = {
                "Authorization": f"Bearer {provider_access_key}",
                "Content-Type": "application/json",
            }
            _openai_http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_s),
                trust_env=False,
            )
            self.openai_client = AsyncOpenAI(
                api_key=provider_access_key,
                base_url=self.base_url,
                timeout=self.timeout_s,
                max_retries=max(0, int(max_retries)),
                http_client=_openai_http_client,
            )
        elif provider == "gemini":
            if not provider_access_key:
                raise LLMError("No active Gemini access key configured")
            self.base_url = (base_url or GEMINI_API_BASE).rstrip("/")
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": provider_access_key,
            }
            self.gemini_client = genai.Client(api_key=provider_access_key)
        elif provider == "deepseek":
            if not provider_access_key:
                raise LLMError("No active DeepSeek access key configured")
            self.base_url = (base_url or "https://api.deepseek.com").rstrip("/")
            headers = {"Authorization": f"Bearer {provider_access_key}"}
            self.transport = OpenAIChatTransport(
                api_key=provider_access_key,
                base_url=self.base_url,
                timeout=self.timeout_s,
            )
        elif provider == "anthropic":
            if not provider_access_key:
                raise LLMError("No active Anthropic access key configured")
            self.base_url = (base_url or "https://api.anthropic.com").rstrip("/")
            headers = {"x-api-key": provider_access_key}
            self.transport = AnthropicMessagesTransport(
                api_key=provider_access_key,
                base_url=self.base_url,
                timeout=self.timeout_s,
            )
        elif provider in {"opencode_zen", "opencode_go"}:
            if not provider_access_key:
                raise LLMError("No active OpenCode access key configured")
            definition = provider_registry.get(provider)
            self.base_url = (base_url or "https://opencode.ai").rstrip("/")
            headers = {"Authorization": f"Bearer {provider_access_key}"}
            self.transport = RoutedGatewayTransport(
                api_key=provider_access_key,
                base_url=self.base_url,
                models_path=definition.models_endpoint or "",
                timeout=self.timeout_s,
            )
        else:
            raise LLMError(f"Unknown provider: {provider}")

        limits = httpx.Limits(
            max_keepalive_connections=keepalive_connections,
            max_connections=keepalive_max,
        )
        timeout = httpx.Timeout(self.timeout_s)
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            limits=limits,
            headers=headers,
            trust_env=False,
        )

    # -------------------------------------------------------------------------
    def resolve_provider_access_key(self, provider: ProviderName) -> str | None:
        credential_scope = provider_registry.get(provider).credential_scope
        access_key_serializer = AccessKeySerializer()
        try:
            return access_key_serializer.get_active_key_value(credential_scope)
        except Exception as exc:  # noqa: BLE001
            provider_label = provider_registry.get(provider).display_name
            raise LLMError(
                f"Failed to load active {provider_label} access key"
            ) from exc

    # -------------------------------------------------------------------------
    async def close(self) -> None:
        if self.openai_client is not None:
            await self.openai_client.close()
        if self.transport is not None:
            await self.transport.close()
        await self.client.aclose()

    # -------------------------------------------------------------------------
    async def __aenter__(self) -> CloudLLMClient:
        return self

    # -------------------------------------------------------------------------
    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # -------------------------------------------------------------------------
    async def list_models(self) -> list[str]:
        return [item.id for item in await self.list_model_descriptors()]

    # -------------------------------------------------------------------------
    async def list_model_descriptors(
        self, *, force_refresh: bool = False
    ) -> list[CloudModelDescriptor]:
        if self.transport is not None:
            return await self.transport.list_models(force_refresh=force_refresh)
        if self.provider == "openai":
            try:
                resp = await self.client.get("/models")
            except httpx.TimeoutException as e:
                raise LLMTimeout("Timed out listing OpenAI models") from e
            self.raise_for_status(resp)
            data = resp.json()
            return [
                CloudModelDescriptor(
                    id=str(item["id"]),
                    display_name=str(item.get("name") or item["id"]),
                )
                for item in data.get("data", [])
                if isinstance(item, dict) and item.get("id")
            ]
        if self.provider == "gemini":
            return await self._list_gemini_model_descriptors()
        return []

    # -------------------------------------------------------------------------
    async def _list_gemini_model_descriptors(self) -> list[CloudModelDescriptor]:
        if self.gemini_client is None:
            raise LLMError("Gemini client is not configured")

        try:
            raw_models = await asyncio.to_thread(
                _list_gemini_models_sync, self.gemini_client
            )
        except Exception as exc:  # noqa: BLE001
            raise self._map_provider_exception(exc) from exc

        models: list[CloudModelDescriptor] = []
        for item in raw_models:
            name = str(getattr(item, "name", "") or "").strip()
            model_id = name.removeprefix("models/")
            actions = getattr(item, "supported_actions", None)
            if actions is None:
                actions = getattr(item, "supported_generation_methods", ())
            normalized_actions = {
                str(action).replace("_", "").lower() for action in (actions or ())
            }
            if not model_id or "generatecontent" not in normalized_actions:
                continue
            input_token_limit = self._coerce_optional_int(
                getattr(item, "input_token_limit", None)
            )
            output_token_limit = self._coerce_optional_int(
                getattr(item, "output_token_limit", None)
            )
            thinking_metadata = getattr(item, "thinking", None)
            if thinking_metadata is None:
                thinking_metadata = getattr(item, "thinking_config", None)
            supports_thinking = (
                bool(thinking_metadata)
                if thinking_metadata is not None
                else (
                    any("think" in action for action in normalized_actions)
                    if any("think" in action for action in normalized_actions)
                    else None
                )
            )
            temperature_metadata = getattr(item, "temperature", None)
            models.append(
                CloudModelDescriptor(
                    id=model_id,
                    display_name=str(getattr(item, "display_name", None) or model_id),
                    input_token_limit=input_token_limit,
                    output_token_limit=output_token_limit,
                    supports_thinking=supports_thinking,
                    supports_temperature=(
                        bool(temperature_metadata)
                        if temperature_metadata is not None
                        else None
                    ),
                )
            )
        return models

    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_optional_int(value: object) -> int | None:
        if value is None:
            return None
        try:
            parsed = int(str(value))
        except TypeError, ValueError:
            return None
        return parsed if parsed > 0 else None

    # -------------------------------------------------------------------------
    async def check_model_availability(self, name: str) -> None:
        models = set(await self.list_models())
        if models and name not in models:
            raise LLMError(f"Model '{name}' not found for provider {self.provider}")

    # -------------------------------------------------------------------------
    @staticmethod
    def is_gpt5_family_model(model: str | None) -> bool:
        normalized = (model or "").strip().lower()
        return normalized.startswith("gpt-5")

    # -------------------------------------------------------------------------
    @staticmethod
    def raise_for_status(resp: httpx.Response) -> None:
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            error_code, retryable = CloudLLMClient._http_status_error_code(
                resp.status_code
            )
            raise LLMError(
                f"Cloud provider returned HTTP {resp.status_code}",
                error_code=error_code,
                retryable=retryable,
            ) from e

    # -------------------------------------------------------------------------
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        format: str | None = None,
        options: dict[str, Any] | None = None,
        purpose: GenerationPurpose = GenerationPurpose.CLINICAL_SYNTHESIS,
        timeline_complexity: str = "moderate",
    ) -> dict[str, Any] | str:
        resolved_model = model or self.default_model
        if not resolved_model:
            raise LLMError("Model is required")
        effective = LLMRuntimeConfig.resolve_effective_inference_config(
            purpose=purpose,
            provider=self.provider,
            model=resolved_model,
            timeline_complexity=timeline_complexity,
        )
        options_payload = {
            key: value for key, value in (options or {}).items() if key != "temperature"
        }
        if effective.temperature is not None:
            options_payload["temperature"] = effective.temperature
        options_payload.setdefault("max_output_tokens", effective.output_token_limit)

        try:
            if self.transport is not None:
                result = await self.transport.chat(
                    ChatRequest(
                        model=resolved_model,
                        messages=messages,
                        options=options_payload,
                        json_mode=format == "json",
                        reasoning_level=effective.effective_reasoning_level.value,
                        reasoning_parameter=effective.reasoning_parameter,
                        reasoning_reserve=effective.reasoning_reserve,
                        output_token_limit=effective.output_token_limit,
                    )
                )
                return self._normalize_content(result.content)
            if self.provider == "openai":
                return await self._chat_openai(
                    resolved_model=resolved_model,
                    format=format,
                    options=options_payload,
                    messages=messages,
                    effective=effective,
                )
            if self.provider == "gemini":
                return await self._chat_gemini(
                    resolved_model=resolved_model,
                    options=options_payload,
                    messages=messages,
                    schema=None,
                    json_mode=format == "json",
                    effective=effective,
                )
        except Exception as exc:  # noqa: BLE001
            raise self._map_provider_exception(exc) from exc
        raise LLMError(f"Provider '{self.provider}' does not support chat yet")

    # -------------------------------------------------------------------------
    async def _chat_openai(
        self,
        *,
        resolved_model: str,
        format: str | None,
        options: dict[str, Any] | None,
        messages: list[dict[str, str]],
        effective: EffectiveInferenceConfig,
    ) -> dict[str, Any] | str:
        if self.openai_client is None:
            raise LLMError("OpenAI client is not configured")
        instructions, input_messages = self._build_openai_responses_input(messages)
        kwargs: dict[str, Any] = {"model": resolved_model, "input": input_messages}
        if instructions:
            kwargs["instructions"] = instructions
        supports_sampling = not self.is_gpt5_family_model(resolved_model)
        if supports_sampling and options and "temperature" in options:
            kwargs["temperature"] = float(options["temperature"])
        if supports_sampling and options and "top_p" in options:
            kwargs["top_p"] = float(options["top_p"])
        if effective.output_token_limit > 0:
            kwargs["max_output_tokens"] = effective.output_token_limit
        if (
            effective.effective_reasoning_level.value != "off"
            and effective.reasoning_parameter
            in {
                "effort",
                "level",
            }
        ):
            kwargs["reasoning"] = {"effort": effective.effective_reasoning_level.value}
        if format == "json":
            json_instruction = "Return the response as one valid JSON object."
            kwargs["instructions"] = (
                f"{instructions}\n\n{json_instruction}"
                if instructions
                else json_instruction
            )
            kwargs["input"] = [
                *input_messages,
                {"role": "user", "content": json_instruction},
            ]
            kwargs["text"] = {"format": {"type": "json_object"}}
        response = await self.openai_client.responses.create(**kwargs)
        return self._normalize_content(self._extract_openai_output_text(response))

    # -------------------------------------------------------------------------
    async def _chat_gemini(
        self,
        *,
        resolved_model: str,
        options: dict[str, Any] | None,
        messages: list[dict[str, str]],
        schema: type[T] | None,
        json_mode: bool,
        effective: EffectiveInferenceConfig | None = None,
    ) -> dict[str, Any] | str:
        if self.gemini_client is None:
            raise LLMError("Gemini client is not configured")
        system_instruction, contents = self._build_gemini_contents(messages)
        config_kwargs: dict[str, Any] = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if options and "temperature" in options:
            config_kwargs["temperature"] = max(
                0.0, min(2.0, float(options["temperature"]))
            )
        if json_mode or schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
        if schema is not None:
            config_kwargs["response_json_schema"] = schema.model_json_schema()
        if effective is not None:
            config_kwargs["max_output_tokens"] = effective.output_token_limit
            if effective.reasoning_parameter == "level":
                if effective.effective_reasoning_level.value == "off":
                    config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                        thinking_budget=0
                    )
                else:
                    sdk_level = (
                        genai_types.ThinkingLevel.LOW
                        if effective.effective_reasoning_level.value
                        in {"low", "medium"}
                        else genai_types.ThinkingLevel.HIGH
                    )
                    config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                        thinking_level=sdk_level
                    )
        config = self._build_gemini_config(config_kwargs)
        response = await asyncio.to_thread(
            self.gemini_client.models.generate_content,
            model=resolved_model,
            contents=contents,
            config=config,
        )
        return self._normalize_content(getattr(response, "text", response))

    # -------------------------------------------------------------------------
    @staticmethod
    def resolve_gemini_model_resource(model: str | None) -> str:
        model_name = (model or "").strip()
        if not model_name:
            raise LLMError("Gemini model is required")
        if model_name.startswith("models/"):
            return model_name
        return f"models/{model_name}"

    # -------------------------------------------------------------------------
    @staticmethod
    def _build_openai_responses_input(
        messages: list[dict[str, str]],
    ) -> tuple[str | None, list[dict[str, str]]]:
        instructions: list[str] = []
        input_messages: list[dict[str, str]] = []
        for item in messages:
            role = str(item.get("role", "user")).strip().lower()
            content = str(item.get("content", ""))
            if role == "system":
                if content.strip():
                    instructions.append(content.strip())
            elif role in {"assistant", "model"}:
                input_messages.append({"role": "assistant", "content": content})
            else:
                input_messages.append({"role": "user", "content": content})
        if not input_messages:
            input_messages.append({"role": "user", "content": ""})
        return "\n\n".join(instructions) or None, input_messages

    # -------------------------------------------------------------------------
    @staticmethod
    def _build_gemini_contents(
        messages: list[dict[str, str]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        system_instruction: list[str] = []
        contents: list[dict[str, Any]] = []
        for item in messages:
            role = str(item.get("role", "user")).strip().lower()
            content = str(item.get("content", ""))
            if role == "system":
                if content.strip():
                    system_instruction.append(content.strip())
                continue
            gemini_role = "model" if role in {"assistant", "model"} else "user"
            contents.append({"role": gemini_role, "parts": [{"text": content}]})
        if not contents:
            contents.append({"role": "user", "parts": [{"text": ""}]})
        return "\n\n".join(system_instruction) or None, contents

    # -------------------------------------------------------------------------
    @staticmethod
    def _build_gemini_config(config_kwargs: dict[str, Any]) -> Any | None:
        if not config_kwargs:
            return None
        return genai_types.GenerateContentConfig(**config_kwargs)

    # -------------------------------------------------------------------------
    @staticmethod
    def _extract_openai_output_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text
        output = getattr(response, "output", None)
        if isinstance(output, list):
            chunks: list[str] = []
            for item in output:
                content = getattr(item, "content", None)
                if not isinstance(content, list):
                    continue
                for part in content:
                    text = getattr(part, "text", None)
                    if isinstance(text, str):
                        chunks.append(text)
            if chunks:
                return "".join(chunks)
        return str(response)

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_content(content: Any) -> dict[str, Any] | str:
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

    # -------------------------------------------------------------------------
    @staticmethod
    def _http_status_error_code(status_code: int) -> tuple[str, bool]:
        if status_code in {401, 403}:
            return "authentication", False
        if status_code == 404:
            return "configuration", False
        if status_code == 408:
            return "timeout", True
        if status_code == 429:
            return "rate_limited", True
        if 500 <= status_code <= 599:
            return "upstream_error", True
        return "provider_error", False

    # -------------------------------------------------------------------------
    @staticmethod
    def _map_provider_exception(exc: Exception) -> LLMError:
        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, (TimeoutError, APITimeoutError)):
            return LLMTimeout("Timed out waiting for cloud chat response")
        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeout("Timed out waiting for cloud chat response")
        if isinstance(exc, (httpx.NetworkError, APIConnectionError)):
            return LLMError(
                "Cloud provider connection failed",
                error_code="network_unavailable",
                retryable=True,
            )
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code
            error_code, retryable = CloudLLMClient._http_status_error_code(status_code)
            return LLMError(
                f"Cloud provider returned HTTP {status_code}",
                error_code=error_code,
                retryable=retryable,
            )
        if isinstance(exc, APIStatusError):
            status_code = getattr(exc, "status_code", None)
            if not isinstance(status_code, int):
                response = getattr(exc, "response", None)
                status_code = getattr(response, "status_code", None)
            if isinstance(status_code, int):
                error_code, retryable = CloudLLMClient._http_status_error_code(
                    status_code
                )
                return LLMError(
                    f"Cloud provider returned HTTP {status_code}",
                    error_code=error_code,
                    retryable=retryable,
                )
        timeout_error = getattr(genai_errors, "TimeoutError", None)
        if timeout_error is not None and isinstance(exc, timeout_error):
            return LLMTimeout("Timed out waiting for cloud chat response")
        if isinstance(exc, OpenAIError):
            return LLMError(f"Cloud LLM call failed: {exc}")
        error_name = exc.__class__.__name__.lower()
        if "timeout" in error_name:
            return LLMTimeout("Timed out waiting for cloud chat response")
        return LLMError(f"Cloud LLM call failed: {exc}")

    # -------------------------------------------------------------------------
    async def llm_text_call(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        purpose: GenerationPurpose = GenerationPurpose.CLINICAL_SYNTHESIS,
    ) -> str:
        resolved_model = model or (self.default_model or "")
        raw = await self.chat(
            model=resolved_model,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt},
            ],
            purpose=purpose,
        )
        return json.dumps(raw) if isinstance(raw, dict) else str(raw)

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    async def llm_structured_call(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        purpose: GenerationPurpose = GenerationPurpose.STRUCTURED_EXTRACTION,
        use_json_mode: bool = True,
        max_repair_attempts: int = 2,
        timeline_complexity: str = "moderate",
    ) -> T:
        parser = StructuredOutputParser(schema=schema)
        format_instructions = parser.get_format_instructions()
        resolved_model = model or (self.default_model or "")
        messages = self.build_structured_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            format_instructions=format_instructions,
        )

        if self.provider == "openai" and use_json_mode:
            try:
                effective = LLMRuntimeConfig.resolve_effective_inference_config(
                    purpose=purpose,
                    provider=self.provider,
                    model=resolved_model,
                    timeline_complexity=timeline_complexity,
                )
                return await self._structured_openai(
                    model=resolved_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=schema,
                    effective=effective,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "OpenAI native structured output failed; falling back to local parser: %s",
                    exc,
                )

        if self.provider == "gemini" and use_json_mode:
            try:
                effective = LLMRuntimeConfig.resolve_effective_inference_config(
                    purpose=purpose,
                    provider=self.provider,
                    model=resolved_model,
                    timeline_complexity=timeline_complexity,
                )
                raw = await self._chat_gemini(
                    resolved_model=resolved_model,
                    options=None,
                    messages=[
                        {"role": "system", "content": system_prompt.strip()},
                        {"role": "user", "content": user_prompt},
                    ],
                    schema=schema,
                    json_mode=True,
                    effective=effective,
                )
                text = json.dumps(raw) if isinstance(raw, dict) else str(raw)
                return parser.parse(text)
            except Exception as exc:  # noqa: BLE001
                raise self._map_provider_exception(exc) from exc

        raw = await self.chat(
            model=resolved_model,
            messages=messages,
            format="json" if use_json_mode else None,
            options=None,
            purpose=purpose,
            timeline_complexity=timeline_complexity,
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

    # -------------------------------------------------------------------------
    async def _structured_openai(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        effective: EffectiveInferenceConfig,
    ) -> T:
        if self.openai_client is None:
            raise LLMError("OpenAI client is not configured")
        kwargs: dict[str, Any] = {
            "model": model,
            "instructions": system_prompt.strip(),
            "input": [{"role": "user", "content": user_prompt}],
            "text_format": schema,
        }
        if effective.temperature is not None:
            kwargs["temperature"] = effective.temperature
        kwargs["max_output_tokens"] = effective.output_token_limit
        if (
            effective.effective_reasoning_level.value != "off"
            and effective.reasoning_parameter
            in {
                "effort",
                "level",
            }
        ):
            kwargs["reasoning"] = {"effort": effective.effective_reasoning_level.value}
        try:
            response = await self.openai_client.responses.parse(**kwargs)
        except Exception as exc:  # noqa: BLE001
            mapped = self._map_provider_exception(exc)
            if isinstance(mapped, LLMTimeout) or isinstance(exc, OpenAIError):
                raise mapped from exc
            raise
        parsed = getattr(response, "output_parsed", None)
        if isinstance(parsed, schema):
            return parsed
        if parsed is not None:
            return schema.model_validate(parsed)
        text = self._extract_openai_output_text(response)
        return StructuredOutputParser(schema=schema).parse(text)

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
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
                        "Structured parse failed after retries: schema=%s attempts=%s output_length=%s output_hash=%s error=%s",
                        parser.schema.__name__,
                        attempt + 1,
                        len(text or ""),
                        short_output_hash(text or ""),
                        type(err).__name__,
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
                    purpose=GenerationPurpose.JSON_REPAIR,
                )
                text = json.dumps(raw) if isinstance(raw, dict) else str(raw)

        raise RuntimeError("No structured output produced by the model")

    # -------------------------------------------------------------------------
    @staticmethod
    def parse_json(obj_or_text: dict[str, Any] | str) -> dict[str, Any] | None:
        if isinstance(obj_or_text, dict):
            return obj_or_text
        try:
            return parse_json_object_strict(obj_or_text)
        except ValueError:
            return None
