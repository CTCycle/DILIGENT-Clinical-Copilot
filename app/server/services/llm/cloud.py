from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, OpenAIError

from common.constants import GEMINI_API_BASE, OPENAI_API_BASE
from common.utils.logger import logger
from configurations.llm_configs import LLMRuntimeConfig
from configurations.startup import server_settings
from repositories.serialization.access_keys import AccessKeySerializer
from services.llm.structured import (
    StructuredOutputParser,
    T,
    parse_json_dict,
)

ProviderName = Literal["openai", "gemini"]


###############################################################################
class LLMError(RuntimeError):
    pass


###############################################################################
class LLMTimeout(LLMError):
    """Raised when requests exceed the configured timeout."""


###############################################################################
class CloudLLMClient:
    """
    Async client for hosted/proprietary LLMs (OpenAI, Gemini, etc.) that follows
    the app's shared LLM call shape.

    """

    def __init__(
        self,
        *,
        provider: ProviderName = "openai",
        base_url: str | None = None,
        timeout_s: float = server_settings.runtime.default_llm_timeout,
        keepalive_connections: int = 10,
        keepalive_max: int = 20,
        default_model: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self.provider: ProviderName = provider
        self.default_model = default_model
        self.timeout_s = float(timeout_s)
        provider_access_key = self.resolve_provider_access_key(provider)
        self.provider_access_key = provider_access_key
        self.openai_client: AsyncOpenAI | None = None
        self.gemini_client: Any | None = None

        if provider == "openai":
            if not provider_access_key:
                raise LLMError("No active OpenAI access key configured")
            self.base_url = (base_url or OPENAI_API_BASE).rstrip("/")
            headers = {
                "Authorization": f"Bearer {provider_access_key}",
                "Content-Type": "application/json",
            }
            self.openai_client = AsyncOpenAI(
                api_key=provider_access_key,
                base_url=self.base_url,
                timeout=self.timeout_s,
                max_retries=max(0, int(max_retries)),
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
        except Exception:  # noqa: BLE001
            # Some environments expose the key store in read-only mode; in that
            # case, fall back to a read-only fetch without updating last_used_at.
            try:
                row = access_key_serializer.get_active_key(provider, mark_used=False)
            except Exception as fallback_exc:  # noqa: BLE001
                provider_label = "OpenAI" if provider == "openai" else "Gemini"
                raise LLMError(
                    f"Failed to load active {provider_label} access key"
                ) from fallback_exc
        if row is None:
            return None
        try:
            return access_key_serializer.decrypt_key_row(row)
        except Exception as exc:  # noqa: BLE001
            provider_label = "OpenAI" if provider == "openai" else "Gemini"
            raise LLMError(
                f"Failed to decrypt active {provider_label} access key"
            ) from exc

    # ---------------------------------------------------------------------
    async def close(self) -> None:
        if self.openai_client is not None:
            await self.openai_client.close()
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

        try:
            if self.provider == "openai":
                return await self._chat_openai(
                    resolved_model=resolved_model,
                    format=format,
                    options=options_payload,
                    messages=messages,
                )
            if self.provider == "gemini":
                return await self._chat_gemini(
                    resolved_model=resolved_model,
                    options=options_payload,
                    messages=messages,
                    schema=None,
                    json_mode=format == "json",
                )
        except Exception as exc:  # noqa: BLE001
            raise self._map_provider_exception(exc) from exc
        raise LLMError(f"Provider '{self.provider}' does not support chat yet")

    # ---------------------------------------------------------------------
    async def _chat_openai(
        self,
        *,
        resolved_model: str,
        format: str | None,
        options: dict[str, Any] | None,
        messages: list[dict[str, str]],
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
        if format == "json":
            kwargs["text"] = {"format": {"type": "json_object"}}
        response = await self.openai_client.responses.create(**kwargs)
        return self._normalize_content(self._extract_openai_output_text(response))

    # ---------------------------------------------------------------------
    async def _chat_gemini(
        self,
        *,
        resolved_model: str,
        options: dict[str, Any] | None,
        messages: list[dict[str, str]],
        schema: type[T] | None,
        json_mode: bool,
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
        config = self._build_gemini_config(config_kwargs)
        response = await asyncio.to_thread(
            self.gemini_client.models.generate_content,
            model=resolved_model,
            contents=contents,
            config=config,
        )
        return self._normalize_content(getattr(response, "text", response))

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

    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    @staticmethod
    def _build_gemini_config(config_kwargs: dict[str, Any]) -> Any | None:
        if not config_kwargs:
            return None
        return genai_types.GenerateContentConfig(**config_kwargs)

    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    @staticmethod
    def _map_provider_exception(exc: Exception) -> LLMError:
        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, (TimeoutError, APITimeoutError)):
            return LLMTimeout("Timed out waiting for cloud chat response")
        if isinstance(exc, (httpx.TimeoutException, APIConnectionError)):
            return LLMTimeout("Timed out waiting for cloud chat response")
        timeout_error = getattr(genai_errors, "TimeoutError", None)
        if timeout_error is not None and isinstance(exc, timeout_error):
            return LLMTimeout("Timed out waiting for cloud chat response")
        if isinstance(exc, OpenAIError):
            return LLMError(f"Cloud LLM call failed: {exc}")
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
        system_with_format = f"{system_prompt.strip()}\n\n{format_instructions}"
        messages = self.build_structured_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            format_instructions=format_instructions,
        )

        if self.provider == "openai" and use_json_mode:
            try:
                return await self._structured_openai(
                    model=resolved_model,
                    system_prompt=system_with_format,
                    user_prompt=user_prompt,
                    schema=schema,
                    temperature=temperature,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "OpenAI native structured output failed; falling back to local parser: %s",
                    exc,
                )

        if self.provider == "gemini" and use_json_mode:
            try:
                raw = await self._chat_gemini(
                    resolved_model=resolved_model,
                    options=(
                        None
                        if self.is_gpt5_family_model(resolved_model)
                        else {"temperature": temperature}
                    ),
                    messages=[
                        {"role": "system", "content": system_with_format},
                        {"role": "user", "content": user_prompt},
                    ],
                    schema=schema,
                    json_mode=True,
                )
                text = json.dumps(raw) if isinstance(raw, dict) else str(raw)
                return parser.parse(text)
            except Exception as exc:  # noqa: BLE001
                raise self._map_provider_exception(exc) from exc

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
    async def _structured_openai(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        temperature: float,
    ) -> T:
        if self.openai_client is None:
            raise LLMError("OpenAI client is not configured")
        kwargs: dict[str, Any] = {
            "model": model,
            "instructions": system_prompt.strip(),
            "input": [{"role": "user", "content": user_prompt}],
            "text_format": schema,
        }
        if not self.is_gpt5_family_model(model):
            kwargs["temperature"] = float(temperature)
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

