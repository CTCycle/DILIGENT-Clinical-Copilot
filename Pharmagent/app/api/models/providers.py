from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, Literal, TypeAlias, TypeVar, cast
import httpx
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from Pharmagent.app.logger import logger
from Pharmagent.app.constants import (
    OLLAMA_HOST_DEFAULT,
    OPENAI_API_BASE,
    GEMINI_API_BASE,
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Type variable for typed schema returns
T = TypeVar("T", bound=BaseModel)

ProviderName = Literal["openai", "azure-openai", "anthropic", "gemini"]


###############################################################################
class OllamaError(RuntimeError):
    pass


class OllamaTimeout(OllamaError):
    """Raised when requests to Ollama exceed the configured timeout."""


ProgressCb: TypeAlias = Callable[[dict[str, Any]], None | Awaitable[None]]


###############################################################################
class OllamaClient:
    """
    Async wrapper around the Ollama REST API.
      - list_models()
      - pull()
      - chat()   (non-stream, returns final content)
      - chat_stream() (yields streamed content chunks)
      - check_model_availability()

    Usage:

        async with AsyncOllamaClient() as client:
            await client.check_model_availability("llama3.1:8b")
            out = await client.chat(
                model="llama3.1:8b",
                messages=[{"role":"user","content":"Hi"}],
                format="json")

    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        keepalive_connections: int = 10,
        keepalive_max: int = 20,
    ) -> None:
        self.base_url = (base_url or OLLAMA_HOST_DEFAULT).rstrip("/")
        limits = httpx.Limits(
            max_keepalive_connections=keepalive_connections,
            max_connections=keepalive_max,
        )
        timeout = httpx.Timeout(timeout_s)
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, limits=limits
        )

    # -------------------------------------------------------------------------
    async def close(self) -> None:
        await self._client.aclose()

    # -------------------------------------------------------------------------
    async def __aenter__(self) -> OllamaClient:
        return self

    # -------------------------------------------------------------------------
    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # -------------------------------------------------------------------------
    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = resp.text
            raise OllamaError(f"Ollama HTTP {resp.status_code}: {detail}") from e

    # -------------------------------------------------------------------------
    @staticmethod
    async def _maybe_await(cb: ProgressCb | None, evt: dict[str, Any]) -> None:
        if cb is None:
            return
        try:
            res = cb(evt)
            if inspect.isawaitable(res):
                await res
        except Exception as e:  # don't break the pull loop on callback errors
            # attach minimal context; callers can log externally
            raise OllamaError(f"Progress callback failed: {e!r}") from e

    # -------------------------------------------------------------------------
    async def list_models(self) -> list[str]:
        try:
            resp = await self._client.get("/api/tags")
        except httpx.TimeoutException as e:
            raise OllamaTimeout("Timed out listing Ollama models") from e
        self._raise_for_status(resp)
        payload = resp.json()
        return [m["name"] for m in payload.get("models", []) if "name" in m]

    # -------------------------------------------------------------------------
    async def pull(
        self,
        name: str,
        *,
        stream: bool = False,
        progress_callback: ProgressCb | None = None,
        poll_sleep_s: float = 0.05,
    ) -> None:
        """
        Pull a model by name. If stream=True, will iterate server events and optionally
        invoke progress_callback(event_dict) (sync or async).

        """
        payload = {"name": name, "stream": bool(stream)}

        try:
            if stream:
                async with self._client.stream("POST", "/api/pull", json=payload) as r:
                    self._raise_for_status(r)
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        try:
                            evt = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        await self._maybe_await(progress_callback, evt)
                        # Ollama sends {"status":"success", ...} at completion
                        if str(evt.get("status", "")).lower() == "success":
                            return
                        # small cooperative pause
                        await asyncio.sleep(poll_sleep_s)
                return
            else:
                resp = await self._client.post("/api/pull", json=payload)
                self._raise_for_status(resp)
                return
        except httpx.TimeoutException as e:
            raise OllamaTimeout(f"Timed out pulling model '{name}'") from e

    # -------------------------------------------------------------------------
    async def check_model_availability(
        self, name: str, *, auto_pull: bool = True
    ) -> None:
        names = set(await self.list_models())
        if name not in names and auto_pull:
            await self.pull(name, stream=False)
        elif name not in names:
            raise OllamaError(f"Model '{name}' not found and auto_pull=False")

    # -------------------------------------------------------------------------
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        format: str | None = "json",
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> dict[str, Any] | str:
        """
        Non-streaming chat. Returns parsed JSON (dict) if possible, else raw string.

        """
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if format:
            body["format"] = format
        if options:
            body["options"] = options
        if keep_alive:
            body["keep_alive"] = keep_alive

        try:
            resp = await self._client.post("/api/chat", json=body)
        except httpx.TimeoutException as e:
            raise OllamaTimeout("Timed out waiting for Ollama chat response") from e
        self._raise_for_status(resp)

        data = resp.json()
        content = (data.get("message") or {}).get("content", "")

        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
        return str(content)

    # -------------------------------------------------------------------------
    async def chat_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        format: str | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Streamed chat. Yields each event (already JSON-decoded).
        Caller can aggregate tokens or forward server-sent chunks to a client.
        """
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if format:
            body["format"] = format
        if options:
            body["options"] = options
        if keep_alive:
            body["keep_alive"] = keep_alive

        try:
            async with self._client.stream("POST", "/api/chat", json=body) as r:
                self._raise_for_status(r)
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    yield evt
        except httpx.TimeoutException as e:
            raise OllamaTimeout("Timed out during streamed chat response") from e

    # -------------------------------------------------------------------------
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
        """
        Call your Ollama LLM and validate the response against a Pydantic schema
        using LangChain's PydanticOutputParser.

        - Injects format instructions so the LLM knows to return the expected JSON.
        - Parses & validates. If invalid, makes up to `max_repair_attempts` repair calls.
        - Returns an instance of `schema` (a Pydantic model).

        This function is LLM-agnostic beyond the Ollama client; you can reuse it
        across parsers by supplying different prompts/schemas.

        """
        parser = PydanticOutputParser(pydantic_object=schema)
        format_instructions = parser.get_format_instructions()

        messages = [
            {
                "role": "system",
                "content": f"{system_prompt.strip()}\n\n{format_instructions}",
            },
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw = await self.chat(
                model=model,
                messages=messages,
                format="json" if use_json_mode else None,
                options={"temperature": temperature},
            )

        except OllamaError as e:
            raise RuntimeError(f"LLM call failed: {e}") from e

        # Unify to text for the LC parser
        text = json.dumps(raw) if isinstance(raw, dict) else str(raw)

        # First parse attempt + bounded auto-repair loop
        for attempt in range(max_repair_attempts + 1):
            try:
                return cast(T, parser.parse(text))
            except Exception as err:
                if attempt >= max_repair_attempts:
                    # Surface original model output in logs for debugging
                    logger.error(
                        "Structured parse failed after retries. Last text: %s", text
                    )
                    raise RuntimeError(f"Structured parsing failed: {err}") from err

                # Ask the model to repair to valid JSON that matches the schema.
                repair_messages = [
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
                try:
                    raw = await self.chat(
                        model=model,
                        messages=repair_messages,
                        format="json" if use_json_mode else None,
                        options={"temperature": 0.0},
                    )
                    text = json.dumps(raw) if isinstance(raw, dict) else str(raw)

                except OllamaError as e:
                    raise RuntimeError(f"Repair attempt failed: {e}") from e

        # If execution reaches here, no valid model could be parsed and no
        # exception was raised within the loop (should be unreachable).
        raise RuntimeError("No structured output produced by the model")

    # -------------------------------------------------------------------------
    @staticmethod
    def parse_json(obj_or_text: dict[str, Any] | str) -> dict[str, Any] | None:
        """
        Robustly return a dict JSON object from either a dict or a text blob with JSON inside.

        """
        if isinstance(obj_or_text, dict):
            return obj_or_text

        if not isinstance(obj_or_text, str) or not obj_or_text.strip():
            return None

        try:
            loaded = json.loads(obj_or_text)
            return loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            pass

        # Extract first top-level JSON object (handles extra text/noise).
        m = re.search(r"\{(?:[^{}]|(?R))*\}", obj_or_text, flags=re.DOTALL)
        if not m:
            return None
        try:
            loaded = json.loads(m.group(0))
            return loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            return None


###############################################################################
class LLMError(RuntimeError):
    pass


class LLMTimeout(LLMError):
    """Raised when requests exceed the configured timeout."""


###############################################################################
def get_llm_client(
    provider: str = "ollama",
    **kwargs: Any,
) -> Any:
    """Factory returning an LLM client with a unified interface.

    provider: "ollama" | "openai" | "gemini" (others raise a clear error).
    kwargs are forwarded to the underlying client constructors.
    """
    p = provider.strip().lower()
    if p == "ollama":
        return OllamaClient(
            base_url=kwargs.get("base_url"),
            timeout_s=kwargs.get("timeout_s", 120.0),
            keepalive_connections=kwargs.get("keepalive_connections", 10),
            keepalive_max=kwargs.get("keepalive_max", 20),
        )
    if p in ("openai", "gemini"):
        return CloudLLMClient(
            provider=p,  # type: ignore[arg-type]
            base_url=kwargs.get("base_url"),
            timeout_s=kwargs.get("timeout_s", 120.0),
            keepalive_connections=kwargs.get("keepalive_connections", 10),
            keepalive_max=kwargs.get("keepalive_max", 20),
            default_model=kwargs.get("default_model"),
        )
    raise LLMError(f"Unknown or unsupported provider: {provider}")


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
        timeout_s: float = 120.0,
        keepalive_connections: int = 10,
        keepalive_max: int = 20,
        default_model: str | None = None,
    ) -> None:
        self.provider: ProviderName = provider
        self.default_model = default_model

        if provider == "openai":
            if not OPENAI_API_KEY:
                raise LLMError("OPENAI_API_KEY is not set")
            self.base_url = (base_url or OPENAI_API_BASE).rstrip("/")
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }
        elif provider == "gemini":
            if not GEMINI_API_KEY:
                raise LLMError("GEMINI_API_KEY is not set")
            self.base_url = (base_url or GEMINI_API_BASE).rstrip("/")
            headers = {"Content-Type": "application/json"}
        elif provider in ("azure-openai", "anthropic"):
            # Stub: add credentials via environment variables and default bases
            # when these providers are added.
            raise LLMError(f"Provider '{provider}' not yet configured")
        else:
            raise LLMError(f"Unknown provider: {provider}")

        limits = httpx.Limits(
            max_keepalive_connections=keepalive_connections,
            max_connections=keepalive_max,
        )
        timeout = httpx.Timeout(timeout_s)
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, limits=limits, headers=headers
        )

    # ---------------------------------------------------------------------
    async def close(self) -> None:
        await self._client.aclose()

    # ---------------------------------------------------------------------
    async def __aenter__(self) -> "CloudLLMClient":
        return self

    # ---------------------------------------------------------------------
    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ---------------------------------------------------------------------
    async def list_models(self) -> list[str]:
        if self.provider == "openai":
            try:
                resp = await self._client.get("/models")
            except httpx.TimeoutException as e:
                raise LLMTimeout("Timed out listing OpenAI models") from e
            self._raise_for_status(resp)
            data = resp.json()
            return [m["id"] for m in data.get("data", []) if "id" in m]

        # Gemini provides model list via a separate endpoint; keep minimal.
        return []

    # ---------------------------------------------------------------------
    async def check_model_availability(self, name: str) -> None:
        models = set(await self.list_models())
        if models and name not in models:
            raise LLMError(f"Model '{name}' not found for provider {self.provider}")

    # ---------------------------------------------------------------------
    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
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
        format: str | None = "json",
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,  # unused but kept for compatibility
    ) -> dict[str, Any] | str:
        if self.provider == "openai":
            return await self._chat_openai(
                model=model, messages=messages, format=format, options=options
            )
        if self.provider == "gemini":
            return await self._chat_gemini(model=model, messages=messages)
        raise LLMError(f"Provider '{self.provider}' does not support chat yet")

    # ---------------------------------------------------------------------
    async def _chat_openai(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        format: str | None,
        options: dict[str, Any] | None,
    ) -> dict[str, Any] | str:
        body: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": False,
        }
        if options:
            if "temperature" in options:
                body["temperature"] = options["temperature"]
            if "top_p" in options:
                body["top_p"] = options["top_p"]
        if format == "json":
            body["response_format"] = {"type": "json_object"}

        try:
            resp = await self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as e:
            raise LLMTimeout("Timed out waiting for OpenAI chat response") from e
        self._raise_for_status(resp)

        data = resp.json()
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get(
            "content", ""
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
    def _to_gemini_contents(
        messages: list[dict[str, str]],
    ) -> tuple[list[dict[str, Any]], str | None]:
        contents: list[dict[str, Any]] = []
        system_text: str | None = None
        for m in messages:
            role = m.get("role", "user")
            text = m.get("content", "")
            if role == "system":
                system_text = (
                    f"{(system_text + '\n') if system_text else ''}{text}"
                    if text
                    else system_text
                )
                continue
            gem_role = "user" if role == "user" else "model"
            contents.append({"role": gem_role, "parts": [{"text": text}]})
        return contents, system_text

    # ---------------------------------------------------------------------
    async def _chat_gemini(
        self, *, model: str, messages: list[dict[str, str]]
    ) -> dict[str, Any] | str:
        contents, system_text = self._to_gemini_contents(messages)
        params = f"?key={GEMINI_API_KEY}"
        path = f"/models/{model or self.default_model}:generateContent{params}"

        body: dict[str, Any] = {"contents": contents}
        if system_text:
            body["system_instruction"] = {"parts": [{"text": system_text}]}

        try:
            resp = await self._client.post(path, json=body)
        except httpx.TimeoutException as e:
            raise LLMTimeout("Timed out waiting for Gemini chat response") from e
        self._raise_for_status(resp)

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
        parser = PydanticOutputParser(pydantic_object=schema)
        format_instructions = parser.get_format_instructions()

        messages = [
            {
                "role": "system",
                "content": f"{system_prompt.strip()}\n\n{format_instructions}",
            },
            {"role": "user", "content": user_prompt},
        ]

        raw = await self.chat(
            model=model or (self.default_model or ""),
            messages=messages,
            format="json" if use_json_mode else None,
            options={"temperature": temperature},
        )

        text = json.dumps(raw) if isinstance(raw, dict) else str(raw)

        for attempt in range(max_repair_attempts + 1):
            try:
                return cast(T, parser.parse(text))
            except Exception as err:
                if attempt >= max_repair_attempts:
                    logger.error(
                        "Structured parse failed after retries. Last text: %s", text
                    )
                    raise RuntimeError(f"Structured parsing failed: {err}") from err

                repair_messages = [
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

                raw = await self.chat(
                    model=model or (self.default_model or ""),
                    messages=repair_messages,
                    format="json" if use_json_mode else None,
                    options={"temperature": 0.0},
                )
                text = json.dumps(raw) if isinstance(raw, dict) else str(raw)

        raise RuntimeError("No structured output produced by the model")

    # ---------------------------------------------------------------------
    @staticmethod
    def parse_json(obj_or_text: dict[str, Any] | str) -> dict[str, Any] | None:
        if isinstance(obj_or_text, dict):
            return obj_or_text
        if not isinstance(obj_or_text, str) or not obj_or_text.strip():
            return None
        try:
            loaded = json.loads(obj_or_text)
            return loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            return None
