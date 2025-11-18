from __future__ import annotations

import asyncio
import contextlib
import ctypes
import inspect
import json
import math
import os
import re
import shutil
import sys
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, Literal, TypeAlias, TypeVar, cast

import httpx
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from DILIGENT.src.packages.configurations import ClientRuntimeConfig, configurations
from DILIGENT.src.packages.constants import (
    GEMINI_API_BASE,
    OPENAI_API_BASE,
    PARSING_MODEL_CHOICES,
)
from DILIGENT.src.packages.logger import logger
from DILIGENT.src.packages.types import extract_positive_int
from DILIGENT.src.packages.variables import env_variables

DEFAULT_LLM_TIMEOUT = configurations.external_data.default_llm_timeout
OLLAMA_HOST_DEFAULT = configurations.ollama_host_default

OPENAI_API_KEY = env_variables.get("OPENAI_API_KEY")
GEMINI_API_KEY = env_variables.get("GEMINI_API_KEY")

# Type variable for typed schema returns
T = TypeVar("T", bound=BaseModel)

ProviderName = Literal["openai", "azure-openai", "anthropic", "gemini"]
RuntimePurpose = Literal["clinical", "parser"]


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

    pull_locks: dict[str, asyncio.Lock] = {}
    pull_locks_guard: asyncio.Lock | None = None
    MODEL_CACHE_TTL = 30.0

    def __init__(
        self,
        base_url: str | None = None,
        timeout_s: float = DEFAULT_LLM_TIMEOUT,
        keepalive_connections: int = 10,
        keepalive_max: int = 20,
        default_model: str | None = None,
    ) -> None:
        self.base_url = (base_url or OLLAMA_HOST_DEFAULT).rstrip("/")
        self.default_model = (default_model or "").strip() or None
        limits = httpx.Limits(
            max_keepalive_connections=keepalive_connections,
            max_connections=keepalive_max,
        )
        timeout = httpx.Timeout(timeout_s)
        self.client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, limits=limits
        )
        self.legacy_generate = False
        self.model_cache: set[str] = set()
        self.model_cache_list: list[str] = []
        self.model_cache_expiry = 0.0
        self.model_cache_lock = asyncio.Lock()
        self.model_context_limits: dict[str, int] = {}

    # -------------------------------------------------------------------------
    async def close(self) -> None:
        await self.client.aclose()

    # -------------------------------------------------------------------------
    async def __aenter__(self) -> OllamaClient:
        return self

    # -------------------------------------------------------------------------
    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # -------------------------------------------------------------------------
    def resolve_model_name(self, name: str | None) -> str:
        candidate = (name or "").strip()
        if candidate:
            return candidate
        if self.default_model:
            return self.default_model
        raise OllamaError("Model name must be provided.")

    # -------------------------------------------------------------------------
    @classmethod
    def get_pull_guard(cls) -> asyncio.Lock:
        if cls.pull_locks_guard is None:
            cls.pull_locks_guard = asyncio.Lock()
        return cls.pull_locks_guard

    # -------------------------------------------------------------------------
    @classmethod
    async def get_model_lock(cls, name: str) -> asyncio.Lock:
        async with cls.get_pull_guard():
            lock = cls.pull_locks.get(name)
            if lock is None:
                lock = asyncio.Lock()
                cls.pull_locks[name] = lock
            return lock

    # -------------------------------------------------------------------------
    async def refresh_model_cache(self) -> set[str]:
        try:
            resp = await self.client.get("/api/tags")
        except httpx.TimeoutException as e:
            raise OllamaTimeout("Timed out listing Ollama models") from e
        except httpx.RequestError as e:  # noqa: PERF203 - convert to domain error
            raise OllamaError(f"Failed to list Ollama models: {e}") from e

        self.raise_for_status(resp)

        payload = resp.json()
        names: list[str] = [m["name"] for m in payload.get("models", []) if "name" in m]
        loop = asyncio.get_running_loop()
        async with self.model_cache_lock:
            self.model_cache = set(names)
            self.model_cache_list = names
            self.model_cache_expiry = loop.time() + self.MODEL_CACHE_TTL
        return set(names)

    # -------------------------------------------------------------------------
    async def get_cached_models(self, *, force_refresh: bool = False) -> set[str]:
        loop = asyncio.get_running_loop()
        async with self.model_cache_lock:
            cache_valid = (
                bool(self.model_cache)
                and loop.time() < self.model_cache_expiry
                and not force_refresh
            )
            if cache_valid:
                return set(self.model_cache)
        return await self.refresh_model_cache()

    # -------------------------------------------------------------------------
    def prepare_generation_parameters(
        self,
        *,
        temperature: float | None,
        think: bool | None,
        options: dict[str, Any] | None,
    ) -> tuple[float, bool, dict[str, Any] | None]:
        default_temp = ClientRuntimeConfig.get_ollama_temperature()
        if temperature is None:
            temp_value = default_temp
        else:
            try:
                temp_value = float(temperature)
            except (TypeError, ValueError):
                temp_value = default_temp
        options_payload = dict(options) if options else None
        if options_payload and "temperature" in options_payload:
            if temperature is None:
                try:
                    temp_value = float(options_payload["temperature"])
                except (TypeError, ValueError):
                    temp_value = default_temp
            options_payload.pop("temperature", None)
            if not options_payload:
                options_payload = None
        temp_value = max(0.0, min(2.0, float(temp_value)))
        if think is None:
            think_value = ClientRuntimeConfig.is_ollama_reasoning_enabled()
        else:
            think_value = bool(think)
        return round(temp_value, 2), think_value, options_payload

    # -------------------------------------------------------------------------
    @staticmethod
    def compose_payload(
        payload: dict[str, Any],
        *,
        format: str | None,
        options: dict[str, Any] | None,
        keep_alive: str | None,
    ) -> dict[str, Any]:
        if format:
            payload["format"] = format
        if options:
            payload["options"] = options
        if keep_alive:
            payload["keep_alive"] = keep_alive
        return payload

    # -------------------------------------------------------------------------
    def build_chat_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
        format: str | None,
        temperature: float,
        think: bool,
        options: dict[str, Any] | None,
        keep_alive: str | None,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "think": think,
        }
        return self.compose_payload(
            payload,
            format=format,
            options=options,
            keep_alive=keep_alive,
        )

    # -------------------------------------------------------------------------
    def build_generate_payload(
        self,
        *,
        model: str,
        prompt: str,
        stream: bool,
        format: str | None,
        temperature: float,
        think: bool,
        options: dict[str, Any] | None,
        keep_alive: str | None,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "temperature": temperature,
            "think": think,
        }
        return self.compose_payload(
            payload,
            format=format,
            options=options,
            keep_alive=keep_alive,
        )

    # -------------------------------------------------------------------------
    async def ensure_context_option(
        self,
        *,
        model: str,
        messages: list[dict[str, str]] | None,
        prompt: str | None,
        options: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if options and "num_ctx" in options:
            return options
        context_window = await self.calculate_context_window(
            model=model,
            messages=messages,
            prompt=prompt,
        )
        if not context_window:
            return options
        merged = dict(options) if options else {}
        merged.setdefault("num_ctx", context_window)
        return merged

    # -------------------------------------------------------------------------
    async def prepare_common_options(
        self,
        *,
        model: str,
        temperature: float | None,
        think: bool | None,
        options: dict[str, Any] | None,
        messages: list[dict[str, str]] | None = None,
        prompt: str | None = None,
    ) -> tuple[str, float, bool, dict[str, Any] | None]:
        resolved_model = self.resolve_model_name(model)
        await self.ensure_model_ready(resolved_model)
        temp_value, think_value, options_payload = self.prepare_generation_parameters(
            temperature=temperature,
            think=think,
            options=options,
        )
        enriched = await self.ensure_context_option(
            model=resolved_model,
            messages=messages,
            prompt=prompt,
            options=options_payload,
        )
        return resolved_model, temp_value, think_value, enriched

    # -------------------------------------------------------------------------
    async def ensure_model_ready(self, name: str) -> None:
        model = self.resolve_model_name(name)
        logger.debug("Verifying cached availability for Ollama model '%s'", model)
        available = await self.get_cached_models()
        if model in available:
            return

        lock = await self.get_model_lock(model)
        async with lock:
            available = await self.get_cached_models(force_refresh=True)
            if model in available:
                return
            logger.info("Pulling Ollama model '%s'", model)
            await self.pull(model, stream=False)
            logger.info("Completed pull for Ollama model '%s'", model)
            available = await self.get_cached_models(force_refresh=True)
            if model not in available:
                raise OllamaError(f"Model '{model}' was not found after pull completed")

    # -------------------------------------------------------------------------
    async def embed(
        self,
        *,
        model: str | None = None,
        input_texts: list[str],
    ) -> list[list[float]]:
        if not input_texts:
            return []

        resolved_model = self.resolve_model_name(model)
        await self.ensure_model_ready(resolved_model)
        embeddings: list[list[float]] = []
        for text in input_texts:
            payload = {"model": resolved_model, "prompt": text}
            try:
                resp = await self.client.post("/api/embeddings", json=payload)
            except httpx.TimeoutException as exc:
                raise OllamaTimeout("Timed out requesting Ollama embeddings") from exc
            except httpx.RequestError as exc:  # noqa: PERF203 - convert to domain error
                raise OllamaError(f"Failed to request Ollama embeddings: {exc}") from exc

            self.raise_for_status(resp)
            data = resp.json()
            vector = data.get("embedding")
            if not isinstance(vector, list):
                raise OllamaError("Invalid embedding payload returned by Ollama")
            try:
                embeddings.append([float(value) for value in vector])
            except (TypeError, ValueError) as exc:
                raise OllamaError("Non-numeric values found in Ollama embeddings") from exc

        if len(embeddings) != len(input_texts):
            raise OllamaError("Mismatch between Ollama embeddings and inputs")

        return embeddings

    @staticmethod
    def raise_for_status(resp: httpx.Response) -> None:
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = resp.text
            raise OllamaError(f"Ollama HTTP {resp.status_code}: {detail}") from e

    # -------------------------------------------------------------------------
    @staticmethod
    async def maybe_await(cb: ProgressCb | None, evt: dict[str, Any]) -> None:
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
        await self.get_cached_models(force_refresh=True)
        async with self.model_cache_lock:
            return list(self.model_cache_list)

    # -----------------------------------------------------------------------------
    @staticmethod
    def messages_to_prompt(messages: list[dict[str, str]]) -> str:
        role_map = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
        }
        parts: list[str] = []
        for message in messages:
            role = str(message.get("role", "user")).strip().lower()
            label = role_map.get(role, role.title() if role else "User")
            content = str(message.get("content", ""))
            if content:
                parts.append(f"{label}: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    # -------------------------------------------------------------------------
    async def pull(
        self,
        name: str,
        *,
        stream: bool = False,
        progress_callback: ProgressCb | None = None,
        poll_sleep_s: float = 0.0,
    ) -> None:
        """
        Pull a model by name. If stream=True, will iterate server events and optionally
        invoke progress_callback(event_dict) (sync or async).

        """
        payload = {"name": name, "stream": bool(stream)}
        completed = False
        try:
            if stream:
                async with self.client.stream("POST", "/api/pull", json=payload) as r:
                    self.raise_for_status(r)
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        try:
                            evt = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        await self.maybe_await(progress_callback, evt)
                        if str(evt.get("status", "")).lower() == "success":
                            completed = True
                            break
                        if poll_sleep_s > 0:
                            await asyncio.sleep(poll_sleep_s)
            else:
                resp = await self.client.post("/api/pull", json=payload)
                self.raise_for_status(resp)
                completed = True
        except httpx.TimeoutException as e:
            raise OllamaTimeout(f"Timed out pulling model '{name}'") from e
        if completed:
            try:
                await self.refresh_model_cache()
            except (OllamaError, OllamaTimeout) as exc:
                logger.debug("Failed to refresh Ollama model cache after pull: %s", exc)

    # -------------------------------------------------------------------------
    async def show_model(self, name: str) -> dict[str, Any]:
        payload = {"name": name}
        try:
            resp = await self.client.post("/api/show", json=payload)
        except httpx.TimeoutException as e:
            raise OllamaTimeout(f"Timed out retrieving metadata for '{name}'") from e
        except httpx.RequestError as e:  # noqa: PERF203 - convert to domain error
            raise OllamaError(f"Failed to query model '{name}': {e}") from e

        self.raise_for_status(resp)

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            raise OllamaError(f"Invalid JSON received for model '{name}'") from e

        if not isinstance(data, dict):
            raise OllamaError(f"Unexpected payload for model '{name}'")

        return data

    # -------------------------------------------------------------------------
    async def is_server_online(self) -> bool:
        try:
            resp = await self.client.get("/api/tags")
            resp.raise_for_status()
        except (httpx.RequestError, httpx.HTTPStatusError):
            return False
        return True

    # -------------------------------------------------------------------------
    async def start_server(
        self,
        *,
        wait_timeout_s: float = 15.0,
        poll_interval_s: float = 0.5,
    ) -> Literal["started", "already_running"]:
        if await self.is_server_online():
            return "already_running"

        if shutil.which("ollama") is None:
            raise OllamaError("Ollama executable not found in PATH.")

        try:
            process = await asyncio.create_subprocess_exec(
                "ollama",
                "serve",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError as e:
            raise OllamaError("Ollama executable not found.") from e
        except Exception as e:  # noqa: BLE001
            raise OllamaError(f"Failed to launch Ollama server: {e}") from e

        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_timeout_s

        while loop.time() < deadline:
            if await self.is_server_online():
                return "started"

            if process.returncode not in (None, 0):
                code = process.returncode
                raise OllamaError(f"Ollama server exited unexpectedly with code {code}")

            await asyncio.sleep(poll_interval_s)

        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        with contextlib.suppress(Exception):
            await process.wait()

        raise OllamaTimeout("Timed out waiting for Ollama server to start")

    # -------------------------------------------------------------------------
    @staticmethod
    def parse_size_to_bytes(value: Any) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return 0
            match = re.match(
                r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[kKmMgGtTpP]?i?[bB])?", cleaned
            )
            if not match:
                return 0
            number = float(match.group("num"))
            unit = (match.group("unit") or "b").lower()
            factors = {
                "b": 1,
                "kb": 1_000,
                "kib": 1_024,
                "mb": 1_000_000,
                "mib": 1_048_576,
                "gb": 1_000_000_000,
                "gib": 1_073_741_824,
                "tb": 1_000_000_000_000,
                "tib": 1_099_511_627_776,
                "pb": 1_000_000_000_000_000,
                "pib": 1_125_899_906_842_624,
            }
            factor = factors.get(unit, 1)
            return int(number * factor)
        return 0

    # -------------------------------------------------------------------------
    @staticmethod
    def get_available_memory_bytes() -> int:
        if sys.platform == "win32":

            class MemoryStatus(ctypes.Structure):
                fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
            return 0

        if hasattr(os, "sysconf"):
            try:
                page_size = os.sysconf("SC_PAGE_SIZE")
                if "SC_AVPHYS_PAGES" in os.sysconf_names:
                    pages = os.sysconf("SC_AVPHYS_PAGES")
                else:
                    pages = os.sysconf("SC_PHYS_PAGES")
                if isinstance(page_size, int) and isinstance(pages, int):
                    return page_size * pages
            except (ValueError, OSError, AttributeError):
                pass

        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            value = int(parts[1])
                            unit = parts[2].lower() if len(parts) >= 3 else "kb"
                            if unit in {"kb", "kib"}:
                                return value * 1_024
                            if unit in {"mb", "mib"}:
                                return value * 1_048_576
                            if unit in {"gb", "gib"}:
                                return value * 1_073_741_824
                            return value
        except (FileNotFoundError, PermissionError, ValueError):
            pass

        return 0

    # -------------------------------------------------------------------------
    async def check_model_availability(
        self, name: str, *, auto_pull: bool = True
    ) -> None:
        model = self.resolve_model_name(name)
        if auto_pull:
            await self.ensure_model_ready(model)
            return
        names = await self.get_cached_models(force_refresh=True)
        if model not in names:
            raise OllamaError(f"Model '{model}' not found and auto_pull=False")

    # -------------------------------------------------------------------------
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        format: str | None = None,
        temperature: float | None = None,
        think: bool | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> dict[str, Any] | str:
        """
        Non-streaming chat. Returns parsed JSON (dict) if possible, else raw string.

        """
        (
            resolved_model,
            temp_value,
            think_value,
            options_payload,
        ) = await self.prepare_common_options(
            model=model,
            temperature=temperature,
            think=think,
            options=options,
            messages=messages,
        )

        if self.legacy_generate:
            return await self.chat_via_generate(
                model=resolved_model,
                messages=messages,
                format=format,
                temperature=temp_value,
                think=think_value,
                options=options_payload,
                keep_alive=keep_alive,
            )

        body = self.build_chat_payload(
            model=resolved_model,
            messages=messages,
            stream=False,
            format=format,
            temperature=temp_value,
            think=think_value,
            options=options_payload,
            keep_alive=keep_alive,
        )

        try:
            resp = await self.client.post("/api/chat", json=body)
        except httpx.TimeoutException as e:
            raise OllamaTimeout("Timed out waiting for Ollama chat response") from e

        if resp.status_code == 404:
            await resp.aread()
            self.legacy_generate = True
            return await self.chat_via_generate(
                model=resolved_model,
                messages=messages,
                format=format,
                temperature=temp_value,
                think=think_value,
                options=options_payload,
                keep_alive=keep_alive,
            )

        self.raise_for_status(resp)

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
        temperature: float | None = None,
        think: bool | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Streamed chat. Yields each event (already JSON-decoded).
        Caller can aggregate tokens or forward server-sent chunks to a client.

        """
        (
            resolved_model,
            temp_value,
            think_value,
            options_payload,
        ) = await self.prepare_common_options(
            model=model,
            temperature=temperature,
            think=think,
            options=options,
            messages=messages,
        )

        if self.legacy_generate:
            async for evt in self.chat_stream_via_generate(
                model=resolved_model,
                messages=messages,
                format=format,
                temperature=temp_value,
                think=think_value,
                options=options_payload,
                keep_alive=keep_alive,
            ):
                yield evt
            return

        body = self.build_chat_payload(
            model=resolved_model,
            messages=messages,
            stream=True,
            format=format,
            temperature=temp_value,
            think=think_value,
            options=options_payload,
            keep_alive=keep_alive,
        )

        use_fallback = False

        try:
            async with self.client.stream("POST", "/api/chat", json=body) as r:
                if r.status_code == 404:
                    use_fallback = True
                    await r.aread()
                else:
                    self.raise_for_status(r)
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

        if use_fallback:
            self.legacy_generate = True
            async for evt in self.chat_stream_via_generate(
                model=resolved_model,
                messages=messages,
                format=format,
                temperature=temp_value,
                think=think_value,
                options=options_payload,
                keep_alive=keep_alive,
            ):
                yield evt

    # -----------------------------------------------------------------------------
    async def chat_via_generate(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        format: str | None,
        temperature: float,
        think: bool,
        options: dict[str, Any] | None,
        keep_alive: str | None,
    ) -> dict[str, Any] | str:
        prompt = self.messages_to_prompt(messages)
        resolved_model = self.resolve_model_name(model)
        options = await self.ensure_context_option(
            model=resolved_model,
            messages=None,
            prompt=prompt,
            options=options,
        )
        payload = self.build_generate_payload(
            model=resolved_model,
            prompt=prompt,
            stream=False,
            format=format,
            temperature=temperature,
            think=think,
            options=options,
            keep_alive=keep_alive,
        )

        try:
            resp = await self.client.post("/api/generate", json=payload)
        except httpx.TimeoutException as e:
            raise OllamaTimeout("Timed out waiting for Ollama generate response") from e

        self.raise_for_status(resp)
        data = resp.json()
        content = data.get("response", "")

        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
        return str(content)

    # -----------------------------------------------------------------------------
    async def chat_stream_via_generate(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        format: str | None,
        temperature: float,
        think: bool,
        options: dict[str, Any] | None,
        keep_alive: str | None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        prompt = self.messages_to_prompt(messages)
        resolved_model = self.resolve_model_name(model)
        options = await self.ensure_context_option(
            model=resolved_model,
            messages=None,
            prompt=prompt,
            options=options,
        )
        payload = self.build_generate_payload(
            model=resolved_model,
            prompt=prompt,
            stream=True,
            format=format,
            temperature=temperature,
            think=think,
            options=options,
            keep_alive=keep_alive,
        )

        try:
            async with self.client.stream("POST", "/api/generate", json=payload) as r:
                self.raise_for_status(r)
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    yield evt
        except httpx.TimeoutException as e:
            raise OllamaTimeout("Timed out during streamed generate response") from e

    # -------------------------------------------------------------------------
    @classmethod
    def extract_context_limit(cls, metadata: dict[str, Any]) -> int | None:
        if not isinstance(metadata, dict):
            return None
        containers: list[dict[str, Any]] = [metadata]
        for key in ("details", "model_info", "options"):
            block = metadata.get(key)
            if isinstance(block, dict):
                containers.append(block)
        for block in containers:
            for field in ("context_length", "context", "num_ctx", "ctx"):
                if field in block:
                    candidate = extract_positive_int(block[field])
                    if candidate:
                        return candidate
        return None

    # -------------------------------------------------------------------------
    async def get_model_context_limit(self, name: str) -> int | None:
        cached = self.model_context_limits.get(name)
        if cached is not None:
            return cached or None
        try:
            metadata = await self.show_model(name)
        except OllamaError:
            self.model_context_limits[name] = 0
            return None
        limit = self.extract_context_limit(metadata) or 0
        self.model_context_limits[name] = limit
        return limit or None

    # -------------------------------------------------------------------------
    @staticmethod
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return 0
        pieces = re.findall(r"\w+|[^\w\s]", normalized)
        approximate = max(len(pieces), math.ceil(len(normalized) / 4))
        return max(approximate, 1)

    # -------------------------------------------------------------------------
    async def calculate_context_window(
        self,
        *,
        model: str,
        messages: list[dict[str, str]] | None = None,
        prompt: str | None = None,
        min_ctx: int = 512,
        padding_tokens: int = 32,
        slack_ratio: float = 0.2,
    ) -> int | None:
        contents: list[str] = []
        if messages:
            for message in messages:
                content = message.get("content") if isinstance(message, dict) else None
                if content:
                    contents.append(str(content))
        if prompt:
            contents.append(prompt)
        if not contents:
            return None
        total_tokens = sum(self.estimate_tokens(chunk) for chunk in contents)
        if total_tokens <= 0:
            return None
        expanded = int(math.ceil(total_tokens * (1 + slack_ratio))) + padding_tokens
        target = max(min_ctx, expanded)
        limit = await self.get_model_context_limit(model)
        if limit and limit > 0:
            upper = min(limit, target)
            floor = min(limit, min_ctx)
            return max(upper, floor)
        return target

    # -------------------------------------------------------------------------
    async def collect_structured_fallbacks(self, preferred: list[str]) -> list[str]:
        available: set[str] = set()
        try:
            available = await self.get_cached_models()
        except (OllamaError, OllamaTimeout) as exc:
            logger.debug("Failed to list Ollama models for fallback: %s", exc)
            available = set()

        fallbacks: list[str] = []
        if available:
            for name in PARSING_MODEL_CHOICES:
                if (
                    name in available
                    and name not in preferred
                    and name not in fallbacks
                ):
                    fallbacks.append(name)
        else:
            for name in PARSING_MODEL_CHOICES:
                if name not in preferred and name not in fallbacks:
                    fallbacks.append(name)

        return fallbacks

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

        preferred: list[str] = []
        for candidate in (
            (model or "").strip(),
            (self.default_model or "").strip(),
            (ClientRuntimeConfig.get_parsing_model() or "").strip(),
        ):
            if candidate and candidate not in preferred:
                preferred.append(candidate)

        if not preferred:
            preferred = await self.collect_structured_fallbacks([])

        queue = preferred.copy()
        tried: set[str] = set()
        missing: list[str] = []
        last_missing_error: Exception | None = None
        fallbacks: list[str] | None = None

        while queue:
            active_model = queue.pop(0)
            if not active_model or active_model in tried:
                continue
            tried.add(active_model)

            try:
                raw = await self.chat(
                    model=active_model,
                    messages=messages,
                    format="json" if use_json_mode else None,
                    temperature=temperature,
                )
            except OllamaError as e:
                message = str(e).lower()
                if "not found" in message or "404" in message:
                    missing.append(active_model)
                    last_missing_error = e
                    if fallbacks is None:
                        fallbacks = await self.collect_structured_fallbacks(preferred)
                        preferred.extend(fallbacks)
                    for candidate in fallbacks:
                        if candidate not in tried and candidate not in queue:
                            queue.append(candidate)
                    continue
                raise RuntimeError(f"LLM call failed: {e}") from e

            text = json.dumps(raw) if isinstance(raw, dict) else str(raw)

            for attempt in range(max_repair_attempts + 1):
                try:
                    return cast(T, parser.parse(text))
                except Exception as err:
                    if attempt >= max_repair_attempts:
                        logger.error(
                            "Structured parse failed after retries. Last text: %s",
                            text,
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
                    try:
                        raw = await self.chat(
                            model=active_model,
                            messages=repair_messages,
                            format="json" if use_json_mode else None,
                            temperature=0.0,
                        )
                        text = json.dumps(raw) if isinstance(raw, dict) else str(raw)

                    except OllamaError as e:
                        raise RuntimeError(f"Repair attempt failed: {e}") from e

        if last_missing_error:
            attempted = ", ".join(missing)
            raise RuntimeError(
                "LLM call failed: no local parsing models were found. "
                f"Tried: {attempted}"
            ) from last_missing_error

        raise RuntimeError("LLM call failed: no parsing model candidates available")

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
        timeout_s: float = DEFAULT_LLM_TIMEOUT,
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
        self.client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, limits=limits, headers=headers
        )

    # ---------------------------------------------------------------------
    async def close(self) -> None:
        await self.client.aclose()

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
                resp = await self.client.get("/models")
            except httpx.TimeoutException as e:
                raise LLMTimeout("Timed out listing OpenAI models") from e
            self.raise_for_status(resp)
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
        keep_alive: str | None = None,  # unused but kept for compatibility
    ) -> dict[str, Any] | str:
        if self.provider == "openai":
            return await self.chat_openai(
                model=model, messages=messages, format=format, options=options
            )
        if self.provider == "gemini":
            return await self.chat_gemini(model=model, messages=messages)
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
            resp = await self.client.post("/chat/completions", json=body)
        except httpx.TimeoutException as e:
            raise LLMTimeout("Timed out waiting for OpenAI chat response") from e
        self.raise_for_status(resp)

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
    def to_gemini_contents(
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
    async def chat_gemini(
        self, *, model: str, messages: list[dict[str, str]]
    ) -> dict[str, Any] | str:
        contents, system_text = self.to_gemini_contents(messages)
        params = f"?key={GEMINI_API_KEY}"
        path = f"/models/{model or self.default_model}:generateContent{params}"

        body: dict[str, Any] = {"contents": contents}
        if system_text:
            body["system_instruction"] = {"parts": [{"text": system_text}]}

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
        requests_payload = [
            {"content": {"parts": [{"text": text}]}} for text in input_texts
        ]
        body = {"requests": requests_payload}
        path = f"/models/{model or self.default_model}:batchEmbedContents?key={GEMINI_API_KEY}"

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


###############################################################################
def select_llm_provider(
    provider: str = "ollama",
    **kwargs: Any,
) -> OllamaClient | CloudLLMClient:
    """Factory returning an LLM client with a unified interface.
    provider: "ollama" | "openai" | "gemini" (others raise a clear error).
    kwargs are forwarded to the underlying client constructors.

    """
    p = provider.strip().lower()
    if p == "ollama":
        return OllamaClient(
            base_url=kwargs.get("base_url"),
            timeout_s=kwargs.get("timeout_s", DEFAULT_LLM_TIMEOUT),
            keepalive_connections=kwargs.get("keepalive_connections", 10),
            keepalive_max=kwargs.get("keepalive_max", 20),
            default_model=kwargs.get("default_model"),
        )
    if p in ("openai", "gemini"):
        return CloudLLMClient(
            provider=p,  # type: ignore[arg-type]
            base_url=kwargs.get("base_url"),
            timeout_s=kwargs.get("timeout_s", DEFAULT_LLM_TIMEOUT),
            keepalive_connections=kwargs.get("keepalive_connections", 10),
            keepalive_max=kwargs.get("keepalive_max", 20),
            default_model=kwargs.get("default_model"),
        )
    raise LLMError(f"Unknown or unsupported provider: {provider}")


###############################################################################
def initialize_llm_client(
    *, purpose: RuntimePurpose = "clinical", **kwargs: Any
) -> OllamaClient | CloudLLMClient:
    kwargs.setdefault("timeout_s", DEFAULT_LLM_TIMEOUT)
    provider, default_model = ClientRuntimeConfig.resolve_provider_and_model(purpose)
    selected_model = kwargs.pop("default_model", default_model)
    return select_llm_provider(
        provider=provider,
        default_model=selected_model,
        **kwargs,
    )
