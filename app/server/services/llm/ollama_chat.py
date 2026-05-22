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
import subprocess
import time
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, Literal, NoReturn, TypeAlias

import httpx
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings

from services.llm.structured import (
    StructuredOutputParser,
    parse_json_dict,
    T,
)
from configurations.startup import server_settings
from configurations.llm_configs import LLMRuntimeConfig
from common.constants import (
    TEXT_EXTRACTION_MODEL_CHOICES,
)
from common.utils.logger import logger
from common.utils.types import extract_positive_int


ProviderName = Literal["openai", "gemini"]
RuntimePurpose = Literal["clinical", "parser"]


###############################################################################
class OllamaError(RuntimeError):
    pass


###############################################################################
class OllamaTimeout(OllamaError):
    """Raised when requests to Ollama exceed the configured timeout."""


ProgressCb: TypeAlias = Callable[[dict[str, Any]], None | Awaitable[None]]


###############################################################################
def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


###############################################################################
def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


###############################################################################
def _build_langchain_messages(messages: list[dict[str, str]]) -> list[BaseMessage]:
    output: list[BaseMessage] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        content = str(message.get("content", ""))
        if role == "system":
            output.append(SystemMessage(content=content))
        elif role in {"assistant", "model"}:
            output.append(AIMessage(content=content))
        else:
            output.append(HumanMessage(content=content))
    return output


###############################################################################
def _normalize_langchain_content(content: Any) -> dict[str, Any] | str:
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
                continue
            if isinstance(item, str):
                chunks.append(item)
                continue
            chunks.append(str(item))
        content = "".join(chunks)
    if isinstance(content, str):
        try:
            loaded = json.loads(content)
        except json.JSONDecodeError:
            return content
        return loaded if isinstance(loaded, dict) else content
    return str(content)


###############################################################################
def _map_ollama_langchain_exception(exc: Exception) -> OllamaError:
    if isinstance(exc, OllamaError):
        return exc
    if isinstance(exc, TimeoutError):
        return OllamaTimeout("Timed out waiting for Ollama response")
    if isinstance(exc, httpx.TimeoutException):
        return OllamaTimeout("Timed out waiting for Ollama response")
    error_name = exc.__class__.__name__.lower()
    if "timeout" in error_name:
        return OllamaTimeout("Timed out waiting for Ollama response")
    return OllamaError(f"Ollama request failed: {exc}")


###############################################################################

# Extracted from the facade module; functions intentionally accept the facade instance.

def resolve_model_name(self, name: str | None) -> str:
    candidate = (name or "").strip()
    if candidate:
        return candidate
    if self.default_model:
        return self.default_model
    raise OllamaError("Model name must be provided.")

def get_pull_guard(cls) -> asyncio.Lock:
    if cls.pull_locks_guard is None:
        cls.pull_locks_guard = asyncio.Lock()
    return cls.pull_locks_guard

async def get_model_lock(cls, name: str) -> asyncio.Lock:
    async with cls.get_pull_guard():
        lock = cls.pull_locks.get(name)
        if lock is None:
            lock = asyncio.Lock()
            cls.pull_locks[name] = lock
        return lock

async def refresh_model_cache(self) -> set[str]:
    try:
        resp = await self.client.get("/api/tags")
    except httpx.TimeoutException as e:
        raise OllamaTimeout("Timed out listing Ollama models") from e
    except httpx.RequestError as e:  # noqa: PERF203 - convert to domain error
        raise OllamaError(f"Failed to list Ollama models: {e}") from e

    self.raise_for_status(resp)

    payload = resp.json()
    names: list[str] = []
    sizes: dict[str, int] = {}
    vram_sizes: dict[str, int] = {}
    for raw_model in payload.get("models", []):
        if not isinstance(raw_model, dict):
            continue
        name = str(raw_model.get("name", "")).strip()
        if not name:
            continue
        names.append(name)
        model_size, model_vram = self.extract_footprint_from_payload(raw_model)
        if model_size > 0:
            sizes[name] = model_size
        if model_vram > 0:
            vram_sizes[name] = model_vram
    loop = asyncio.get_running_loop()
    async with self.model_cache_lock:
        self.model_cache = set(names)
        self.model_cache_list = names
        self.model_size_bytes = sizes
        self.model_vram_bytes = vram_sizes
        self.model_cache_expiry = loop.time() + self.MODEL_CACHE_TTL
    return set(names)

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

def prepare_generation_parameters(
    self,
    *,
    temperature: float | None,
    think: bool | None,
    options: dict[str, Any] | None,
) -> tuple[float, bool, dict[str, Any] | None]:
    temp_value, options_payload = self.resolve_temperature(temperature, options)
    temp_value = max(0.0, min(2.0, float(temp_value)))
    if think is None:
        think_value = LLMRuntimeConfig.is_ollama_reasoning_enabled()
    else:
        think_value = bool(think)
    return round(temp_value, 2), think_value, options_payload

def resolve_temperature(
    temperature: float | None, options: dict[str, Any] | None
) -> tuple[float, dict[str, Any] | None]:
    default_temp = LLMRuntimeConfig.get_ollama_temperature()
    options_payload = dict(options) if options else None
    temp_value = default_temp
    if temperature is not None:
        try:
            temp_value = float(temperature)
        except (TypeError, ValueError):
            temp_value = default_temp
    if options_payload and "temperature" in options_payload:
        if temperature is None:
            try:
                temp_value = float(options_payload["temperature"])
            except (TypeError, ValueError):
                temp_value = default_temp
        options_payload.pop("temperature", None)
        if not options_payload:
            options_payload = None
    return temp_value, options_payload

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

def _build_ollama_chat_model(
    self,
    *,
    model: str,
    format: str | None,
    temperature: float,
    think: bool,
    options: dict[str, Any] | None,
    keep_alive: str | None,
) -> ChatOllama:
    kwargs: dict[str, Any] = {
        "model": model,
        "base_url": self.base_url,
        "temperature": temperature,
        "client_kwargs": {"timeout": self.timeout_s},
    }
    if format:
        kwargs["format"] = format
    if keep_alive:
        kwargs["keep_alive"] = keep_alive
    if think:
        kwargs["reasoning"] = True

    supported_options = {
        "mirostat",
        "mirostat_eta",
        "mirostat_tau",
        "num_ctx",
        "num_gpu",
        "num_thread",
        "num_predict",
        "repeat_last_n",
        "repeat_penalty",
        "seed",
        "stop",
        "tfs_z",
        "top_k",
        "top_p",
        "min_p",
    }
    for key, value in (options or {}).items():
        if key in supported_options and key not in kwargs:
            kwargs[key] = value

    return ChatOllama(**kwargs)

def _build_ollama_embeddings_model(
    self,
    *,
    model: str,
) -> OllamaEmbeddings:
    kwargs: dict[str, Any] = {
        "model": model,
        "base_url": self.base_url,
        "client_kwargs": {"timeout": self.timeout_s},
    }
    return OllamaEmbeddings(**kwargs)

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
    embeddings_model = self._build_ollama_embeddings_model(model=resolved_model)
    try:
        vectors = await asyncio.to_thread(
            embeddings_model.embed_documents,
            input_texts,
        )
    except Exception as exc:  # noqa: BLE001
        mapped = _map_ollama_langchain_exception(exc)
        if isinstance(mapped, OllamaTimeout):
            raise OllamaTimeout("Timed out requesting Ollama embeddings") from exc
        raise mapped from exc

    embeddings: list[list[float]] = []
    for vector in vectors:
        if not isinstance(vector, list):
            raise OllamaError("Invalid embedding payload returned by Ollama")
        try:
            embeddings.append([float(value) for value in vector])
        except (TypeError, ValueError) as exc:
            raise OllamaError(
                "Non-numeric values found in Ollama embeddings"
            ) from exc
    if len(embeddings) != len(input_texts):
        raise OllamaError("Mismatch between Ollama embeddings and inputs")
    return embeddings

def raise_for_status(resp: httpx.Response) -> None:
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        detail = resp.text
        raise OllamaError(f"Ollama HTTP {resp.status_code}: {detail}") from e

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

def decode_response_content(content: Any) -> Any:
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return str(content)

async def iter_json_stream_events(
    response: httpx.Response,
) -> AsyncGenerator[dict[str, Any], None]:
    async for line in response.aiter_lines():
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        yield evt

async def list_models(self) -> list[str]:
    await self.get_cached_models(force_refresh=True)
    async with self.model_cache_lock:
        return list(self.model_cache_list)

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
    try:
        if stream:
            completed = await self.pull_stream(
                payload=payload,
                progress_callback=progress_callback,
                poll_sleep_s=poll_sleep_s,
            )
        else:
            completed = await self.pull_once(payload=payload)
    except httpx.TimeoutException as e:
        raise OllamaTimeout(f"Timed out pulling model '{name}'") from e
    await self.refresh_cache_after_pull(completed)

async def pull_stream(
    self,
    *,
    payload: dict[str, Any],
    progress_callback: ProgressCb | None,
    poll_sleep_s: float,
) -> bool:
    async with self.client.stream("POST", "/api/pull", json=payload) as r:
        self.raise_for_status(r)
        async for evt in self.iter_json_stream_events(r):
            await self.maybe_await(progress_callback, evt)
            if str(evt.get("status", "")).lower() == "success":
                return True
            if poll_sleep_s > 0:
                await asyncio.sleep(poll_sleep_s)
    return False

async def pull_once(self, *, payload: dict[str, Any]) -> bool:
    resp = await self.client.post("/api/pull", json=payload)
    self.raise_for_status(resp)
    return True

async def refresh_cache_after_pull(self, completed: bool) -> None:
    if not completed:
        return
    try:
        await self.refresh_model_cache()
    except OllamaError as exc:
        logger.debug("Failed to refresh Ollama model cache after pull: %s", exc)

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

async def is_server_online(self) -> bool:
    try:
        resp = await self.client.get("/api/tags")
        resp.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError):
        return False
    return True

async def start_server(
    self,
    *,
    wait_timeout_s: float = server_settings.runtime.ollama_server_start_timeout,
    poll_interval_s: float = server_settings.jobs.polling_interval,
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
    resolved_keep_alive = await self.resolve_policy_keep_alive(
        active_model=resolved_model,
        requested_keep_alive=keep_alive,
    )
    chat_model = self._build_ollama_chat_model(
        model=resolved_model,
        format=format,
        temperature=temp_value,
        think=think_value,
        options=options_payload,
        keep_alive=resolved_keep_alive,
    )
    lc_messages = _build_langchain_messages(messages)
    try:
        response = await chat_model.ainvoke(lc_messages)
    except Exception as exc:  # noqa: BLE001
        mapped = _map_ollama_langchain_exception(exc)
        if isinstance(mapped, OllamaTimeout):
            raise OllamaTimeout(
                "Timed out waiting for Ollama chat response"
            ) from exc
        raise mapped from exc

    content = _normalize_langchain_content(response.content)
    await self.maybe_prefetch_target_model(active_model=resolved_model)
    return content

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
    resolved_keep_alive = await self.resolve_policy_keep_alive(
        active_model=resolved_model,
        requested_keep_alive=keep_alive,
    )
    chat_model = self._build_ollama_chat_model(
        model=resolved_model,
        format=format,
        temperature=temp_value,
        think=think_value,
        options=options_payload,
        keep_alive=resolved_keep_alive,
    )
    lc_messages = _build_langchain_messages(messages)
    content_parts: list[str] = []
    try:
        async for chunk in chat_model.astream(lc_messages):
            if not isinstance(chunk, AIMessageChunk):
                normalized = _normalize_langchain_content(
                    getattr(chunk, "content", "")
                )
            else:
                normalized = _normalize_langchain_content(chunk.content)
            text = (
                json.dumps(normalized)
                if isinstance(normalized, dict)
                else str(normalized)
            )
            if not text:
                continue
            content_parts.append(text)
            yield {"message": {"role": "assistant", "content": text}, "done": False}
    except Exception as exc:  # noqa: BLE001
        mapped = _map_ollama_langchain_exception(exc)
        if isinstance(mapped, OllamaTimeout):
            raise OllamaTimeout("Timed out during streamed chat response") from exc
        raise mapped from exc
    final_content = "".join(content_parts)
    yield {"message": {"role": "assistant", "content": final_content}, "done": True}
    await self.maybe_prefetch_target_model(active_model=resolved_model)

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

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return 0
    pieces = re.findall(r"\w+|[^\w\s]", normalized)
    approximate = max(len(pieces), math.ceil(len(normalized) / 4))
    return max(approximate, 1)

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
