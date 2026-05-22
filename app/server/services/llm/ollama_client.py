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
from services.llm import ollama_chat, ollama_residency, ollama_structured

ollama_chat.OllamaError = OllamaError
ollama_chat.OllamaTimeout = OllamaTimeout
ollama_chat._map_ollama_langchain_exception = _map_ollama_langchain_exception
ollama_structured.OllamaError = OllamaError

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
    RESIDENCY_PLAN_TTL = 20.0
    DEFAULT_MODEL_FOOTPRINT_BYTES = 4 * 1_073_741_824

    def __init__(
        self,
        base_url: str | None = None,
        timeout_s: float = server_settings.runtime.default_llm_timeout,
        keepalive_connections: int = 10,
        keepalive_max: int = 20,
        default_model: str | None = None,
    ) -> None:
        self.base_url = (
            base_url or server_settings.llm_defaults.ollama_host_default
        ).rstrip("/")
        self.default_model = (default_model or "").strip() or None
        self.timeout_s = float(timeout_s)
        limits = httpx.Limits(
            max_keepalive_connections=keepalive_connections,
            max_connections=keepalive_max,
        )
        timeout = httpx.Timeout(timeout_s)
        self.client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, limits=limits
        )
        self.model_cache: set[str] = set()
        self.model_cache_list: list[str] = []
        self.model_size_bytes: dict[str, int] = {}
        self.model_vram_bytes: dict[str, int] = {}
        self.model_cache_expiry = 0.0
        self.model_cache_lock = asyncio.Lock()
        self.model_context_limits: dict[str, int] = {}
        self.residency_lock = asyncio.Lock()
        self.residency_plan_cache: dict[str, Any] | None = None
        self.residency_plan_cache_expiry = 0.0
        self.residency_usage_window_s = max(
            _env_float("OLLAMA_PREFETCH_USAGE_WINDOW_S", 120.0),
            30.0,
        )
        self.residency_transition_window_s = max(
            _env_float("OLLAMA_PREFETCH_TRANSITION_WINDOW_S", 60.0),
            5.0,
        )
        self.residency_prefetch_cooldown_s = max(
            _env_float("OLLAMA_PREFETCH_COOLDOWN_S", 20.0),
            1.0,
        )
        self.residency_ram_safety_ratio = max(
            _env_float("OLLAMA_RAM_SAFETY_RATIO", 0.75),
            0.1,
        )
        self.residency_vram_safety_ratio = max(
            _env_float("OLLAMA_VRAM_SAFETY_RATIO", 0.85),
            0.1,
        )
        self.residency_dual_keep_alive = _env_str(
            "OLLAMA_DUAL_RESIDENT_KEEP_ALIVE", "4h"
        )
        self.residency_single_keep_alive = _env_str(
            "OLLAMA_SINGLE_RESIDENT_KEEP_ALIVE", "30m"
        )
        self.residency_usage_history: deque[tuple[float, str]] = deque(maxlen=256)
        self.prefetch_last_run_by_model: dict[str, float] = {}
        self.prefetch_tasks: dict[str, asyncio.Task[None]] = {}

    # -------------------------------------------------------------------------
    async def close(self) -> None:
        for task in self.prefetch_tasks.values():
            if task.done():
                continue
            task.cancel()
        await self.client.aclose()

    # -------------------------------------------------------------------------
    async def __aenter__(self) -> OllamaClient:
        return self

    # -------------------------------------------------------------------------
    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # -------------------------------------------------------------------------
    def resolve_model_name(self, name: str | None) -> str:
        return ollama_chat.resolve_model_name(self, name)

    # -------------------------------------------------------------------------
    @classmethod
    def get_pull_guard(cls) -> asyncio.Lock:
        return ollama_chat.get_pull_guard(cls)

    # -------------------------------------------------------------------------
    @classmethod
    async def get_model_lock(cls, name: str) -> asyncio.Lock:
        return await ollama_chat.get_model_lock(cls, name)

    # -------------------------------------------------------------------------
    async def refresh_model_cache(self) -> set[str]:
        return await ollama_chat.refresh_model_cache(self)

    # -------------------------------------------------------------------------
    async def get_cached_models(self, *, force_refresh: bool = False) -> set[str]:
        return await ollama_chat.get_cached_models(self, force_refresh=force_refresh)

    # -------------------------------------------------------------------------
    @staticmethod
    def get_residency_targets() -> dict[str, str]:
        return ollama_residency.get_residency_targets()

    # -------------------------------------------------------------------------
    @staticmethod
    def dedupe_models(models: list[str]) -> list[str]:
        return ollama_residency.dedupe_models(models)

    # -------------------------------------------------------------------------
    @classmethod
    def extract_bytes_from_fields(
        cls,
        payload: dict[str, Any],
        *,
        fields: tuple[str, ...],
    ) -> int:
        return ollama_residency.extract_bytes_from_fields(cls, payload, fields=fields)

    # -------------------------------------------------------------------------
    @classmethod
    def extract_footprint_from_payload(
        cls,
        payload: dict[str, Any],
    ) -> tuple[int, int]:
        return ollama_residency.extract_footprint_from_payload(cls, payload)

    # -------------------------------------------------------------------------
    async def list_running_models(self) -> dict[str, dict[str, Any]]:
        return await ollama_residency.list_running_models(self)

    # -------------------------------------------------------------------------
    async def get_model_footprint_bytes(
        self,
        model: str,
        *,
        running_models: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[int, int]:
        return await ollama_residency.get_model_footprint_bytes(self, model, running_models=running_models)

    # -------------------------------------------------------------------------
    async def evaluate_dual_residency_plan(self) -> dict[str, Any]:
        return await ollama_residency.evaluate_dual_residency_plan(self)

    # -------------------------------------------------------------------------
    async def get_cached_residency_plan(
        self,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        return await ollama_residency.get_cached_residency_plan(self, force_refresh=force_refresh)

    # -------------------------------------------------------------------------
    async def resolve_policy_keep_alive(
        self,
        *,
        active_model: str,
        requested_keep_alive: str | None,
    ) -> str | None:
        return await ollama_residency.resolve_policy_keep_alive(self, active_model=active_model, requested_keep_alive=requested_keep_alive)

    # -------------------------------------------------------------------------
    def record_target_usage(self, model: str) -> None:
        return ollama_residency.record_target_usage(self, model)

    # -------------------------------------------------------------------------
    def predict_next_target_model(
        self,
        *,
        current_model: str,
        target_models: list[str],
    ) -> str | None:
        return ollama_residency.predict_next_target_model(self, current_model=current_model, target_models=target_models)

    def _recent_residency_history(
        self, candidates: list[str]
    ) -> list[tuple[float, str]]:
        return ollama_residency._recent_residency_history(self, candidates)

    @staticmethod
    def _count_residency_frequency(
        history: list[tuple[float, str]],
        frequency: dict[str, int],
    ) -> None:
        return ollama_residency._count_residency_frequency(history, frequency)

    def _count_residency_transitions(
        self,
        history: list[tuple[float, str]],
    ) -> dict[tuple[str, str], int]:
        return ollama_residency._count_residency_transitions(self, history)

    @staticmethod
    def _select_target_model(
        *,
        current_model: str,
        candidates: list[str],
        history: list[tuple[float, str]],
        frequency: dict[str, int],
        transitions: dict[tuple[str, str], int],
    ) -> str | None:
        return ollama_residency._select_target_model(current_model=current_model, candidates=candidates, history=history, frequency=frequency, transitions=transitions)

    # -------------------------------------------------------------------------
    def handle_prefetch_task_done(self, task: asyncio.Task[None]) -> None:
        return ollama_residency.handle_prefetch_task_done(self, task)

    # -------------------------------------------------------------------------
    async def prefetch_model(
        self,
        *,
        model: str,
        keep_alive: str,
    ) -> None:
        return await ollama_residency.prefetch_model(self, model=model, keep_alive=keep_alive)

    # -------------------------------------------------------------------------
    async def maybe_prefetch_target_model(self, *, active_model: str) -> None:
        return await ollama_residency.maybe_prefetch_target_model(self, active_model=active_model)

    # -------------------------------------------------------------------------
    def prepare_generation_parameters(
        self,
        *,
        temperature: float | None,
        think: bool | None,
        options: dict[str, Any] | None,
    ) -> tuple[float, bool, dict[str, Any] | None]:
        return ollama_chat.prepare_generation_parameters(self, temperature=temperature, think=think, options=options)

    # -------------------------------------------------------------------------
    @staticmethod
    def resolve_temperature(
        temperature: float | None, options: dict[str, Any] | None
    ) -> tuple[float, dict[str, Any] | None]:
        return ollama_chat.resolve_temperature(temperature, options)

    # -------------------------------------------------------------------------
    @staticmethod
    def compose_payload(
        payload: dict[str, Any],
        *,
        format: str | None,
        options: dict[str, Any] | None,
        keep_alive: str | None,
    ) -> dict[str, Any]:
        return ollama_chat.compose_payload(payload, format=format, options=options, keep_alive=keep_alive)

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
        return ollama_chat.build_chat_payload(self, model=model, messages=messages, stream=stream, format=format, temperature=temperature, think=think, options=options, keep_alive=keep_alive)

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
        return ollama_chat.build_generate_payload(self, model=model, prompt=prompt, stream=stream, format=format, temperature=temperature, think=think, options=options, keep_alive=keep_alive)

    # -------------------------------------------------------------------------
    async def ensure_context_option(
        self,
        *,
        model: str,
        messages: list[dict[str, str]] | None,
        prompt: str | None,
        options: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        return await ollama_chat.ensure_context_option(self, model=model, messages=messages, prompt=prompt, options=options)

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
        return await ollama_chat.prepare_common_options(self, model=model, temperature=temperature, think=think, options=options, messages=messages, prompt=prompt)

    # -------------------------------------------------------------------------
    async def ensure_model_ready(self, name: str) -> None:
        return await ollama_chat.ensure_model_ready(self, name)

    # -------------------------------------------------------------------------
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
        return ollama_chat._build_ollama_chat_model(self, model=model, format=format, temperature=temperature, think=think, options=options, keep_alive=keep_alive)

    # -------------------------------------------------------------------------
    def _build_ollama_embeddings_model(
        self,
        *,
        model: str,
    ) -> OllamaEmbeddings:
        return ollama_chat._build_ollama_embeddings_model(self, model=model)

    # -------------------------------------------------------------------------
    async def embed(
        self,
        *,
        model: str | None = None,
        input_texts: list[str],
    ) -> list[list[float]]:
        return await ollama_chat.embed(self, model=model, input_texts=input_texts)

    # -------------------------------------------------------------------------
    @staticmethod
    def raise_for_status(resp: httpx.Response) -> None:
        return ollama_chat.raise_for_status(resp)

    # -------------------------------------------------------------------------
    @staticmethod
    async def maybe_await(cb: ProgressCb | None, evt: dict[str, Any]) -> None:
        return await ollama_chat.maybe_await(cb, evt)

    # -------------------------------------------------------------------------
    @staticmethod
    def decode_response_content(content: Any) -> Any:
        return ollama_chat.decode_response_content(content)

    # -------------------------------------------------------------------------
    @staticmethod
    async def iter_json_stream_events(
        response: httpx.Response,
    ) -> AsyncGenerator[dict[str, Any], None]:
        return await ollama_chat.iter_json_stream_events(response)

    # -------------------------------------------------------------------------
    async def list_models(self) -> list[str]:
        return await ollama_chat.list_models(self)

    # -----------------------------------------------------------------------------
    @staticmethod
    def messages_to_prompt(messages: list[dict[str, str]]) -> str:
        return ollama_chat.messages_to_prompt(messages)

    # -------------------------------------------------------------------------
    async def pull(
        self,
        name: str,
        *,
        stream: bool = False,
        progress_callback: ProgressCb | None = None,
        poll_sleep_s: float = 0.0,
    ) -> None:
        return await ollama_chat.pull(self, name, stream=stream, progress_callback=progress_callback, poll_sleep_s=poll_sleep_s)

    # -------------------------------------------------------------------------
    async def pull_stream(
        self,
        *,
        payload: dict[str, Any],
        progress_callback: ProgressCb | None,
        poll_sleep_s: float,
    ) -> bool:
        return await ollama_chat.pull_stream(self, payload=payload, progress_callback=progress_callback, poll_sleep_s=poll_sleep_s)

    # -------------------------------------------------------------------------
    async def pull_once(self, *, payload: dict[str, Any]) -> bool:
        return await ollama_chat.pull_once(self, payload=payload)

    # -------------------------------------------------------------------------
    async def refresh_cache_after_pull(self, completed: bool) -> None:
        return await ollama_chat.refresh_cache_after_pull(self, completed)

    # -------------------------------------------------------------------------
    async def show_model(self, name: str) -> dict[str, Any]:
        return await ollama_chat.show_model(self, name)

    # -------------------------------------------------------------------------
    async def is_server_online(self) -> bool:
        return await ollama_chat.is_server_online(self)

    # -------------------------------------------------------------------------
    async def start_server(
        self,
        *,
        wait_timeout_s: float = server_settings.runtime.ollama_server_start_timeout,
        poll_interval_s: float = server_settings.jobs.polling_interval,
    ) -> Literal["started", "already_running"]:
        return await ollama_chat.start_server(self, wait_timeout_s=wait_timeout_s, poll_interval_s=poll_interval_s)

    # -------------------------------------------------------------------------
    @staticmethod
    def parse_size_to_bytes(value: Any) -> int:
        return ollama_residency.parse_size_to_bytes(value)

    # -------------------------------------------------------------------------
    @staticmethod
    def get_available_memory_bytes() -> int:
        return ollama_residency.get_available_memory_bytes()

    # -------------------------------------------------------------------------
    @staticmethod
    def get_available_vram_bytes() -> int:
        return ollama_residency.get_available_vram_bytes()

    # -------------------------------------------------------------------------
    @staticmethod
    def _get_available_vram_nvidia_smi() -> int:
        return ollama_residency._get_available_vram_nvidia_smi()

    # -------------------------------------------------------------------------
    @staticmethod
    def _get_available_memory_windows() -> int:
        return ollama_residency._get_available_memory_windows()

    # -------------------------------------------------------------------------
    @staticmethod
    def _get_available_memory_sysconf() -> int:
        return ollama_residency._get_available_memory_sysconf()

    # -------------------------------------------------------------------------
    @staticmethod
    def _parse_meminfo_line(line: str) -> int | None:
        return ollama_residency._parse_meminfo_line(line)

    # -------------------------------------------------------------------------
    @staticmethod
    def _get_available_memory_proc() -> int:
        return ollama_residency._get_available_memory_proc()

    # -------------------------------------------------------------------------
    async def check_model_availability(
        self, name: str, *, auto_pull: bool = True
    ) -> None:
        return await ollama_chat.check_model_availability(self, name, auto_pull=auto_pull)

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
        return await ollama_chat.chat(self, model=model, messages=messages, format=format, temperature=temperature, think=think, options=options, keep_alive=keep_alive)

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
        async for event in ollama_chat.chat_stream(
            self,
            model=model,
            messages=messages,
            format=format,
            temperature=temperature,
            think=think,
            options=options,
            keep_alive=keep_alive,
        ):
            yield event

    # -------------------------------------------------------------------------
    @classmethod
    def extract_context_limit(cls, metadata: dict[str, Any]) -> int | None:
        return ollama_chat.extract_context_limit(cls, metadata)

    # -------------------------------------------------------------------------
    async def get_model_context_limit(self, name: str) -> int | None:
        return await ollama_chat.get_model_context_limit(self, name)

    # -------------------------------------------------------------------------
    @staticmethod
    def estimate_tokens(text: str) -> int:
        return ollama_chat.estimate_tokens(text)

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
        return await ollama_chat.calculate_context_window(self, model=model, messages=messages, prompt=prompt, min_ctx=min_ctx, padding_tokens=padding_tokens, slack_ratio=slack_ratio)

    # -------------------------------------------------------------------------
    async def collect_structured_fallbacks(self, preferred: list[str]) -> list[str]:
        return await ollama_structured.collect_structured_fallbacks(self, preferred)

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
        return await ollama_structured.llm_structured_call(self, model=model, system_prompt=system_prompt, user_prompt=user_prompt, schema=schema, temperature=temperature, use_json_mode=use_json_mode, max_repair_attempts=max_repair_attempts)

    # -------------------------------------------------------------------------
    @staticmethod
    def build_structured_messages(
        *,
        system_prompt: str,
        user_prompt: str,
        format_instructions: str,
    ) -> list[dict[str, str]]:
        return ollama_structured.build_structured_messages(system_prompt=system_prompt, user_prompt=user_prompt, format_instructions=format_instructions)

    # -------------------------------------------------------------------------
    async def resolve_text_extraction_models(self, model: str) -> list[str]:
        return await ollama_structured.resolve_text_extraction_models(self, model)

    # -------------------------------------------------------------------------
    @staticmethod
    def is_missing_model_error(err: OllamaError) -> bool:
        return ollama_structured.is_missing_model_error(err)

    # -------------------------------------------------------------------------
    async def _chat_structured_model(
        self,
        *,
        active_model: str,
        messages: list[dict[str, str]],
        use_json_mode: bool,
        temperature: float,
    ) -> dict[str, Any] | str:
        return await ollama_structured._chat_structured_model(self, active_model=active_model, messages=messages, use_json_mode=use_json_mode, temperature=temperature)

    # -------------------------------------------------------------------------
    async def _extend_structured_model_queue(
        self,
        *,
        queue: list[str],
        preferred_models: list[str],
        tried: set[str],
        fallbacks: list[str] | None,
    ) -> list[str]:
        return await ollama_structured._extend_structured_model_queue(self, queue=queue, preferred_models=preferred_models, tried=tried, fallbacks=fallbacks)

    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_llm_text(raw: dict[str, Any] | str) -> str:
        return ollama_structured._coerce_llm_text(raw)

    # -------------------------------------------------------------------------
    @staticmethod
    def _raise_structured_models_exhausted(
        *,
        last_missing_error: Exception | None,
        missing: list[str],
    ) -> NoReturn:
        return ollama_structured._raise_structured_models_exhausted(last_missing_error=last_missing_error, missing=missing)

    # -------------------------------------------------------------------------
    @staticmethod
    def build_repair_messages(
        *,
        system_prompt: str,
        format_instructions: str,
        text: str,
    ) -> list[dict[str, str]]:
        return ollama_structured.build_repair_messages(system_prompt=system_prompt, format_instructions=format_instructions, text=text)

    # -------------------------------------------------------------------------
    async def call_with_structured_models(
        self,
        *,
        parser: StructuredOutputParser[T],
        messages: list[dict[str, str]],
        system_prompt: str,
        format_instructions: str,
        preferred: list[str],
        temperature: float,
        use_json_mode: bool,
        max_repair_attempts: int,
    ) -> T:
        return await ollama_structured.call_with_structured_models(self, parser=parser, messages=messages, system_prompt=system_prompt, format_instructions=format_instructions, preferred=preferred, temperature=temperature, use_json_mode=use_json_mode, max_repair_attempts=max_repair_attempts)

    # -------------------------------------------------------------------------
    async def parse_with_repairs(
        self,
        *,
        parser: StructuredOutputParser[T],
        text: str,
        active_model: str,
        system_prompt: str,
        format_instructions: str,
        use_json_mode: bool,
        max_repair_attempts: int,
    ) -> T:
        return await ollama_structured.parse_with_repairs(self, parser=parser, text=text, active_model=active_model, system_prompt=system_prompt, format_instructions=format_instructions, use_json_mode=use_json_mode, max_repair_attempts=max_repair_attempts)

    # -------------------------------------------------------------------------
    @staticmethod
    def extract_first_json_object(text: str) -> str | None:
        return ollama_structured.extract_first_json_object(text)

    # -------------------------------------------------------------------------
    @staticmethod
    def parse_json(obj_or_text: dict[str, Any] | str) -> dict[str, Any] | None:
        return ollama_structured.parse_json(obj_or_text)


###############################################################################
