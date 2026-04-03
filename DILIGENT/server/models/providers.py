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

from DILIGENT.server.models.cloud import CloudLLMClient, LLMError, LLMTimeout
from DILIGENT.server.models.structured import StructuredOutputParser, parse_json_dict, T
from DILIGENT.server.configurations import LLMRuntimeConfig, server_settings
from DILIGENT.server.common.constants import (
    PARSING_MODEL_CHOICES,
)
from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.common.utils.types import extract_positive_int


ProviderName = Literal["openai", "azure-openai", "anthropic", "gemini"]
RuntimePurpose = Literal["clinical", "parser"]

__all__ = [
    "OllamaClient",
    "OllamaError",
    "OllamaTimeout",
    "CloudLLMClient",
    "LLMError",
    "LLMTimeout",
    "select_llm_provider",
    "initialize_llm_client",
]


###############################################################################
class OllamaError(RuntimeError):
    pass

###############################################################################
class OllamaTimeout(OllamaError):
    """Raised when requests to Ollama exceed the configured timeout."""

###############################################################################
class _OllamaChatFallback(Exception):
    """Internal control flow for switching to /api/generate streaming."""

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
    RESIDENCY_PLAN_TTL = 20.0
    DEFAULT_MODEL_FOOTPRINT_BYTES = 4 * 1_073_741_824

    def __init__(
        self,
        base_url: str | None = None,
        timeout_s: float = server_settings.external_data.default_llm_timeout,
        keepalive_connections: int = 10,
        keepalive_max: int = 20,
        default_model: str | None = None,
    ) -> None:
        self.base_url = (base_url or server_settings.llm_defaults.ollama_host_default).rstrip("/")
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
        self.model_size_bytes: dict[str, int] = {}
        self.model_vram_bytes: dict[str, int] = {}
        self.model_cache_expiry = 0.0
        self.model_cache_lock = asyncio.Lock()
        self.model_context_limits: dict[str, int] = {}
        self.residency_lock = asyncio.Lock()
        self.residency_plan_cache: dict[str, Any] | None = None
        self.residency_plan_cache_expiry = 0.0
        self.residency_usage_window_s = self.resolve_env_float(
            "OLLAMA_PREFETCH_USAGE_WINDOW_S",
            default=300.0,
            minimum=30.0,
        )
        self.residency_transition_window_s = self.resolve_env_float(
            "OLLAMA_PREFETCH_TRANSITION_WINDOW_S",
            default=120.0,
            minimum=5.0,
        )
        self.residency_prefetch_cooldown_s = self.resolve_env_float(
            "OLLAMA_PREFETCH_COOLDOWN_S",
            default=15.0,
            minimum=1.0,
        )
        self.residency_ram_safety_ratio = self.resolve_env_float(
            "OLLAMA_RAM_SAFETY_RATIO",
            default=0.85,
            minimum=0.1,
        )
        self.residency_vram_safety_ratio = self.resolve_env_float(
            "OLLAMA_VRAM_SAFETY_RATIO",
            default=0.85,
            minimum=0.1,
        )
        self.residency_dual_keep_alive = os.getenv(
            "OLLAMA_DUAL_RESIDENT_KEEP_ALIVE",
            "4h",
        ).strip() or "4h"
        self.residency_single_keep_alive = os.getenv(
            "OLLAMA_SINGLE_RESIDENT_KEEP_ALIVE",
            "30m",
        ).strip() or "30m"
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
    @staticmethod
    def resolve_env_float(name: str, *, default: float, minimum: float = 0.0) -> float:
        raw = (os.getenv(name) or "").strip()
        if not raw:
            return default
        try:
            parsed = float(raw)
        except ValueError:
            return default
        if parsed < minimum:
            return default
        return parsed

    # -------------------------------------------------------------------------
    @staticmethod
    def get_residency_targets() -> dict[str, str]:
        targets: dict[str, str] = {}
        clinical = (LLMRuntimeConfig.get_clinical_model() or "").strip()
        text_extraction = (LLMRuntimeConfig.get_parsing_model() or "").strip()
        if clinical:
            targets["clinical"] = clinical
        if text_extraction:
            targets["text_extraction"] = text_extraction
        return targets

    # -------------------------------------------------------------------------
    @staticmethod
    def dedupe_models(models: list[str]) -> list[str]:
        unique: list[str] = []
        for name in models:
            value = name.strip()
            if value and value not in unique:
                unique.append(value)
        return unique

    # -------------------------------------------------------------------------
    @classmethod
    def extract_bytes_from_fields(
        cls,
        payload: dict[str, Any],
        *,
        fields: tuple[str, ...],
    ) -> int:
        if not isinstance(payload, dict):
            return 0
        containers: list[dict[str, Any]] = [payload]
        for key in ("details", "model_info", "options"):
            block = payload.get(key)
            if isinstance(block, dict):
                containers.append(block)
        maximum = 0
        for block in containers:
            for field in fields:
                if field not in block:
                    continue
                maximum = max(maximum, cls.parse_size_to_bytes(block[field]))
        return maximum

    # -------------------------------------------------------------------------
    @classmethod
    def extract_footprint_from_payload(
        cls,
        payload: dict[str, Any],
    ) -> tuple[int, int]:
        ram_bytes = cls.extract_bytes_from_fields(
            payload,
            fields=("size", "size_bytes", "memory", "memory_size", "loaded_size"),
        )
        vram_bytes = cls.extract_bytes_from_fields(
            payload,
            fields=("size_vram", "vram", "vram_size", "gpu_size", "gpu_memory"),
        )
        return ram_bytes, vram_bytes

    # -------------------------------------------------------------------------
    async def list_running_models(self) -> dict[str, dict[str, Any]]:
        try:
            resp = await self.client.get("/api/ps")
        except (httpx.TimeoutException, httpx.RequestError):
            return {}
        if resp.status_code == 404:
            return {}
        try:
            self.raise_for_status(resp)
        except OllamaError:
            return {}
        try:
            payload = resp.json()
        except json.JSONDecodeError:
            return {}
        running: dict[str, dict[str, Any]] = {}
        for row in payload.get("models", []):
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip()
            if name:
                running[name] = row
        return running

    # -------------------------------------------------------------------------
    async def get_model_footprint_bytes(
        self,
        model: str,
        *,
        running_models: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[int, int]:
        ram_bytes = 0
        vram_bytes = 0
        if running_models and model in running_models:
            ram_bytes, vram_bytes = self.extract_footprint_from_payload(
                running_models[model]
            )
        ram_bytes = max(ram_bytes, self.model_size_bytes.get(model, 0))
        vram_bytes = max(vram_bytes, self.model_vram_bytes.get(model, 0))

        if ram_bytes <= 0 or vram_bytes <= 0:
            try:
                metadata = await self.show_model(model)
            except OllamaError:
                metadata = {}
            inferred_ram, inferred_vram = self.extract_footprint_from_payload(metadata)
            ram_bytes = max(ram_bytes, inferred_ram)
            vram_bytes = max(vram_bytes, inferred_vram)

        if ram_bytes <= 0:
            ram_bytes = self.DEFAULT_MODEL_FOOTPRINT_BYTES
        return ram_bytes, vram_bytes

    # -------------------------------------------------------------------------
    async def evaluate_dual_residency_plan(self) -> dict[str, Any]:
        targets = self.get_residency_targets()
        target_models = self.dedupe_models(list(targets.values()))
        available_ram = self.get_available_memory_bytes()
        available_vram = self.get_available_vram_bytes()
        running_models = await self.list_running_models()
        model_ram: dict[str, int] = {}
        model_vram: dict[str, int] = {}
        for model in target_models:
            ram_bytes, vram_bytes = await self.get_model_footprint_bytes(
                model,
                running_models=running_models,
            )
            model_ram[model] = ram_bytes
            model_vram[model] = vram_bytes

        required_ram = sum(model_ram.values())
        required_vram = sum(model_vram.values())
        ram_budget = int(available_ram * self.residency_ram_safety_ratio)
        vram_budget = int(available_vram * self.residency_vram_safety_ratio)
        has_vram_signal = available_vram > 0
        dual_possible = (
            len(target_models) >= 2
            and available_ram > 0
            and required_ram <= ram_budget
            and (not has_vram_signal or required_vram <= vram_budget)
        )
        plan = {
            "targets": targets,
            "models": target_models,
            "available_ram": available_ram,
            "available_vram": available_vram,
            "required_ram": required_ram,
            "required_vram": required_vram,
            "dual_residency": dual_possible,
        }
        logger.debug(
            (
                "Ollama residency plan dual=%s models=%s "
                "available_ram=%s required_ram=%s available_vram=%s required_vram=%s"
            ),
            dual_possible,
            target_models,
            available_ram,
            required_ram,
            available_vram,
            required_vram,
        )
        return plan

    # -------------------------------------------------------------------------
    async def get_cached_residency_plan(
        self,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        now = loop.time()
        async with self.residency_lock:
            if (
                not force_refresh
                and self.residency_plan_cache is not None
                and now < self.residency_plan_cache_expiry
            ):
                return dict(self.residency_plan_cache)

        plan = await self.evaluate_dual_residency_plan()
        async with self.residency_lock:
            self.residency_plan_cache = plan
            self.residency_plan_cache_expiry = loop.time() + self.RESIDENCY_PLAN_TTL
            return dict(plan)

    # -------------------------------------------------------------------------
    async def resolve_policy_keep_alive(
        self,
        *,
        active_model: str,
        requested_keep_alive: str | None,
    ) -> str | None:
        if requested_keep_alive:
            return requested_keep_alive
        plan = await self.get_cached_residency_plan()
        active_models = plan.get("models") or []
        if active_model not in active_models:
            return None
        if bool(plan.get("dual_residency")):
            return self.residency_dual_keep_alive
        return self.residency_single_keep_alive

    # -------------------------------------------------------------------------
    def record_target_usage(self, model: str) -> None:
        now = time.monotonic()
        self.residency_usage_history.append((now, model))
        cutoff = now - self.residency_usage_window_s
        while self.residency_usage_history:
            event_ts, _ = self.residency_usage_history[0]
            if event_ts >= cutoff:
                break
            self.residency_usage_history.popleft()

    # -------------------------------------------------------------------------
    def predict_next_target_model(
        self,
        *,
        current_model: str,
        target_models: list[str],
    ) -> str | None:
        candidates = self.dedupe_models(target_models)
        if current_model not in candidates:
            return None
        if len(candidates) < 2:
            return None

        now = time.monotonic()
        cutoff = now - self.residency_usage_window_s
        history = [
            (ts, model)
            for ts, model in self.residency_usage_history
            if ts >= cutoff and model in candidates
        ]

        frequency: dict[str, int] = {model: 0 for model in candidates}
        transitions: dict[tuple[str, str], int] = {}
        for _, model in history:
            frequency[model] = frequency.get(model, 0) + 1
        for (prev_ts, prev_model), (next_ts, next_model) in zip(history, history[1:]):
            if (next_ts - prev_ts) > self.residency_transition_window_s:
                continue
            key = (prev_model, next_model)
            transitions[key] = transitions.get(key, 0) + 1

        selected: str | None = None
        selected_score = -1.0
        for candidate in candidates:
            if candidate == current_model:
                continue
            transition_score = transitions.get((current_model, candidate), 0) * 3.0
            frequency_score = float(frequency.get(candidate, 0))
            recency_score = 0.5 if history and history[-1][1] == candidate else 0.0
            score = transition_score + frequency_score + recency_score
            if score > selected_score:
                selected = candidate
                selected_score = score
        if selected:
            return selected

        for candidate in candidates:
            if candidate != current_model:
                return candidate
        return None

    # -------------------------------------------------------------------------
    def handle_prefetch_task_done(self, task: asyncio.Task[None]) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            try:
                task.result()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Ollama model prefetch task failed: %s", exc)

    # -------------------------------------------------------------------------
    async def prefetch_model(
        self,
        *,
        model: str,
        keep_alive: str,
    ) -> None:
        try:
            await self.ensure_model_ready(model)
            payload = self.compose_payload(
                {
                    "model": model,
                    "prompt": "",
                    "stream": False,
                    "temperature": 0.0,
                    "think": False,
                },
                format=None,
                options={"num_predict": 0},
                keep_alive=keep_alive,
            )
            resp = await self.client.post("/api/generate", json=payload)
            self.raise_for_status(resp)
            logger.debug(
                "Prefetched Ollama model '%s' with keep_alive='%s'",
                model,
                keep_alive,
            )
        except OllamaError as exc:
            logger.debug("Skipping Ollama prefetch for '%s': %s", model, exc)
        except httpx.TimeoutException as exc:
            logger.debug("Timed out prefetching Ollama model '%s': %s", model, exc)
        except httpx.RequestError as exc:
            logger.debug("Request error prefetching Ollama model '%s': %s", model, exc)

    # -------------------------------------------------------------------------
    async def maybe_prefetch_target_model(self, *, active_model: str) -> None:
        plan = await self.get_cached_residency_plan()
        models = plan.get("models") or []
        if active_model not in models:
            return
        self.record_target_usage(active_model)

        if bool(plan.get("dual_residency")):
            candidate = next((name for name in models if name != active_model), None)
            keep_alive = self.residency_dual_keep_alive
        else:
            candidate = self.predict_next_target_model(
                current_model=active_model,
                target_models=models,
            )
            keep_alive = self.residency_single_keep_alive
        if not candidate:
            return

        now = time.monotonic()
        last_run = self.prefetch_last_run_by_model.get(candidate, 0.0)
        if (now - last_run) < self.residency_prefetch_cooldown_s:
            return
        current_task = self.prefetch_tasks.get(candidate)
        if current_task is not None and not current_task.done():
            return
        self.prefetch_last_run_by_model[candidate] = now
        task = asyncio.create_task(
            self.prefetch_model(model=candidate, keep_alive=keep_alive),
            name=f"ollama-prefetch:{candidate}",
        )
        task.add_done_callback(self.handle_prefetch_task_done)
        self.prefetch_tasks[candidate] = task

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    @staticmethod
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

    # -------------------------------------------------------------------------
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
    @staticmethod
    def decode_response_content(content: Any) -> Any:
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
        return str(content)

    # -------------------------------------------------------------------------
    @staticmethod
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

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    async def pull_once(self, *, payload: dict[str, Any]) -> bool:
        resp = await self.client.post("/api/pull", json=payload)
        self.raise_for_status(resp)
        return True

    # -------------------------------------------------------------------------
    async def refresh_cache_after_pull(self, completed: bool) -> None:
        if not completed:
            return
        try:
            await self.refresh_model_cache()
        except OllamaError as exc:
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
        wait_timeout_s: float = server_settings.external_data.ollama_server_start_timeout,
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
        for getter in (
            OllamaClient._get_available_memory_windows,
            OllamaClient._get_available_memory_sysconf,
            OllamaClient._get_available_memory_proc,
        ):
            available = getter()
            if available:
                return available
        return 0

    # -------------------------------------------------------------------------
    @staticmethod
    def get_available_vram_bytes() -> int:
        env_value = (os.getenv("OLLAMA_AVAILABLE_VRAM_BYTES") or "").strip()
        if env_value:
            parsed = OllamaClient.parse_size_to_bytes(env_value)
            if parsed > 0:
                return parsed
        return OllamaClient._get_available_vram_nvidia_smi()

    # -------------------------------------------------------------------------
    @staticmethod
    def _get_available_vram_nvidia_smi() -> int:
        if shutil.which("nvidia-smi") is None:
            return 0
        command = [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=1.5,
            )
        except (OSError, subprocess.SubprocessError):
            return 0
        if result.returncode != 0:
            return 0
        total = 0
        for line in result.stdout.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            token = cleaned.split()[0]
            try:
                mib = int(token)
            except ValueError:
                continue
            total += mib * 1_048_576
        return total

    # -------------------------------------------------------------------------
    @staticmethod
    def _get_available_memory_windows() -> int:
        kernel32 = getattr(getattr(ctypes, "windll", None), "kernel32", None)
        memory_status_fn = getattr(kernel32, "GlobalMemoryStatusEx", None)
        if memory_status_fn is None:
            return 0

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
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
        if memory_status_fn(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return 0

    # -------------------------------------------------------------------------
    @staticmethod
    def _get_available_memory_sysconf() -> int:
        sysconf = getattr(os, "sysconf", None)
        if not callable(sysconf):
            return 0
        try:
            page_size = sysconf("SC_PAGE_SIZE")
            sysconf_names = getattr(os, "sysconf_names", {})
            if "SC_AVPHYS_PAGES" in sysconf_names:
                pages = sysconf("SC_AVPHYS_PAGES")
            else:
                pages = sysconf("SC_PHYS_PAGES")
            if isinstance(page_size, int) and isinstance(pages, int):
                return page_size * pages
        except (ValueError, OSError, AttributeError):
            pass
        return 0

    # -------------------------------------------------------------------------
    @staticmethod
    def _parse_meminfo_line(line: str) -> int | None:
        if not line.startswith("MemAvailable:"):
            return None
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            value = int(parts[1])
        except ValueError:
            return None
        unit = parts[2].lower() if len(parts) >= 3 else "kb"
        multiplier = {
            "kb": 1_024,
            "kib": 1_024,
            "mb": 1_048_576,
            "mib": 1_048_576,
            "gb": 1_073_741_824,
            "gib": 1_073_741_824,
        }.get(unit)
        return value * multiplier if multiplier else value

    # -------------------------------------------------------------------------
    @staticmethod
    def _get_available_memory_proc() -> int:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    parsed = OllamaClient._parse_meminfo_line(line)
                    if parsed is not None:
                        return parsed
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
        resolved_keep_alive = await self.resolve_policy_keep_alive(
            active_model=resolved_model,
            requested_keep_alive=keep_alive,
        )

        if self.legacy_generate:
            content = await self.chat_via_generate(
                model=resolved_model,
                messages=messages,
                format=format,
                temperature=temp_value,
                think=think_value,
                options=options_payload,
                keep_alive=resolved_keep_alive,
            )
            await self.maybe_prefetch_target_model(active_model=resolved_model)
            return content

        body = self.build_chat_payload(
            model=resolved_model,
            messages=messages,
            stream=False,
            format=format,
            temperature=temp_value,
            think=think_value,
            options=options_payload,
            keep_alive=resolved_keep_alive,
        )

        try:
            resp = await self.client.post("/api/chat", json=body)
        except httpx.TimeoutException as e:
            raise OllamaTimeout("Timed out waiting for Ollama chat response") from e

        if resp.status_code == 404:
            await resp.aread()
            self.legacy_generate = True
            content = await self.chat_via_generate(
                model=resolved_model,
                messages=messages,
                format=format,
                temperature=temp_value,
                think=think_value,
                options=options_payload,
                keep_alive=resolved_keep_alive,
            )
            await self.maybe_prefetch_target_model(active_model=resolved_model)
            return content

        self.raise_for_status(resp)

        data = resp.json()
        content = (data.get("message") or {}).get("content", "")
        await self.maybe_prefetch_target_model(active_model=resolved_model)
        return self.decode_response_content(content)

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
        resolved_keep_alive = await self.resolve_policy_keep_alive(
            active_model=resolved_model,
            requested_keep_alive=keep_alive,
        )

        if self.legacy_generate:
            async for evt in self.chat_stream_via_generate(
                model=resolved_model,
                messages=messages,
                format=format,
                temperature=temp_value,
                think=think_value,
                options=options_payload,
                keep_alive=resolved_keep_alive,
            ):
                yield evt
            await self.maybe_prefetch_target_model(active_model=resolved_model)
            return

        body = self.build_chat_payload(
            model=resolved_model,
            messages=messages,
            stream=True,
            format=format,
            temperature=temp_value,
            think=think_value,
            options=options_payload,
            keep_alive=resolved_keep_alive,
        )
        try:
            async for evt in self._stream_chat_http(body):
                yield evt
            await self.maybe_prefetch_target_model(active_model=resolved_model)
            return
        except _OllamaChatFallback:
            self.legacy_generate = True

        async for evt in self.chat_stream_via_generate(
            model=resolved_model,
            messages=messages,
            format=format,
            temperature=temp_value,
            think=think_value,
            options=options_payload,
            keep_alive=resolved_keep_alive,
        ):
            yield evt
        await self.maybe_prefetch_target_model(active_model=resolved_model)

    # -------------------------------------------------------------------------
    async def _stream_chat_http(
        self, body: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        try:
            async with self.client.stream("POST", "/api/chat", json=body) as r:
                if r.status_code == 404:
                    await r.aread()
                    raise _OllamaChatFallback
                self.raise_for_status(r)
                async for evt in self.iter_json_stream_events(r):
                    yield evt
        except httpx.TimeoutException as e:
            raise OllamaTimeout("Timed out during streamed chat response") from e

    # -----------------------------------------------------------------------------
    async def build_generate_payload_from_messages(
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
        prompt = self.messages_to_prompt(messages)
        resolved_model = self.resolve_model_name(model)
        resolved_options = await self.ensure_context_option(
            model=resolved_model,
            messages=None,
            prompt=prompt,
            options=options,
        )
        return self.build_generate_payload(
            model=resolved_model,
            prompt=prompt,
            stream=stream,
            format=format,
            temperature=temperature,
            think=think,
            options=resolved_options,
            keep_alive=keep_alive,
        )

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
        payload = await self.build_generate_payload_from_messages(
            model=model,
            messages=messages,
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
        return self.decode_response_content(content)

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
        payload = await self.build_generate_payload_from_messages(
            model=model,
            messages=messages,
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
                async for evt in self.iter_json_stream_events(r):
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
        except OllamaError as exc:
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
        using a local JSON-schema-guided parser.

        - Injects format instructions so the LLM knows to return the expected JSON.
        - Parses & validates. If invalid, makes up to `max_repair_attempts` repair calls.
        - Returns an instance of `schema` (a Pydantic model).

        This function is LLM-agnostic beyond the Ollama client; you can reuse it
        across parsers by supplying different prompts/schemas.

        """
        parser = StructuredOutputParser(schema=schema)
        format_instructions = parser.get_format_instructions()
        messages = self.build_structured_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            format_instructions=format_instructions,
        )
        preferred = await self.resolve_parsing_models(model)
        return await self.call_with_structured_models(
            parser=parser,
            messages=messages,
            system_prompt=system_prompt,
            format_instructions=format_instructions,
            preferred=preferred,
            temperature=temperature,
            use_json_mode=use_json_mode,
            max_repair_attempts=max_repair_attempts,
        )

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
    async def resolve_parsing_models(self, model: str) -> list[str]:
        preferred: list[str] = []
        for candidate in (
            (model or "").strip(),
            (self.default_model or "").strip(),
            (LLMRuntimeConfig.get_parsing_model() or "").strip(),
        ):
            if candidate and candidate not in preferred:
                preferred.append(candidate)
        if not preferred:
            preferred = await self.collect_structured_fallbacks([])
        return preferred

    # -------------------------------------------------------------------------
    @staticmethod
    def is_missing_model_error(err: OllamaError) -> bool:
        message = str(err).lower()
        return "not found" in message or "404" in message

    # -------------------------------------------------------------------------
    async def _chat_structured_model(
        self,
        *,
        active_model: str,
        messages: list[dict[str, str]],
        use_json_mode: bool,
        temperature: float,
    ) -> dict[str, Any] | str:
        try:
            return await self.chat(
                model=active_model,
                messages=messages,
                format="json" if use_json_mode else None,
                temperature=temperature,
            )
        except OllamaError as err:
            if self.is_missing_model_error(err):
                raise
            raise RuntimeError(f"LLM call failed: {err}") from err

    # -------------------------------------------------------------------------
    async def _extend_structured_model_queue(
        self,
        *,
        queue: list[str],
        preferred_models: list[str],
        tried: set[str],
        fallbacks: list[str] | None,
    ) -> list[str]:
        if fallbacks is None:
            fallbacks = await self.collect_structured_fallbacks(preferred_models)
        for candidate in fallbacks:
            if candidate and candidate not in tried and candidate not in queue:
                queue.append(candidate)
        return fallbacks

    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_llm_text(raw: dict[str, Any] | str) -> str:
        return json.dumps(raw) if isinstance(raw, dict) else str(raw)

    # -------------------------------------------------------------------------
    @staticmethod
    def _raise_structured_models_exhausted(
        *,
        last_missing_error: Exception | None,
        missing: list[str],
    ) -> NoReturn:
        if last_missing_error:
            attempted = ", ".join(missing)
            raise RuntimeError(
                "LLM call failed: no local parsing models were found. "
                f"Tried: {attempted}"
            ) from last_missing_error
        raise RuntimeError("LLM call failed: no parsing model candidates available")

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
                raw = await self._chat_structured_model(
                    active_model=active_model,
                    messages=messages,
                    use_json_mode=use_json_mode,
                    temperature=temperature,
                )
            except OllamaError as e:
                missing.append(active_model)
                last_missing_error = e
                fallbacks = await self._extend_structured_model_queue(
                    queue=queue,
                    preferred_models=preferred,
                    tried=tried,
                    fallbacks=fallbacks,
                )
                continue

            return await self.parse_with_repairs(
                parser=parser,
                text=self._coerce_llm_text(raw),
                active_model=active_model,
                system_prompt=system_prompt,
                format_instructions=format_instructions,
                use_json_mode=use_json_mode,
                max_repair_attempts=max_repair_attempts,
            )

        self._raise_structured_models_exhausted(
            last_missing_error=last_missing_error,
            missing=missing,
        )

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
                try:
                    raw = await self.chat(
                        model=active_model,
                        messages=repair_messages,
                        format="json" if use_json_mode else None,
                        temperature=0.0,
                    )
                    text = self._coerce_llm_text(raw)

                except OllamaError as e:
                    raise RuntimeError(f"Repair attempt failed: {e}") from e

        raise RuntimeError("No structured output produced by the model")

    # -------------------------------------------------------------------------
    @staticmethod
    def extract_first_json_object(text: str) -> str | None:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            start = match.start()
            try:
                parsed, end = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return text[start : start + end]
        return None

    # -------------------------------------------------------------------------
    @staticmethod
    def parse_json(obj_or_text: dict[str, Any] | str) -> dict[str, Any] | None:
        return parse_json_dict(obj_or_text)


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
            timeout_s=kwargs.get("timeout_s", server_settings.external_data.default_llm_timeout),
            keepalive_connections=kwargs.get("keepalive_connections", 10),
            keepalive_max=kwargs.get("keepalive_max", 20),
            default_model=kwargs.get("default_model"),
        )
    if p in ("openai", "gemini"):
        return CloudLLMClient(
            provider=p,  # type: ignore[arg-type]
            base_url=kwargs.get("base_url"),
            timeout_s=kwargs.get("timeout_s", server_settings.external_data.default_llm_timeout),
            keepalive_connections=kwargs.get("keepalive_connections", 10),
            keepalive_max=kwargs.get("keepalive_max", 20),
            default_model=kwargs.get("default_model"),
        )
    raise LLMError(f"Unknown or unsupported provider: {provider}")


###############################################################################
def initialize_llm_client(
    *, purpose: RuntimePurpose = "clinical", **kwargs: Any
) -> OllamaClient | CloudLLMClient:
    kwargs.setdefault("timeout_s", server_settings.external_data.default_llm_timeout)
    provider, default_model = LLMRuntimeConfig.resolve_provider_and_model(purpose)
    selected_model = kwargs.pop("default_model", default_model)
    return select_llm_provider(
        provider=provider,
        default_model=selected_model,
        **kwargs,
    )

