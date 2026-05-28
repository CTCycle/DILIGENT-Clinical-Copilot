from __future__ import annotations

import asyncio
import contextlib
import ctypes
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypeAlias

import httpx
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from common.utils.logger import logger
from configurations.llm_configs import LLMRuntimeConfig

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
    except TypeError, ValueError:
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


def get_residency_targets() -> dict[str, str]:
    targets: dict[str, str] = {}
    clinical = (LLMRuntimeConfig.get_clinical_model() or "").strip()
    text_extraction = (LLMRuntimeConfig.get_text_extraction_model() or "").strip()
    if clinical:
        targets["clinical"] = clinical
    if text_extraction:
        targets["text_extraction"] = text_extraction
    return targets


def dedupe_models(models: list[str]) -> list[str]:
    unique: list[str] = []
    for name in models:
        value = name.strip()
        if value and value not in unique:
            unique.append(value)
    return unique


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


async def list_running_models(self) -> dict[str, dict[str, Any]]:
    try:
        resp = await self.client.get("/api/ps")
    except httpx.TimeoutException, httpx.RequestError:
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


def record_target_usage(self, model: str) -> None:
    now = time.monotonic()
    self.residency_usage_history.append((now, model))
    cutoff = now - self.residency_usage_window_s
    while self.residency_usage_history:
        event_ts, _ = self.residency_usage_history[0]
        if event_ts >= cutoff:
            break
        self.residency_usage_history.popleft()


def predict_next_target_model(
    self,
    *,
    current_model: str,
    target_models: list[str],
) -> str | None:
    candidates = self.dedupe_models(target_models)
    if current_model not in candidates or len(candidates) < 2:
        return None

    history = self._recent_residency_history(candidates)
    frequency: dict[str, int] = dict.fromkeys(candidates, 0)
    self._count_residency_frequency(history, frequency)
    transitions = self._count_residency_transitions(history)

    selected = self._select_target_model(
        current_model=current_model,
        candidates=candidates,
        history=history,
        frequency=frequency,
        transitions=transitions,
    )
    if selected is not None:
        return selected

    return next(
        (candidate for candidate in candidates if candidate != current_model), None
    )


def _recent_residency_history(self, candidates: list[str]) -> list[tuple[float, str]]:
    now = time.monotonic()
    cutoff = now - self.residency_usage_window_s
    return [
        (ts, model)
        for ts, model in self.residency_usage_history
        if ts >= cutoff and model in candidates
    ]


def _count_residency_frequency(
    history: list[tuple[float, str]],
    frequency: dict[str, int],
) -> None:
    for _, model in history:
        frequency[model] += 1


def _count_residency_transitions(
    self,
    history: list[tuple[float, str]],
) -> dict[tuple[str, str], int]:
    transitions: dict[tuple[str, str], int] = {}
    for (prev_ts, prev_model), (next_ts, next_model) in zip(history, history[1:]):
        if (next_ts - prev_ts) > self.residency_transition_window_s:
            continue
        key = (prev_model, next_model)
        transitions[key] = transitions.get(key, 0) + 1
    return transitions


def _select_target_model(
    *,
    current_model: str,
    candidates: list[str],
    history: list[tuple[float, str]],
    frequency: dict[str, int],
    transitions: dict[tuple[str, str], int],
) -> str | None:
    selected: str | None = None
    selected_score = -1.0
    last_model = history[-1][1] if history else None
    for candidate in candidates:
        if candidate == current_model:
            continue
        score = (
            transitions.get((current_model, candidate), 0) * 3.0
            + float(frequency.get(candidate, 0))
            + (0.5 if last_model == candidate else 0.0)
        )
        if score > selected_score:
            selected = candidate
            selected_score = score
    return selected


def handle_prefetch_task_done(self, task: asyncio.Task[None]) -> None:
    with contextlib.suppress(asyncio.CancelledError):
        try:
            task.result()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Ollama model prefetch task failed: %s", exc)


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


def get_available_memory_bytes() -> int:
    for getter in (
        _get_available_memory_windows,
        _get_available_memory_sysconf,
        _get_available_memory_proc,
    ):
        available = getter()
        if available:
            return available
    return 0


def get_available_vram_bytes() -> int:
    env_value = (os.getenv("OLLAMA_AVAILABLE_VRAM_BYTES") or "").strip()
    if env_value:
        parsed = parse_size_to_bytes(env_value)
        if parsed > 0:
            return parsed
    return _get_available_vram_nvidia_smi()


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
    except OSError, subprocess.SubprocessError:
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
    except ValueError, OSError, AttributeError:
        pass
    return 0


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


def _get_available_memory_proc() -> int:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                parsed = _parse_meminfo_line(line)
                if parsed is not None:
                    return parsed
    except FileNotFoundError, PermissionError, ValueError:
        pass
    return 0
