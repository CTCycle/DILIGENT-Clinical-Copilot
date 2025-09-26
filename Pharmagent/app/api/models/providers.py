from __future__ import annotations

import asyncio
import contextlib
import ctypes
import inspect
import json
import os
import re
import shutil
import sys
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, Literal, TypeAlias, TypeVar, cast
import httpx
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from langsmith.run_helpers import get_current_run_tree, traceable

from Pharmagent.app.logger import logger
from Pharmagent.app.constants import (
    OLLAMA_HOST_DEFAULT,
    OPENAI_API_BASE,
    GEMINI_API_BASE,
    PARSING_MODEL_CHOICES,
)
from Pharmagent.app.configurations import ClientRuntimeConfig

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Type variable for typed schema returns
T = TypeVar("T", bound=BaseModel)

ProviderName = Literal["openai", "azure-openai", "anthropic", "gemini"]
RuntimePurpose = Literal["agent", "parser"]


###############################################################################
class OllamaError(RuntimeError):
    pass


class OllamaTimeout(OllamaError):
    """Raised when requests to Ollama exceed the configured timeout."""


ProgressCb: TypeAlias = Callable[[dict[str, Any]], None | Awaitable[None]]


# -------------------------------------------------------------------------
def _append_trace_metadata(
    metadata: dict[str, Any] | None = None, tags: list[str] | None = None
) -> None:
    run = get_current_run_tree()
    if not run:
        return
    if metadata:
        try:
            run.add_metadata(metadata)
        except Exception:  # noqa: BLE001 - tracing should never break runtime
            logger.debug("Failed to append LangSmith metadata", exc_info=True)
    if not tags:
        return
    try:
        run.add_tags(tags)
    except Exception:
        logger.debug("Failed to append LangSmith tags", exc_info=True)


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
        timeout_s: float = 1_800.0,
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
            resp = await self.client.get("/api/tags")
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
                async with self.client.stream("POST", "/api/pull", json=payload) as r:
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
                resp = await self.client.post("/api/pull", json=payload)
                self._raise_for_status(resp)
                return
        except httpx.TimeoutException as e:
            raise OllamaTimeout(f"Timed out pulling model '{name}'") from e

    # -------------------------------------------------------------------------
    async def show_model(self, name: str) -> dict[str, Any]:
        payload = {"name": name}
        try:
            resp = await self.client.post("/api/show", json=payload)
        except httpx.TimeoutException as e:
            raise OllamaTimeout(f"Timed out retrieving metadata for '{name}'") from e
        except httpx.RequestError as e:  # noqa: PERF203 - convert to domain error
            raise OllamaError(f"Failed to query model '{name}': {e}") from e

        self._raise_for_status(resp)

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
    def _parse_size_to_bytes(value: Any) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return 0
            match = re.match(r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[kKmMgGtTpP]?i?[bB])?", cleaned)
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
    def _get_available_memory_bytes() -> int:
        if sys.platform == "win32":
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
    async def _warm_model(self, name: str, *, keep_alive: str) -> None:
        messages = [
            {"role": "system", "content": "You are a background warmup assistant."},
            {"role": "user", "content": "Warmup."},
        ]
        await self.chat(
            model=name,
            messages=messages,
            format=None,
            options={"temperature": 0.0},
            keep_alive=keep_alive,
        )

    # -------------------------------------------------------------------------
    async def preload_models(
        self,
        parsing_model: str | None,
        agent_model: str | None,
        *,
        keep_alive: str = "30m",
    ) -> tuple[list[str], list[str]]:
        requested: list[str] = []
        for name in (parsing_model, agent_model):
            if not name:
                continue
            normalized = name.strip()
            if normalized and normalized not in requested:
                requested.append(normalized)

        if not requested:
            return [], []

        for name in requested:
            await self.check_model_availability(name, auto_pull=True)

        memory_budget = self._get_available_memory_bytes()
        sizes: dict[str, int] = {}
        for name in requested:
            try:
                details = await self.show_model(name)
            except OllamaError:
                sizes[name] = 0
                continue
            size = self._parse_size_to_bytes(details.get("size"))
            if size == 0:
                detail_info = details.get("details", {})
                if isinstance(detail_info, dict):
                    size = self._parse_size_to_bytes(detail_info.get("size"))
                if size == 0:
                    size = self._parse_size_to_bytes(
                        details.get("model_info", {}).get("size")
                        if isinstance(details.get("model_info"), dict)
                        else None
                    )
            sizes[name] = size

        to_load = list(requested)

        if memory_budget > 0 and any(sizes.get(name, 0) > 0 for name in requested):
            total_required = sum(sizes.get(name, 0) for name in requested)
            if total_required > memory_budget:
                parser_name = (parsing_model or "").strip()
                parser_size = sizes.get(parser_name, 0)
                if parser_name and parser_size and parser_size <= memory_budget:
                    to_load = [parser_name]
                elif parser_name and parser_size == 0:
                    to_load = [parser_name]
                else:
                    return [], list(requested)

        loaded: list[str] = []
        for name in to_load:
            try:
                await self._warm_model(name, keep_alive=keep_alive)
            except OllamaError as e:
                raise OllamaError(f"Failed to warm model '{name}': {e}") from e
            loaded.append(name)

        skipped = [name for name in requested if name not in loaded]
        return loaded, skipped

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
    @traceable(run_type="llm", name="Ollama.chat")
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        format: str | None = None,
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

        _append_trace_metadata(
            {"provider": "ollama", "model": model, "format": format or "text"},
            ["ollama", "chat"],
        )

        try:
            resp = await self.client.post("/api/chat", json=body)
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
            async with self.client.stream("POST", "/api/chat", json=body) as r:
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
    async def _collect_structured_fallbacks(
        self, preferred: list[str]
    ) -> list[str]:
        available: set[str] = set()
        try:
            available = set(await self.list_models())
        except (OllamaError, OllamaTimeout) as exc:
            logger.debug("Failed to list Ollama models for fallback: %s", exc)
            available = set()

        fallbacks: list[str] = []
        if available:
            for name in PARSING_MODEL_CHOICES:
                if name in available and name not in preferred and name not in fallbacks:
                    fallbacks.append(name)
        else:
            for name in PARSING_MODEL_CHOICES:
                if name not in preferred and name not in fallbacks:
                    fallbacks.append(name)

        return fallbacks

    # -------------------------------------------------------------------------
    @traceable(run_type="chain", name="Ollama.structured_call")
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
            preferred = await self._collect_structured_fallbacks([])

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

            _append_trace_metadata(
                {
                    "provider": "ollama",
                    "model": active_model,
                    "schema": schema.__name__,
                    "use_json_mode": use_json_mode,
                },
                ["ollama", "structured"],
            )

            try:
                raw = await self.chat(
                    model=active_model,
                    messages=messages,
                    format="json" if use_json_mode else None,
                    options={"temperature": temperature},
                )
            except OllamaError as e:
                message = str(e).lower()
                if "not found" in message or "404" in message:
                    missing.append(active_model)
                    last_missing_error = e
                    if fallbacks is None:
                        fallbacks = await self._collect_structured_fallbacks(preferred)
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
                        raise RuntimeError(
                            f"Structured parsing failed: {err}"
                        ) from err

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
                            options={"temperature": 0.0},
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
    @traceable(run_type="chain", name="CloudLLM.chat")
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
            _append_trace_metadata(
                {
                    "provider": "openai",
                    "model": model or (self.default_model or ""),
                    "format": format or "text",
                },
                ["cloud", "openai"],
            )
            return await self._chat_openai(
                model=model, messages=messages, format=format, options=options
            )
        if self.provider == "gemini":
            _append_trace_metadata(
                {
                    "provider": "gemini",
                    "model": model or (self.default_model or ""),
                    "format": format or "text",
                },
                ["cloud", "gemini"],
            )
            return await self._chat_gemini(model=model, messages=messages)
        raise LLMError(f"Provider '{self.provider}' does not support chat yet")

    # ---------------------------------------------------------------------
    @traceable(run_type="llm", name="CloudLLM.chat_openai")
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
            resp = await self.client.post("/chat/completions", json=body)
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
    @traceable(run_type="llm", name="CloudLLM.chat_gemini")
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
            resp = await self.client.post(path, json=body)
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
    @traceable(run_type="chain", name="CloudLLM.structured_call")
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

        _append_trace_metadata(
            {
                "provider": self.provider,
                "model": model or (self.default_model or ""),
                "schema": schema.__name__,
                "use_json_mode": use_json_mode,
            },
            ["cloud", "structured"],
        )

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
            timeout_s=kwargs.get("timeout_s", 120.0),
            keepalive_connections=kwargs.get("keepalive_connections", 10),
            keepalive_max=kwargs.get("keepalive_max", 20),
            default_model=kwargs.get("default_model"),
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
def initialize_llm_client(
    *, purpose: RuntimePurpose = "agent", **kwargs: Any
) -> OllamaClient | CloudLLMClient:
    if ClientRuntimeConfig.is_cloud_enabled():
        provider = ClientRuntimeConfig.get_llm_provider()
        default_model = ClientRuntimeConfig.get_cloud_model()
        if not default_model:
            default_model = (
                ClientRuntimeConfig.get_parsing_model()
                if purpose == "parser"
                else ClientRuntimeConfig.get_agent_model()
            )
    else:
        provider = "ollama"
        default_model = (
            ClientRuntimeConfig.get_parsing_model()
            if purpose == "parser"
            else ClientRuntimeConfig.get_agent_model()
        )
    selected_model = kwargs.pop("default_model", default_model)
    return select_llm_provider(
        provider=provider,
        default_model=selected_model,
        **kwargs,
    )
