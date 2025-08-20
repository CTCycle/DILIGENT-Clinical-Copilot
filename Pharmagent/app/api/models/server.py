from __future__ import annotations

import os
import re
import json
import asyncio
import inspect
from typing import Any, Dict, List, Optional, AsyncGenerator, Callable, Awaitable, Union, Type

import httpx
from pydantic import BaseModel, ValidationError
from langchain_core.output_parsers import PydanticOutputParser

from Pharmagent.app.constants import DATA_PATH
from Pharmagent.app.logger import logger


DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


###############################################################################
class OllamaError(RuntimeError):
    pass

class OllamaTimeout(OllamaError):
    """Raised when requests to Ollama exceed the configured timeout."""


ProgressCb = Callable[[Dict[str, Any]], Union[None, Awaitable[None]]]


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
        base_url: Optional[str] = None,
        timeout_s: float = 120.0,
        keepalive_connections: int = 10,
        keepalive_max: int = 20) -> None:
        self.base_url = (base_url or DEFAULT_OLLAMA_HOST).rstrip("/")
        limits = httpx.Limits(max_keepalive_connections=keepalive_connections, max_connections=keepalive_max)
        timeout = httpx.Timeout(timeout_s)
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout, limits=limits)

    #--------------------------------------------------------------------------
    async def close(self) -> None:
        await self._client.aclose()

    #--------------------------------------------------------------------------
    async def __aenter__(self) -> "OllamaClient":
        return self

    #--------------------------------------------------------------------------
    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    #--------------------------------------------------------------------------
    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = resp.text
            raise OllamaError(f"Ollama HTTP {resp.status_code}: {detail}") from e

    #--------------------------------------------------------------------------
    @staticmethod
    async def _maybe_await(cb: Optional[ProgressCb], evt: Dict[str, Any]) -> None:
        if cb is None:
            return
        try:
            res = cb(evt)
            if inspect.isawaitable(res):
                await res  # type: ignore[func-returns-value]
        except Exception as e:  # don't break the pull loop on callback errors
            # attach minimal context; callers can log externally
            raise OllamaError(f"Progress callback failed: {e!r}") from e

    #--------------------------------------------------------------------------
    async def list_models(self) -> List[str]:
        try:
            resp = await self._client.get("/api/tags")
        except httpx.TimeoutException as e:
            raise OllamaTimeout("Timed out listing Ollama models") from e
        self._raise_for_status(resp)
        payload = resp.json()
        return [m["name"] for m in payload.get("models", []) if "name" in m]

    #--------------------------------------------------------------------------
    async def pull(
        self,
        name: str,
        *,
        stream: bool = False,
        progress_callback: Optional[ProgressCb] = None,
        poll_sleep_s: float = 0.05) -> None:

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

    #--------------------------------------------------------------------------
    async def check_model_availability(self, name: str, *, auto_pull: bool = True) -> None:
        names = set(await self.list_models())
        if name not in names and auto_pull:
            await self.pull(name, stream=False)
        elif name not in names:
            raise OllamaError(f"Model '{name}' not found and auto_pull=False")

    #--------------------------------------------------------------------------
    async def chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        format: Optional[str] = "json",
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None) -> Dict[str, Any] | str:
        """
        Non-streaming chat. Returns parsed JSON (dict) if possible, else raw string.
        """
        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
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

    #--------------------------------------------------------------------------
    async def chat_stream(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streamed chat. Yields each event (already JSON-decoded).
        Caller can aggregate tokens or forward server-sent chunks to a client.
        """
        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": True}
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
        
    #--------------------------------------------------------------------------
    async def llm_structured_call(
        self,
        *,        
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: Type[BaseModel],
        temperature: float = 0.0,
        use_json_mode: bool = True,
        max_repair_attempts: int = 2) -> BaseModel:
        
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
            {"role": "system", "content": f"{system_prompt.strip()}\n\n{format_instructions}"},
            {"role": "user", "content": user_prompt}]

        try:
            raw = await self.chat(
                model=model,
                messages=messages,
                format="json" if use_json_mode else None,
                options={"temperature": temperature})
            
        except OllamaError as e:
            raise RuntimeError(f"LLM call failed: {e}") from e

        # Unify to text for the LC parser
        text = json.dumps(raw) if isinstance(raw, dict) else str(raw)

        # First parse attempt + bounded auto-repair loop
        for attempt in range(max_repair_attempts + 1):
            try:
                return parser.parse(text)
            except Exception as err:
                if attempt >= max_repair_attempts:
                    # Surface original model output in logs for debugging
                    logger.error("Structured parse failed after retries. Last text: %s", text)
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
                        options={"temperature": 0.0})
                    text = json.dumps(raw) if isinstance(raw, dict) else str(raw)
                    
                except OllamaError as e:
                    raise RuntimeError(f"Repair attempt failed: {e}") from e

    #--------------------------------------------------------------------------
    @staticmethod
    def parse_json(obj_or_text: Dict[str, Any] | str) -> Optional[Dict[str, Any]]:
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

    

