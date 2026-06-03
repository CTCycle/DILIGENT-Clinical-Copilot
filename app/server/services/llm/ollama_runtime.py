from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypeAlias

import httpx
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

ProviderName = Literal["openai", "gemini"]
RuntimePurpose = Literal["clinical", "parser"]


class OllamaError(RuntimeError):
    pass


class OllamaTimeout(OllamaError):
    """Raised when requests to Ollama exceed the configured timeout."""


ProgressCb: TypeAlias = Callable[[dict[str, Any]], None | Awaitable[None]]


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


def build_langchain_messages(messages: list[dict[str, str]]) -> list[BaseMessage]:
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


def normalize_langchain_content(content: Any) -> dict[str, Any] | str:
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


def map_ollama_langchain_exception(exc: Exception) -> OllamaError:
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

