from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

###############################################################################
class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    options: dict[str, Any] = {}
    json_mode: bool = False

###############################################################################
class ChatResult(BaseModel):
    content: str
    reasoning_content: str | None = None

###############################################################################
class StructuredRequest(BaseModel, Generic[T]):
    model: str
    messages: list[dict[str, str]]
    schema_type: type[T]

###############################################################################
class ConnectivityResult(BaseModel):
    ok: bool
    response_preview: str | None = None
    error: str | None = None

###############################################################################
class EmbeddingRequest(BaseModel):
    model: str
    inputs: list[str]
