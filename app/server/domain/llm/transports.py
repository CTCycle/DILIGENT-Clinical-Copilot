from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)

###############################################################################
class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    options: dict[str, Any] = Field(default_factory=dict)
    json_mode: bool = False
    reasoning_level: str | None = None
    reasoning_parameter: str | None = None
    reasoning_reserve: int | None = None
    output_token_limit: int | None = None

###############################################################################
class ChatResult(BaseModel):
    content: str
    reasoning_content: str | None = None

###############################################################################
class StructuredRequest(BaseModel, Generic[T]):
    model: str
    messages: list[dict[str, str]]
    schema_type: type[T]
    options: dict[str, Any] = Field(default_factory=dict)
    reasoning_level: str | None = None
    reasoning_parameter: str | None = None
    reasoning_reserve: int | None = None
    output_token_limit: int | None = None

###############################################################################
class ConnectivityResult(BaseModel):
    ok: bool
    response_preview: str | None = None
    error: str | None = None

###############################################################################
class EmbeddingRequest(BaseModel):
    model: str
    inputs: list[str]
