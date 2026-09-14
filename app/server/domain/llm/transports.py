from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from domain.llm.providers import ModelCapabilityMetadata, ToolCallMode

T = TypeVar("T", bound=BaseModel)
RequestOperation = Literal[
    "chat", "structured_output", "json_repair", "connectivity"
]
MessageRole = Literal["system", "user", "assistant", "tool"]
StreamEventKind = Literal[
    "text_delta",
    "reasoning_delta",
    "tool_call_delta",
    "completed",
    "usage",
    "error",
]

###############################################################################
class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

###############################################################################
class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str | None = None

###############################################################################
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: MessageRole
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    reasoning_content: str | None = None

###############################################################################
class UsageMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

###############################################################################
class ChatStreamEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: StreamEventKind
    text: str | None = None
    reasoning: str | None = None
    tool_call_index: int | None = Field(default=None, ge=0)
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments_delta: str | None = None
    finish_reason: str | None = None
    usage: UsageMetadata | None = None
    result: ChatResult | None = None
    error_code: str | None = None
    error_message: str | None = None

###############################################################################
class ChatRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    model: str
    messages: list[ChatMessage]
    options: dict[str, Any] = Field(default_factory=dict)
    json_mode: bool = False
    operation: RequestOperation = "chat"
    json_schema: dict[str, Any] | None = None
    reasoning_level: str | None = None
    reasoning_parameter: str | None = None
    reasoning_reserve: int | None = None
    output_token_limit: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[ToolDefinition] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    stream: bool = False
    capabilities: ModelCapabilityMetadata | None = None
    deadline_at: float | None = None
    cancel_check: Callable[[], bool] | None = Field(default=None, exclude=True)

###############################################################################
class ChatResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: UsageMetadata | None = None
    provider_model: str | None = None
    request_id: str | None = None
    partial: bool = False
    tool_call_mode: ToolCallMode | None = None

ChatStreamEvent.model_rebuild()

###############################################################################
class StructuredRequest(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[ChatMessage]
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
