from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


###############################################################################
class LLMToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


###############################################################################
class LLMToolCallRequest(BaseModel):
    model: str
    system_prompt: str = ""
    user_prompt: str
    tools: list[LLMToolDefinition]
    temperature: float = 0.0


###############################################################################
class LLMToolCall(BaseModel):
    id: str | None = None
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


###############################################################################
class LLMToolCallResult(BaseModel):
    provider: str
    model: str
    content: str | None = None
    tool_calls: list[LLMToolCall] = Field(default_factory=list)


###############################################################################
class LLMToolCallError(BaseModel):
    provider: str
    code: str
    message: str
