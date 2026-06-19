from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ExtractionStrategy = Literal["deterministic", "llm", "hybrid"]


###############################################################################
class ExtractionToolParameterSchema(BaseModel):
    type: Literal["object"] = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


###############################################################################
class ExtractionToolDefinition(BaseModel):
    name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    parameters: ExtractionToolParameterSchema
    supported_section_types: list[str] = Field(default_factory=list)
    default_regex_profile: str = Field(..., min_length=1)
    allowed_configurable_profiles: list[str] = Field(default_factory=list)


###############################################################################
class RegexToolRequest(BaseModel):
    text: str = ""
    source_section: str = "unknown"
    profile: str | None = None


###############################################################################
class RegexToolMatch(BaseModel):
    match_text: str
    normalized_value: str
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=0)
    line_number: int = Field(..., ge=1)
    source_section: str
    pattern_id: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


###############################################################################
class RegexToolResult(BaseModel):
    tool_name: str
    source_section: str
    profile: str
    matches: list[RegexToolMatch] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


###############################################################################
class ExtractionStrategyDecision(BaseModel):
    section: str
    strategy: ExtractionStrategy
    confidence: float = Field(..., ge=0.0, le=1.0)
    therapy_structure_score: float | None = Field(default=None, ge=0.0, le=1.0)
    laboratory_structure_score: float | None = Field(default=None, ge=0.0, le=1.0)
    unresolved_line_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_span_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    ambiguity_count: int = Field(default=0, ge=0)
    reasons: list[str] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)


###############################################################################
class ExtractionToolError(BaseModel):
    tool_name: str
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
