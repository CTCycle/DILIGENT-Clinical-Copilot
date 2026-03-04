from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


###############################################################################
class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=2000)
    mode: Literal["fast", "thorough"] = "fast"
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None

    @field_validator("question", mode="before")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("question cannot be empty")
        return normalized


###############################################################################
class ResearchSource(BaseModel):
    url: str = Field(..., min_length=1, max_length=2000)
    title: str | None = Field(default=None, max_length=500)
    score: float | None = None
    snippet: str | None = Field(default=None, max_length=2000)
    extracted_text: str | None = Field(default=None, max_length=5000)
    retrieved_at: str = Field(..., min_length=1, max_length=64)


###############################################################################
class ResearchCitation(BaseModel):
    claim: str = Field(..., min_length=1, max_length=1000)
    urls: list[str] = Field(default_factory=list)


###############################################################################
class ResearchResponse(BaseModel):
    answer: str = Field(..., min_length=1)
    sources: list[ResearchSource] = Field(default_factory=list)
    citations: list[ResearchCitation] = Field(default_factory=list)
    message: str | None = Field(default=None, max_length=500)

