from __future__ import annotations

import re
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DOMAIN_NAME_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$"
)
MAX_DOMAIN_FILTERS = 25


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

    @staticmethod
    def normalize_domain(value: Any) -> str | None:
        raw = str(value or "").strip().casefold()
        if not raw:
            return None
        parsed = urlparse(raw if "://" in raw else f"https://{raw}")
        domain = (parsed.netloc or parsed.path).strip()
        if not domain:
            return None
        domain = domain.split("/", maxsplit=1)[0].split(":", maxsplit=1)[0].strip(".")
        if domain.startswith("www."):
            domain = domain[4:]
        if not DOMAIN_NAME_RE.fullmatch(domain):
            raise ValueError(f"Invalid domain filter value: {value!r}")
        return domain

    @field_validator("allowed_domains", "blocked_domains", mode="before")
    @classmethod
    def normalize_domain_filters(cls, value: Any) -> list[str] | None:
        if value is None:
            return None

        raw_values: list[Any]
        if isinstance(value, str):
            raw_values = [value]
        elif isinstance(value, (list, tuple, set)):
            raw_values = list(value)
        else:
            raise ValueError("Domain filters must be a list of domain names.")

        normalized: list[str] = []
        seen: set[str] = set()
        for item in raw_values:
            domain = cls.normalize_domain(item)
            if domain is None or domain in seen:
                continue
            seen.add(domain)
            normalized.append(domain)
            if len(normalized) > MAX_DOMAIN_FILTERS:
                raise ValueError(
                    f"A maximum of {MAX_DOMAIN_FILTERS} domain filters is allowed."
                )

        return normalized or None

    @model_validator(mode="after")
    def validate_domain_filters(self) -> "ResearchRequest":
        if not self.allowed_domains or not self.blocked_domains:
            return self
        overlap = sorted(set(self.allowed_domains) & set(self.blocked_domains))
        if overlap:
            raise ValueError(
                "allowed_domains and blocked_domains cannot include the same domain."
            )
        return self


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
class ResearchAnswerPayload(BaseModel):
    answer: str = Field(..., min_length=1)
    citations: list[ResearchCitation] = Field(default_factory=list)


###############################################################################
class ResearchResponse(BaseModel):
    answer: str = Field(..., min_length=1)
    sources: list[ResearchSource] = Field(default_factory=list)
    citations: list[ResearchCitation] = Field(default_factory=list)
    message: str | None = Field(default=None, max_length=500)
