from __future__ import annotations

import re
from datetime import datetime
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

ProviderName = Literal["openai", "gemini", "deepseek", "anthropic", "opencode", "brave"]
SUPPORTED_PROVIDERS: frozenset[ProviderName] = frozenset(
    ("openai", "gemini", "deepseek", "anthropic", "opencode", "brave")
)
CONTROL_CHARACTERS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
MIN_ACCESS_KEY_LENGTH = 16
PLACEHOLDER_ACCESS_KEY_RE = re.compile(
    r"(?:fake|test|example|placeholder|dummy|invalid)", re.IGNORECASE
)

###############################################################################
def normalize_provider_name(provider: str) -> ProviderName:
    normalized = str(provider or "").strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        raise ValueError("Unsupported provider")
    return cast(ProviderName, normalized)

###############################################################################
def normalize_access_key(value: str | None) -> str:
    if value is None:
        raise ValueError("access_key must not be null")
    normalized = CONTROL_CHARACTERS_RE.sub("", str(value)).strip()
    if not normalized:
        raise ValueError("access_key must not be empty")
    if len(normalized) < MIN_ACCESS_KEY_LENGTH:
        raise ValueError("access_key is too short")
    if any(character.isspace() for character in normalized):
        raise ValueError("access_key must not contain whitespace")
    return normalized

###############################################################################
def validate_provider_access_key(provider: ProviderName, value: str | None) -> str:
    """Reject obvious placeholders without retaining or logging the secret."""
    normalized = normalize_access_key(value)
    if PLACEHOLDER_ACCESS_KEY_RE.search(normalized):
        raise ValueError(f"Invalid {provider.title()} access key format.")
    prefix_by_provider = {
        "openai": ("sk-",),
        "gemini": ("AIza",),
        "deepseek": ("sk-",),
        "anthropic": ("sk-ant-",),
    }
    accepted_prefixes = prefix_by_provider.get(provider)
    if accepted_prefixes and not normalized.startswith(accepted_prefixes):
        raise ValueError(f"Invalid {provider.title()} access key format.")
    return normalized

###############################################################################
class AccessKeyCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: ProviderName
    access_key: str = Field(..., min_length=MIN_ACCESS_KEY_LENGTH, max_length=8192)

    # -------------------------------------------------------------------------
    @field_validator("access_key", mode="before")
    @classmethod
    def strip_access_key(cls, value: str | None) -> str:
        return normalize_access_key(value)

###############################################################################
class AccessKeyResponse(BaseModel):
    id: int
    provider: ProviderName
    is_active: bool
    fingerprint: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_used_at: datetime | None = None

###############################################################################
class AccessKeyRecord(BaseModel):
    id: int
    provider: ProviderName
    created_at: datetime
    updated_at: datetime
    is_active: bool
    key_fingerprint: str
    last_used_at: datetime | None = None

###############################################################################
class AccessKeyDeleteResponse(BaseModel):
    status: Literal["success"] = "success"
    deleted: bool = True
