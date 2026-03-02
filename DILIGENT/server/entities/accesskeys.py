from __future__ import annotations

from datetime import datetime
from typing import Literal
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator


ProviderName = Literal["openai", "gemini"]
CONTROL_CHARACTERS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


###############################################################################
class AccessKeyCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: ProviderName
    access_key: str = Field(..., min_length=1, max_length=8192)

    # -------------------------------------------------------------------------
    @field_validator("access_key", mode="before")
    @classmethod
    def strip_access_key(cls, value: str | None) -> str:
        if value is None:
            raise ValueError("access_key must not be null")
        without_controls = CONTROL_CHARACTERS_RE.sub("", str(value))
        normalized = without_controls.strip()
        if not normalized:
            raise ValueError("access_key must not be empty")
        return normalized


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
class AccessKeyDeleteResponse(BaseModel):
    status: Literal["success"] = "success"
    deleted: bool = True
