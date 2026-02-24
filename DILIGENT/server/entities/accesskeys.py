from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


ProviderName = Literal["openai", "gemini"]


###############################################################################
class AccessKeyCreateRequest(BaseModel):
    provider: ProviderName
    access_key: str = Field(..., min_length=1, max_length=8192)


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
