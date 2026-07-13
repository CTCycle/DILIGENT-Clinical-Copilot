from __future__ import annotations

from typing import Any

from pydantic import BaseModel

###############################################################################
class ApiErrorResponse(BaseModel):
    detail: Any
    request_id: str
    retryable: bool
