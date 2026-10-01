# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

###############################################################################
class ApiErrorResponse(BaseModel):
    detail: Any
    request_id: str
    retryable: bool
