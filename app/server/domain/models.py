# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

###############################################################################
class ModelListResponse(BaseModel):
    status: Literal["success"] = "success"
    models: list[str] = Field(..., description="List of available LLMs")
    count: int
