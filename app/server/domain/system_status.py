# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

###############################################################################
class MemoryUsage(BaseModel):
    used_bytes: int
    total_bytes: int

###############################################################################
class ProviderAvailability(BaseModel):
    active: int
    total: int

###############################################################################
class SystemStatusResponse(BaseModel):
    providers: ProviderAvailability
    ollama: Literal["reachable", "unreachable"]
    cpu_percent: float | None = None
    ram: MemoryUsage | None = None
    vram: MemoryUsage | None = None
