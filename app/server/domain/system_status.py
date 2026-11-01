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
class OllamaControl(BaseModel):
    """What DILIGENT can do about the local inference service right now."""

    state: Literal["running", "stopped", "starting", "error"]
    managed: bool
    can_start: bool
    can_stop: bool
    executable_found: bool
    error: str | None = None

###############################################################################
class SystemStatusResponse(BaseModel):
    providers: ProviderAvailability
    ollama: Literal["reachable", "unreachable"]
    ollama_control: OllamaControl
    cpu_percent: float | None = None
    ram: MemoryUsage | None = None
    vram: MemoryUsage | None = None
