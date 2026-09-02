from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


###############################################################################
class DesktopBootstrapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str


###############################################################################
class DesktopShutdownResponse(BaseModel):
    status: Literal["shutting-down"]
