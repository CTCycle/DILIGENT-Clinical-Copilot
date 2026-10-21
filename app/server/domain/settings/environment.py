# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

###############################################################################
class EnvironmentSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    ollama_url: str | None
    ollama_host: str | None
    ollama_port: int | None
