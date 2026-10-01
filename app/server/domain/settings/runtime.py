# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from domain.model_configs import ReasoningLevel

###############################################################################
class LLMRuntimeState(BaseModel):
    model_config = ConfigDict(frozen=False)
    text_extraction_model: str = ""
    clinical_model: str = ""
    llm_provider: str = ""
    cloud_model: str = ""
    use_cloud_services: bool = False
    reasoning_level: ReasoningLevel = ReasoningLevel.OFF
    revision: int = 0
