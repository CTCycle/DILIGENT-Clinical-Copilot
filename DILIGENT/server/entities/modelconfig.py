from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


###############################################################################
class LocalModelCard(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=400)


###############################################################################
class ModelConfigStateResponse(BaseModel):
    status: Literal["success"] = "success"
    local_models: list[LocalModelCard] = Field(default_factory=list)
    cloud_model_choices: dict[str, list[str]] = Field(default_factory=dict)
    use_cloud_services: bool = False
    llm_provider: str = "openai"
    cloud_model: str | None = None
    clinical_model: str | None = None
    text_extraction_model: str | None = None
    ollama_reasoning: bool = False
    updated_at: datetime | None = None


###############################################################################
class ModelConfigUpdateRequest(BaseModel):
    use_cloud_services: bool | None = None
    llm_provider: str | None = None
    cloud_model: str | None = None
    clinical_model: str | None = None
    text_extraction_model: str | None = None
    ollama_reasoning: bool | None = None
