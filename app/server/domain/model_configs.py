from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


###############################################################################
@dataclass(frozen=True)
class ModelConfigSnapshot:
    clinical_model: str | None
    text_extraction_model: str | None
    use_cloud_models: bool
    cloud_provider: str | None
    cloud_model: str | None
    ollama_temperature: float
    cloud_temperature: float
    ollama_reasoning: bool = False
    updated_at: datetime | None = None


###############################################################################
class LocalModelCard(BaseModel):
    name: str
    family: str
    description: str
    available_in_ollama: bool


###############################################################################
class ModelConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    use_cloud_services: bool | None = None
    llm_provider: str | None = None
    cloud_model: str | None = None
    text_extraction_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("text_extraction_model", "text_extraction_model"),
    )
    clinical_model: str | None = None
    ollama_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    cloud_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    ollama_reasoning: bool | None = None


###############################################################################
class ModelConfigStateResponse(BaseModel):
    local_models: list[LocalModelCard]
    cloud_model_choices: dict[str, list[str]]
    use_cloud_services: bool
    llm_provider: str
    cloud_model: str | None
    text_extraction_model: str | None
    clinical_model: str | None
    ollama_temperature: float = Field(ge=0.0, le=2.0)
    cloud_temperature: float = Field(ge=0.0, le=2.0)
    ollama_reasoning: bool
