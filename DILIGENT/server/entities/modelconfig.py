from __future__ import annotations

from datetime import datetime
from typing import Literal
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

SAFE_PROVIDER_RE = re.compile(r"^[a-z][a-z0-9-]{1,31}$")
SAFE_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,199}$")


###############################################################################
class LocalModelCard(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=400)
    family: str = Field(..., min_length=1, max_length=80)
    available_in_ollama: bool = False


###############################################################################
class ModelConfigStateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
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
    model_config = ConfigDict(extra="forbid")
    use_cloud_services: bool | None = None
    llm_provider: str | None = Field(default=None, max_length=32)
    cloud_model: str | None = Field(default=None, max_length=200)
    clinical_model: str | None = Field(default=None, max_length=200)
    text_extraction_model: str | None = Field(default=None, max_length=200)
    ollama_reasoning: bool | None = None

    # -------------------------------------------------------------------------
    @field_validator("llm_provider")
    @classmethod
    def validate_provider(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not normalized:
            return None
        if not SAFE_PROVIDER_RE.fullmatch(normalized):
            raise ValueError("Invalid llm_provider value")
        return normalized

    # -------------------------------------------------------------------------
    @field_validator("cloud_model", "clinical_model", "text_extraction_model")
    @classmethod
    def validate_model_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if not SAFE_MODEL_NAME_RE.fullmatch(normalized):
            raise ValueError("Invalid model name")
        return normalized
