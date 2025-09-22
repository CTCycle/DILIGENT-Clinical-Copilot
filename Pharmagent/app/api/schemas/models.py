from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any

from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM


###############################################################################
class ModelPullResponse(BaseModel):
    status: str = Field(..., description="Operation status: 'success'")

    pulled: bool = Field(
        ...,
        description="True if a pull was performed, False if model was already present",
    )

    model: str = Field(..., description="Model name requested")


###############################################################################
class ModelListResponse(BaseModel):
    status: Literal["success"] = "success"
    models: list[str] = Field(..., description="List of available LLMs")
    count: int




