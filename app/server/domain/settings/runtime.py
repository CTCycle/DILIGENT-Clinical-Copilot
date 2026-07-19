from __future__ import annotations

from pydantic import BaseModel, ConfigDict

###############################################################################
class LLMRuntimeState(BaseModel):
    model_config = ConfigDict(frozen=False)
    text_extraction_model: str = ""
    clinical_model: str = ""
    llm_provider: str = ""
    cloud_model: str = ""
    use_cloud_services: bool = False
    ollama_reasoning: bool = False
    revision: int = 0
