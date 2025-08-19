from typing import Any, Optional, List, Literal
from typing_extensions import Annotated
from pydantic import BaseModel, Field

from Pharmagent.app.api.schemas.placeholders import EXAMPLE_INPUT_DATA



###############################################################################
class ModelPullResponse(BaseModel):
    status: str = Field(
        ..., 
        description="Operation status: 'success'")
    
    pulled: bool = Field(
        ..., 
        description="True if a pull was performed, False if model was already present")
    
    model: str = Field(
        ..., 
        description="Model name requested")
    
###############################################################################
class ModelListResponse(BaseModel):
    status: Literal["success"] = "success"
    models: List[str] = Field(
        ..., 
        description="List of available LLMs")
    count: int