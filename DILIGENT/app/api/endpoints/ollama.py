from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from DILIGENT.app.api.models.providers import OllamaClient, OllamaError, OllamaTimeout
from DILIGENT.app.api.schemas.models import ModelListResponse, ModelPullResponse
from DILIGENT.app.logger import logger

router = APIRouter(prefix="/models", tags=["models"])


###############################################################################
@router.get(
    "/pull",
    response_model=ModelPullResponse,
    status_code=status.HTTP_200_OK,
    summary="Pull a specific Ollama model",
    description="Synchronously pull an Ollama model by name. If the model already exists locally, no pull is performed.",
)
async def pull_model(
    name: str = Query(..., description="Exact Ollama model name, e.g. 'llama3.1:8b'"),
    stream: bool = Query(
        False,
        description="If True, stream pull from Ollama. Endpoint returns only final status (no SSE).",
    ),
) -> ModelPullResponse:
    try:
        async with OllamaClient() as client:
            local = set(await client.list_models())
            already = name in local
            if not already:
                logger.info(f"Downloading model {name} from Ollama library")
                await client.pull(name, stream=stream)
            return ModelPullResponse(status="success", pulled=(not already), model=name)

    except Exception as exc:
        from DILIGENT.app.api.models.providers import OllamaError, OllamaTimeout

        if isinstance(exc, OllamaTimeout):
            raise HTTPException(status_code=504, detail=str(exc))
        if isinstance(exc, OllamaError):
            raise HTTPException(status_code=502, detail=str(exc))
        raise HTTPException(
            status_code=500, detail="Unexpected error while pulling model"
        )


###############################################################################
@router.get(
    "/list",
    response_model=ModelListResponse,
    status_code=status.HTTP_200_OK,
    summary="List locally available Ollama models",
    description="Returns the list of model tags already present on the Ollama host.",
)
async def list_available_models() -> ModelListResponse:
    try:
        async with OllamaClient() as client:
            models = await client.list_models()
        return ModelListResponse(models=models, count=len(models))
    except Exception as exc:
        # Map specific client exceptions to HTTP errors where possible
        if isinstance(exc, OllamaTimeout):
            raise HTTPException(status_code=504, detail=str(exc))
        if isinstance(exc, OllamaError):
            raise HTTPException(status_code=502, detail=str(exc))
        raise HTTPException(
            status_code=500, detail="Unexpected error while listing models"
        )
