from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from DILIGENT.server.models.providers import OllamaClient, OllamaError, OllamaTimeout
from DILIGENT.server.schemas.models import ModelListResponse, ModelPullResponse
from DILIGENT.server.packages.logger import logger

router = APIRouter(prefix="/models", tags=["models"])


###############################################################################
class OllamaEndpoint:
    def __init__(self, *, router: APIRouter) -> None:
        self.router = router
        self.router.add_api_route(
            "/pull",
            self.pull_model,
            methods=["GET"],
            response_model=ModelPullResponse,
            status_code=status.HTTP_200_OK,
            summary="Pull a specific Ollama model",
            description="Synchronously pull an Ollama model by name. If the model already exists locally, no pull is performed.",
        )
        self.router.add_api_route(
            "/list",
            self.list_available_models,
            methods=["GET"],
            response_model=ModelListResponse,
            status_code=status.HTTP_200_OK,
            summary="List locally available Ollama models",
            description="Returns the list of model tags already present on the Ollama host.",
        )

    async def pull_model(
        self,
        name: str = Query(
            ...,
            description="Exact Ollama model name, e.g. 'llama3.1:8b'",
        ),
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
                return ModelPullResponse(
                    status="success", pulled=(not already), model=name
                )
        except Exception as exc:
            from DILIGENT.server.models.providers import (
                OllamaError,
                OllamaTimeout,
            )

            if isinstance(exc, OllamaTimeout):
                raise HTTPException(status_code=504, detail=str(exc))
            if isinstance(exc, OllamaError):
                raise HTTPException(status_code=502, detail=str(exc))
            raise HTTPException(
                status_code=500, detail="Unexpected error while pulling model"
            )

    async def list_available_models(self) -> ModelListResponse:
        try:
            async with OllamaClient() as client:
                models = await client.list_models()
            return ModelListResponse(models=models, count=len(models))
        except Exception as exc:
            if isinstance(exc, OllamaTimeout):
                raise HTTPException(status_code=504, detail=str(exc))
            if isinstance(exc, OllamaError):
                raise HTTPException(status_code=502, detail=str(exc))
            raise HTTPException(
                status_code=500, detail="Unexpected error while listing models"
            )


endpoint = OllamaEndpoint(router=router)
