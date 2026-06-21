from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Query, status

from domain.model_configs import (
    ModelConfigStateResponse,
    ModelConfigUpdateRequest,
    OpenAIConnectivityCheckRequest,
    OpenAIConnectivityCheckResponse,
)
from services.llm.model_config import ModelConfigService

router = APIRouter(prefix="/model-config", tags=["model-config"])


###############################################################################
class ModelConfigEndpoint:
    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        router: APIRouter,
        service: ModelConfigService | None = None,
    ) -> None:
        self.router = router
        self.service = service or ModelConfigService()

    # -------------------------------------------------------------------------
    async def get_state(
        self,
        include_local_availability: Annotated[bool | None, Query()] = None,
    ) -> ModelConfigStateResponse:
        return await self.service.get_state(
            include_local_availability=include_local_availability,
        )

    # -------------------------------------------------------------------------
    async def update_state(
        self,
        payload: ModelConfigUpdateRequest = Body(...),
    ) -> ModelConfigStateResponse:
        return await self.service.update_state(payload)

    # -------------------------------------------------------------------------
    async def check_openai_connectivity(
        self,
        payload: OpenAIConnectivityCheckRequest = Body(
            default_factory=OpenAIConnectivityCheckRequest
        ),
    ) -> OpenAIConnectivityCheckResponse:
        return await self.service.check_openai_connectivity(payload)

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "",
            self.get_state,
            methods=["GET"],
            response_model=ModelConfigStateResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "",
            self.update_state,
            methods=["PUT"],
            response_model=ModelConfigStateResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/openai-connectivity-check",
            self.check_openai_connectivity,
            methods=["POST"],
            response_model=OpenAIConnectivityCheckResponse,
            status_code=status.HTTP_200_OK,
        )


ModelConfigEndpoint(router=router, service=ModelConfigService()).add_routes()
