from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Path, Response, status

from domain.model_configs import (
    ModelConfigPersistResponse,
    ModelCatalogOperationResponse,
    CatalogProviderId,
    ModelConfigStateResponse,
    ModelConfigUpdateRequest,
    ConnectivityCheckRequest,
    ConnectivityCheckResponse,
    EmbeddingStatusResponse,
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
        response: Response,
    ) -> ModelConfigStateResponse:
        response.headers["Cache-Control"] = (
            "no-store, no-cache, max-age=0, must-revalidate"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return await self.service.get_state()

    # -------------------------------------------------------------------------
    async def update_state(
        self,
        payload: ModelConfigUpdateRequest = Body(...),
    ) -> ModelConfigPersistResponse:
        return await self.service.update_state(payload)

    # -------------------------------------------------------------------------
    async def load_catalog(
        self,
        provider: Annotated[CatalogProviderId, Path()],
    ) -> ModelCatalogOperationResponse:
        return await self.service.load_catalog(provider)

    # -------------------------------------------------------------------------
    async def refresh_catalog(
        self,
        provider: Annotated[CatalogProviderId, Path()],
    ) -> ModelCatalogOperationResponse:
        return await self.service.load_catalog(provider, force_refresh=True)

    # -------------------------------------------------------------------------
    async def check_connectivity(
        self,
        payload: ConnectivityCheckRequest = Body(...),
    ) -> ConnectivityCheckResponse:
        return await self.service.check_connectivity(payload)

    # -------------------------------------------------------------------------
    async def get_embedding_status(self) -> EmbeddingStatusResponse:
        return await self.service.get_embedding_status()

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
            "/catalogs/{provider}/load",
            self.load_catalog,
            methods=["POST"],
            response_model=ModelCatalogOperationResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/catalogs/{provider}/refresh",
            self.refresh_catalog,
            methods=["POST"],
            response_model=ModelCatalogOperationResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "",
            self.update_state,
            methods=["PUT"],
            response_model=ModelConfigPersistResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/embedding-status",
            self.get_embedding_status,
            methods=["GET"],
            response_model=EmbeddingStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/connectivity-check",
            self.check_connectivity,
            methods=["POST"],
            response_model=ConnectivityCheckResponse,
            status_code=status.HTTP_200_OK,
        )


ModelConfigEndpoint(router=router, service=ModelConfigService()).add_routes()
