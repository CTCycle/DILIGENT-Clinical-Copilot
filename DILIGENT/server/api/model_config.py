from __future__ import annotations

from fastapi import APIRouter, Body, status

from DILIGENT.server.domain.model_configs import ModelConfigStateResponse, ModelConfigUpdateRequest
from DILIGENT.server.services.model_config_service import ModelConfigService

router = APIRouter(prefix="/model-config", tags=["model-config"])
service = ModelConfigService()


###############################################################################
class ModelConfigEndpoint:
    def __init__(
        self,
        *,
        router: APIRouter,
        service: ModelConfigService | None = None,
        serializer=None,
    ) -> None:
        self.router = router
        self.service = service or ModelConfigService(serializer=serializer)
        self.serializer = self.service.serializer

    # -------------------------------------------------------------------------
    def ensure_defaults(self):
        return self.service.ensure_defaults()

    # -------------------------------------------------------------------------
    def apply_runtime_snapshot(self, snapshot) -> None:
        self.service.apply_runtime_snapshot(snapshot)

    # -------------------------------------------------------------------------
    async def get_state(self) -> ModelConfigStateResponse:
        return await self.service.get_state()

    # -------------------------------------------------------------------------
    async def update_state(
        self,
        payload: ModelConfigUpdateRequest = Body(...),
    ) -> ModelConfigStateResponse:
        return await self.service.update_state(payload)

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


endpoint = ModelConfigEndpoint(router=router, service=service)
endpoint.add_routes()


def sync_runtime_model_config() -> None:
    """Load persisted model-config snapshot and apply it to runtime state."""
    snapshot = endpoint.ensure_defaults()
    endpoint.apply_runtime_snapshot(snapshot)
