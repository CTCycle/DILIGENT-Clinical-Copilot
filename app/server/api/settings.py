from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Path, Response, status

from domain.settings.runtime_ui import (
    RuntimeSettingsCategory,
    RuntimeSettingsStateResponse,
    RuntimeSettingsUpdateRequest,
)
from services.settings.runtime import RuntimeSettingsService

router = APIRouter(prefix="/settings", tags=["settings"])


###############################################################################
class SettingsEndpoint:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        router: APIRouter,
        service: RuntimeSettingsService | None = None,
    ) -> None:
        self.router = router
        self.service = service or RuntimeSettingsService()

    # -------------------------------------------------------------------------
    def get_state(self, response: Response) -> RuntimeSettingsStateResponse:
        response.headers["Cache-Control"] = "no-store, no-cache, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return self.service.get_state()

    # -------------------------------------------------------------------------
    def update_state(
        self,
        payload: RuntimeSettingsUpdateRequest = Body(...),
    ) -> RuntimeSettingsStateResponse:
        return self.service.update_state(payload)

    # -------------------------------------------------------------------------
    def reset_category(
        self,
        category: Annotated[RuntimeSettingsCategory, Path()],
    ) -> RuntimeSettingsStateResponse:
        return self.service.reset_category(category)

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "",
            self.get_state,
            methods=["GET"],
            response_model=RuntimeSettingsStateResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "",
            self.update_state,
            methods=["PATCH"],
            response_model=RuntimeSettingsStateResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/reset/{category}",
            self.reset_category,
            methods=["POST"],
            response_model=RuntimeSettingsStateResponse,
            status_code=status.HTTP_200_OK,
        )


settings_endpoint = SettingsEndpoint(router=router, service=RuntimeSettingsService())
settings_endpoint.add_routes()
