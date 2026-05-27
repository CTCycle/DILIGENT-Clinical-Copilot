from __future__ import annotations

from fastapi import APIRouter, status

from domain.health import HealthResponse

router = APIRouter(tags=["health"])


###############################################################################
def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


router.add_api_route(
    "/health",
    health_check,
    methods=["GET"],
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
)

