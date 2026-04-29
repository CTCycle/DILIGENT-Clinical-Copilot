from __future__ import annotations

from fastapi import APIRouter, status


router = APIRouter(tags=["health"])


###############################################################################
def health_check() -> dict[str, str]:
    return {"status": "ok"}


router.add_api_route(
    "/health",
    health_check,
    methods=["GET"],
    status_code=status.HTTP_200_OK,
)
