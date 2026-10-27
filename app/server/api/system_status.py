# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from fastapi import APIRouter, Response, status

from domain.system_status import SystemStatusResponse
from services.runtime.system_status import SystemStatusService

router = APIRouter(prefix="/system", tags=["system"])

_service = SystemStatusService()

###############################################################################
async def get_system_status(response: Response) -> SystemStatusResponse:
    response.headers["Cache-Control"] = "no-store"
    return await _service.get_status()


router.add_api_route(
    "/status",
    get_system_status,
    methods=["GET"],
    response_model=SystemStatusResponse,
    status_code=status.HTTP_200_OK,
)

###############################################################################
def rebind_database_services() -> None:
    global _service
    _service = SystemStatusService()
