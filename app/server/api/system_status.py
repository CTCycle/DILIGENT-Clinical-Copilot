# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status

from domain.system_status import SystemStatusResponse
from services.runtime.ollama_process import OllamaProcessError
from services.runtime.system_status import SystemStatusService

router = APIRouter(prefix="/system", tags=["system"])

_service = SystemStatusService()

###############################################################################
async def get_system_status(response: Response) -> SystemStatusResponse:
    response.headers["Cache-Control"] = "no-store"
    return await _service.get_status()

###############################################################################
async def start_ollama(response: Response) -> SystemStatusResponse:
    response.headers["Cache-Control"] = "no-store"
    try:
        await _service.start_ollama()
    except OllamaProcessError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return await _service.get_status()

###############################################################################
async def stop_ollama(response: Response) -> SystemStatusResponse:
    response.headers["Cache-Control"] = "no-store"
    if not await _service.stop_ollama():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ollama was not started by DILIGENT and cannot be stopped here.",
        )
    return await _service.get_status()


router.add_api_route(
    "/status",
    get_system_status,
    methods=["GET"],
    response_model=SystemStatusResponse,
    status_code=status.HTTP_200_OK,
)
router.add_api_route(
    "/ollama/start",
    start_ollama,
    methods=["POST"],
    response_model=SystemStatusResponse,
    status_code=status.HTTP_200_OK,
)
router.add_api_route(
    "/ollama/stop",
    stop_ollama,
    methods=["POST"],
    response_model=SystemStatusResponse,
    status_code=status.HTTP_200_OK,
)

###############################################################################
def rebind_database_services() -> None:
    global _service
    _service = SystemStatusService()
