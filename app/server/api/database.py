# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, status

from configurations.database_config import (
    configuration_from_request,
    load_database_configuration,
)
from domain.settings.database import (
    DatabaseConfigurationRequest,
    DatabaseConfigurationStateResponse,
    DatabaseConnectionTestResponse,
    DatabaseSwitchResponse,
)
from services.runtime.database_switch import (
    DatabaseActivationError,
    DatabaseSwitchBusyError,
    get_database_runtime_manager,
)

router = APIRouter(prefix="/database", tags=["database"])


###############################################################################
@router.get("/configuration", response_model=DatabaseConfigurationStateResponse)
def get_database_configuration() -> DatabaseConfigurationStateResponse:
    try:
        return DatabaseConfigurationStateResponse.model_validate(
            get_database_runtime_manager().get_state()
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


###############################################################################
@router.post("/test", response_model=DatabaseConnectionTestResponse)
def test_database_configuration(
    payload: DatabaseConfigurationRequest = Body(...),
) -> DatabaseConnectionTestResponse:
    current = load_database_configuration()
    candidate = configuration_from_request(
        payload.model_dump(mode="python"), current
    )
    return get_database_runtime_manager().test_configuration(candidate)


###############################################################################
@router.put("/configuration", response_model=DatabaseSwitchResponse)
def update_database_configuration(
    payload: DatabaseConfigurationRequest = Body(...),
) -> DatabaseSwitchResponse:
    current = load_database_configuration()
    candidate = configuration_from_request(
        payload.model_dump(mode="python"), current
    )
    try:
        return get_database_runtime_manager().switch_configuration(candidate)
    except DatabaseSwitchBusyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except DatabaseActivationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
