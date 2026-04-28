from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Path, Query, status

from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.domain.keys import (
    AccessKeyCreateRequest,
    AccessKeyDeleteResponse,
    AccessKeyResponse,
)
from DILIGENT.server.services.access_keys_service import AccessKeyService, ProviderName

router = APIRouter(prefix="/access-keys", tags=["access-keys"])
service = AccessKeyService()
serializer = service.serializer


###############################################################################
@router.get("", response_model=list[AccessKeyResponse], status_code=status.HTTP_200_OK)
def list_access_keys(
    provider: ProviderName = Query(..., description="openai, gemini, or brave"),
) -> list[AccessKeyResponse]:
    try:
        return service.list_access_keys(provider)
    except ValueError as exc:
        logger.warning("Access key listing rejected: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid access key provider.",
        ) from exc


###############################################################################
@router.post("", response_model=AccessKeyResponse, status_code=status.HTTP_201_CREATED)
def create_access_key(
    payload: AccessKeyCreateRequest = Body(...),
) -> AccessKeyResponse:
    try:
        return service.create_access_key(payload.provider, payload.access_key)
    except RuntimeError as exc:
        logger.warning("Access key creation failed due to dependency/config issue: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Access key service is unavailable. Please retry shortly.",
        ) from exc
    except ValueError as exc:
        logger.warning("Access key creation rejected: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid access key input.",
        ) from exc


###############################################################################
@router.put(
    "/{key_id}/activate",
    response_model=AccessKeyResponse,
    status_code=status.HTTP_200_OK,
)
def activate_access_key(
    key_id: int = Path(..., ge=1),
    provider: ProviderName = Query(..., description="openai, gemini, or brave"),
) -> AccessKeyResponse:
    try:
        return service.activate_access_key(key_id, provider=provider)
    except ValueError as exc:
        logger.warning("Access key activation rejected for id=%s provider=%s: %s", key_id, provider, exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Access key not found.",
        ) from exc


###############################################################################
@router.delete(
    "/{key_id}",
    response_model=AccessKeyDeleteResponse,
    status_code=status.HTTP_200_OK,
)
def delete_access_key(
    key_id: int = Path(..., ge=1),
    provider: ProviderName = Query(..., description="openai, gemini, or brave"),
) -> AccessKeyDeleteResponse:
    deleted = service.delete_access_key(key_id, provider=provider)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Access key not found",
        )
    return AccessKeyDeleteResponse()

