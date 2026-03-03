from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Body, HTTPException, Path, Query, status

from DILIGENT.server.entities.accesskeys import (
    AccessKeyCreateRequest,
    AccessKeyDeleteResponse,
    AccessKeyResponse,
)
from DILIGENT.server.repositories.serialization.accesskeys import AccessKeySerializer
from DILIGENT.server.repositories.schemas.models import AccessKey

ProviderName = Literal["openai", "gemini"]

router = APIRouter(prefix="/access-keys", tags=["access-keys"])
serializer = AccessKeySerializer()


###############################################################################
def to_response(row: AccessKey) -> AccessKeyResponse:
    return AccessKeyResponse(
        id=int(row.id),
        provider=str(row.provider),  # type: ignore[arg-type]
        is_active=bool(row.is_active),
        fingerprint=str(row.fingerprint),
        created_at=row.created_at,
        updated_at=row.updated_at,
        last_used_at=row.last_used_at,
    )


###############################################################################
@router.get("", response_model=list[AccessKeyResponse], status_code=status.HTTP_200_OK)
def list_access_keys(
    provider: ProviderName = Query(..., description="openai or gemini"),
) -> list[AccessKeyResponse]:
    try:
        rows = serializer.list_keys(provider)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return [to_response(row) for row in rows]


###############################################################################
@router.post("", response_model=AccessKeyResponse, status_code=status.HTTP_201_CREATED)
def create_access_key(
    payload: AccessKeyCreateRequest = Body(...),
) -> AccessKeyResponse:
    try:
        created = serializer.create_key(payload.provider, payload.access_key)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return to_response(created)


###############################################################################
@router.put(
    "/{key_id}/activate",
    response_model=AccessKeyResponse,
    status_code=status.HTTP_200_OK,
)
def activate_access_key(
    key_id: int = Path(..., ge=1),
) -> AccessKeyResponse:
    try:
        row = serializer.activate_key(key_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return to_response(row)


###############################################################################
@router.delete(
    "/{key_id}",
    response_model=AccessKeyDeleteResponse,
    status_code=status.HTTP_200_OK,
)
def delete_access_key(
    key_id: int = Path(..., ge=1),
) -> AccessKeyDeleteResponse:
    deleted = serializer.delete_key(key_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Access key not found",
        )
    return AccessKeyDeleteResponse()
