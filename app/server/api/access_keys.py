from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Path, Query, status

from common.utils.logger import logger
from domain.keys import (
    AccessKeyCreateRequest,
    AccessKeyDeleteResponse,
    AccessKeyResponse,
    ProviderName,
)
from services.security.access_keys import AccessKeyService

router = APIRouter(prefix="/access-keys", tags=["access-keys"])


###############################################################################
class AccessKeyEndpoint:
    def __init__(self, *, router: APIRouter, service: AccessKeyService) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    def list_access_keys(
        self,
        provider: ProviderName = Query(
            ...,
            description="openai, gemini, or openrouter",
        ),
    ) -> list[AccessKeyResponse]:
        try:
            return self.service.list_access_keys(provider)
        except ValueError as exc:
            logger.warning("Access key listing rejected: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid access key provider.",
            ) from exc

    # -------------------------------------------------------------------------
    def create_access_key(
        self,
        payload: AccessKeyCreateRequest = Body(...),
    ) -> AccessKeyResponse:
        try:
            return self.service.create_access_key(payload.provider, payload.access_key)
        except RuntimeError as exc:
            logger.warning(
                "Access key creation failed due to dependency/config issue: %s", exc
            )
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

    # -------------------------------------------------------------------------
    def activate_access_key(
        self,
        key_id: int = Path(..., ge=1),
        provider: ProviderName = Query(
            ...,
            description="openai, gemini, or openrouter",
        ),
    ) -> AccessKeyResponse:
        try:
            return self.service.activate_access_key(key_id, provider=provider)
        except ValueError as exc:
            logger.warning(
                "Access key activation rejected for id=%s provider=%s: %s",
                key_id,
                provider,
                exc,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Access key not found.",
            ) from exc

    # -------------------------------------------------------------------------
    def delete_access_key(
        self,
        key_id: int = Path(..., ge=1),
        provider: ProviderName = Query(
            ...,
            description="openai, gemini, or openrouter",
        ),
    ) -> AccessKeyDeleteResponse:
        deleted = self.service.delete_access_key(key_id, provider=provider)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Access key not found",
            )
        return AccessKeyDeleteResponse()

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "",
            self.list_access_keys,
            methods=["GET"],
            response_model=list[AccessKeyResponse],
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "",
            self.create_access_key,
            methods=["POST"],
            response_model=AccessKeyResponse,
            status_code=status.HTTP_201_CREATED,
        )
        self.router.add_api_route(
            "/{key_id}/activate",
            self.activate_access_key,
            methods=["PUT"],
            response_model=AccessKeyResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/{key_id}",
            self.delete_access_key,
            methods=["DELETE"],
            response_model=AccessKeyDeleteResponse,
            status_code=status.HTTP_200_OK,
        )


AccessKeyEndpoint(router=router, service=AccessKeyService()).add_routes()
