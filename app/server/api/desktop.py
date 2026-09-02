from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status

from common.security.desktop import DESKTOP_SESSION_COOKIE
from domain.desktop import DesktopBootstrapRequest, DesktopShutdownResponse
from services.runtime.desktop import DesktopRuntimeService

router = APIRouter(prefix="/desktop", tags=["desktop"])


###############################################################################
def _runtime(request: Request) -> DesktopRuntimeService:
    runtime = getattr(request.app.state, "desktop_runtime", None)
    if not isinstance(runtime, DesktopRuntimeService):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return runtime


###############################################################################
@router.post("/bootstrap", status_code=status.HTTP_204_NO_CONTENT)
def bootstrap_desktop_session(
    payload: DesktopBootstrapRequest,
    request: Request,
    response: Response,
) -> None:
    runtime = _runtime(request)
    if not runtime.security.consume_bootstrap_token(payload.token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Desktop request is not authorized.",
        )
    response.set_cookie(
        DESKTOP_SESSION_COOKIE,
        payload.token,
        httponly=True,
        samesite="strict",
        secure=False,
        path="/",
    )


###############################################################################
@router.post(
    "/shutdown",
    response_model=DesktopShutdownResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def request_desktop_shutdown(request: Request) -> DesktopShutdownResponse:
    runtime = _runtime(request)
    if not runtime.request_shutdown():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Desktop server is not ready.",
        )
    return DesktopShutdownResponse(status="shutting-down")


__all__ = ["router"]
