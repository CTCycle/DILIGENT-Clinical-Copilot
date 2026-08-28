from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict

from common.security.desktop import DESKTOP_SESSION_COOKIE, DesktopSessionSecurity

router = APIRouter(prefix="/desktop", tags=["desktop"])


###############################################################################
class DesktopBootstrapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str


###############################################################################
def _security(request: Request) -> DesktopSessionSecurity:
    security = getattr(request.app.state, "desktop_security", None)
    if not isinstance(security, DesktopSessionSecurity):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return security


###############################################################################
@router.post("/bootstrap", status_code=status.HTTP_204_NO_CONTENT)
def bootstrap_desktop_session(
    payload: DesktopBootstrapRequest,
    request: Request,
    response: Response,
) -> None:
    security = _security(request)
    if not security.consume_bootstrap_token(payload.token):
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
@router.post("/shutdown", status_code=status.HTTP_202_ACCEPTED)
def request_desktop_shutdown(request: Request) -> dict[str, str]:
    _security(request)
    server = getattr(request.app.state, "desktop_server", None)
    if server is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Desktop server is not ready.",
        )
    server.should_exit = True
    return {"status": "shutting-down"}


__all__ = ["router"]
