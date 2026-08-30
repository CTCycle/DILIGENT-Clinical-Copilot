from __future__ import annotations

import hmac
import os
import threading

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

DESKTOP_SESSION_COOKIE = "diligent_desktop_session"
_STATE_CHANGING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "geolocation=(), camera=(), microphone=()",
    "Content-Security-Policy": (
        "default-src 'self'; connect-src 'self'; img-src 'self' data: blob:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"
    ),
}


###############################################################################
def _is_true(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "on"}


###############################################################################
class DesktopSessionSecurity:
    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.enabled = _is_true(os.getenv("DILIGENT_DESKTOP", ""))
        self.secret = os.getenv("DILIGENT_DESKTOP_SESSION_SECRET", "")
        port_value = os.getenv("DILIGENT_DESKTOP_PORT", "")
        try:
            self.port = int(port_value)
        except ValueError:
            self.port = 0
        self._bootstrap_lock = threading.Lock()
        self._bootstrap_consumed = False

    # -------------------------------------------------------------------------
    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # -------------------------------------------------------------------------
    def host_is_allowed(self, host: str) -> bool:
        return self.enabled and self.port > 0 and host == f"127.0.0.1:{self.port}"

    # -------------------------------------------------------------------------
    def origin_is_allowed(self, origin: str) -> bool:
        return origin == self.origin

    # -------------------------------------------------------------------------
    def consume_bootstrap_token(self, token: str) -> bool:
        if (
            not self.enabled
            or not self.secret
            or not hmac.compare_digest(token, self.secret)
        ):
            return False
        with self._bootstrap_lock:
            if self._bootstrap_consumed:
                return False
            self._bootstrap_consumed = True
            return True

    # -------------------------------------------------------------------------
    def request_has_session(self, request: Request) -> bool:
        cookie = request.cookies.get(DESKTOP_SESSION_COOKIE, "")
        return bool(self.secret) and hmac.compare_digest(cookie, self.secret)


###############################################################################
def _secured(response: Response) -> Response:
    for name, value in _SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    return response


###############################################################################
def _rejected(status_code: int) -> Response:
    return _secured(
        JSONResponse(
            {"detail": "Desktop request is not authorized."}, status_code=status_code
        )
    )


###############################################################################
class DesktopSecurityMiddleware(BaseHTTPMiddleware):
    # -------------------------------------------------------------------------
    def __init__(
        self,
        app: ASGIApp,
        *,
        security: DesktopSessionSecurity,
    ) -> None:
        super().__init__(app)
        self.security = security

    # -------------------------------------------------------------------------
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if not self.security.enabled:
            return await call_next(request)

        if not self.security.host_is_allowed(request.headers.get("host", "")):
            return _rejected(403)

        if (
            request.method in _STATE_CHANGING_METHODS
            and not self.security.origin_is_allowed(request.headers.get("origin", ""))
        ):
            return _rejected(403)

        is_health = request.url.path == "/api/health"
        is_bootstrap = request.url.path == "/api/desktop/bootstrap"
        if request.url.path.startswith("/api/") and not is_health and not is_bootstrap:
            if not self.security.request_has_session(request):
                return _rejected(401)

        return _secured(await call_next(request))


__all__ = [
    "DESKTOP_SESSION_COOKIE",
    "DesktopSecurityMiddleware",
    "DesktopSessionSecurity",
]
