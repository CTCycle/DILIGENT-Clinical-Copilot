# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from starlette.responses import JSONResponse

from common.constants import FASTAPI_API_PREFIX
from services.runtime.database_switch import get_database_runtime_manager


class DatabaseSwitchMiddleware:
    def __init__(self, app) -> None:
        self.app = app
        self._gate = get_database_runtime_manager().request_gate
        self._api_prefix = FASTAPI_API_PREFIX.rstrip("/") + "/"
        self._database_prefix = self._api_prefix + "database/"

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if not path.startswith(self._api_prefix) or path.startswith(self._database_prefix):
            await self.app(scope, receive, send)
            return
        if not self._gate.enter_request():
            response = JSONResponse(
                status_code=503,
                content={"detail": "The database is switching. Retry this request shortly."},
                headers={"Retry-After": "2"},
            )
            await response(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        finally:
            self._gate.leave_request()
