from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from common.security.desktop import DesktopSessionSecurity


###############################################################################
class DesktopServerControl(Protocol):
    should_exit: bool


###############################################################################
class DesktopRuntimeService:
    # -------------------------------------------------------------------------
    def __init__(self, security: DesktopSessionSecurity | None = None) -> None:
        self.security = security if security is not None else DesktopSessionSecurity()
        self._server: DesktopServerControl | None = None

    # -------------------------------------------------------------------------
    def attach_server(self, server: DesktopServerControl) -> None:
        self._server = server

    # -------------------------------------------------------------------------
    def request_shutdown(self) -> bool:
        if self._server is None:
            return False
        self._server.should_exit = True
        return True


###############################################################################
@lru_cache(maxsize=1)
def get_desktop_runtime_service() -> DesktopRuntimeService:
    return DesktopRuntimeService()
