from __future__ import annotations

from api.desktop import router as desktop_router
from common.security.desktop import DesktopSecurityMiddleware, DesktopSessionSecurity
from fastapi import FastAPI
from fastapi.testclient import TestClient
from services.runtime.desktop import DesktopRuntimeService


###############################################################################
class _DesktopServerStub:
    def __init__(self) -> None:
        self.should_exit = False


###############################################################################
def _model_config() -> dict[str, str]:
    return {"status": "ok"}


###############################################################################
def test_packaged_desktop_api_requires_bootstrap_cookie_and_exact_origin(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DILIGENT_DESKTOP", "true")
    monkeypatch.setenv("DILIGENT_DESKTOP_SESSION_SECRET", "session-secret")
    monkeypatch.setenv("DILIGENT_DESKTOP_PORT", "48123")
    application = FastAPI()
    security = DesktopSessionSecurity()
    runtime = DesktopRuntimeService(security)
    server = _DesktopServerStub()
    runtime.attach_server(server)
    application.state.desktop_runtime = runtime
    application.add_middleware(DesktopSecurityMiddleware, security=security)
    application.include_router(desktop_router, prefix="/api")
    application.add_api_route("/api/model-config", _model_config, methods=["GET"])

    headers = {"Host": "127.0.0.1:48123"}
    with TestClient(application) as client:
        unauthorized = client.get("/api/model-config", headers=headers)
        assert unauthorized.status_code == 401
        assert unauthorized.headers["X-Content-Type-Options"] == "nosniff"

        wrong_origin = client.post(
            "/api/desktop/bootstrap",
            headers={**headers, "Origin": "http://127.0.0.1:48124"},
            json={"token": "session-secret"},
        )
        assert wrong_origin.status_code == 403

        bootstrap = client.post(
            "/api/desktop/bootstrap",
            headers={**headers, "Origin": "http://127.0.0.1:48123"},
            json={"token": "session-secret"},
        )
        assert bootstrap.status_code == 204
        assert "HttpOnly" in bootstrap.headers["set-cookie"]
        assert "samesite=strict" in bootstrap.headers["set-cookie"].casefold()

        authorized = client.get("/api/model-config", headers=headers)
        assert authorized.status_code == 200

        replay = client.post(
            "/api/desktop/bootstrap",
            headers={**headers, "Origin": "http://127.0.0.1:48123"},
            json={"token": "session-secret"},
        )
        assert replay.status_code == 401

        shutdown = client.post(
            "/api/desktop/shutdown",
            headers={**headers, "Origin": "http://127.0.0.1:48123"},
        )
        assert shutdown.status_code == 202
        assert shutdown.json() == {"status": "shutting-down"}
        assert server.should_exit is True
