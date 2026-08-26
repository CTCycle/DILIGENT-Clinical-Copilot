from __future__ import annotations

from api.desktop import router as desktop_router
from common.security.desktop import DesktopSecurityMiddleware, DesktopSessionSecurity
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_packaged_desktop_api_requires_bootstrap_cookie_and_exact_origin(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DILIGENT_DESKTOP", "true")
    monkeypatch.setenv("DILIGENT_DESKTOP_SESSION_SECRET", "session-secret")
    monkeypatch.setenv("DILIGENT_DESKTOP_PORT", "48123")
    application = FastAPI()
    security = DesktopSessionSecurity()
    application.state.desktop_security = security
    application.state.desktop_server = type("Server", (), {"should_exit": False})()
    application.add_middleware(DesktopSecurityMiddleware, security=security)
    application.include_router(desktop_router, prefix="/api")

    @application.get("/api/model-config")
    def model_config() -> dict[str, str]:
        return {"status": "ok"}

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
