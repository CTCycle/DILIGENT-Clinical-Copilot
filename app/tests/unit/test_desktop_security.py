# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from api.desktop import router as desktop_router
from common.security.desktop import DESKTOP_SESSION_HEADER
from common.security.desktop import DesktopSecurityMiddleware, DesktopSessionSecurity
from fastapi import FastAPI
from fastapi.testclient import TestClient
from services.runtime.desktop import DesktopRuntimeService


class _DesktopServerStub:
    def __init__(self) -> None:
        self.should_exit = False


def _model_config() -> dict[str, str]:
    return {"status": "ok"}


def test_packaged_desktop_api_requires_header_session_and_exact_tauri_origin(
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

    host = {"Host": "127.0.0.1:48123"}
    tauri_origin = "http://tauri.localhost"
    valid_session = {DESKTOP_SESSION_HEADER: "session-secret"}
    with TestClient(application) as client:
        unauthorized = client.get("/api/model-config", headers=host)
        assert unauthorized.status_code == 401
        assert unauthorized.headers["X-Content-Type-Options"] == "nosniff"

        wrong_origin = client.post(
            "/api/desktop/bootstrap",
            headers={**host, "Origin": "http://127.0.0.1:48123"},
            json={"token": "session-secret"},
        )
        assert wrong_origin.status_code == 403

        bootstrap = client.post(
            "/api/desktop/bootstrap",
            headers={**host, "Origin": tauri_origin},
            json={"token": "session-secret"},
        )
        assert bootstrap.status_code == 204
        assert "set-cookie" not in bootstrap.headers

        authorized = client.get(
            "/api/model-config",
            headers={**host, "Origin": tauri_origin, **valid_session},
        )
        assert authorized.status_code == 200

        invalid_session = client.get(
            "/api/model-config",
            headers={**host, "Origin": tauri_origin, DESKTOP_SESSION_HEADER: "wrong"},
        )
        assert invalid_session.status_code == 401

        replay = client.post(
            "/api/desktop/bootstrap",
            headers={**host, "Origin": tauri_origin},
            json={"token": "session-secret"},
        )
        assert replay.status_code == 401

        unauthorized_shutdown = client.post(
            "/api/desktop/shutdown",
            headers={**host, "Origin": tauri_origin},
        )
        assert unauthorized_shutdown.status_code == 401

        shutdown = client.post(
            "/api/desktop/shutdown",
            headers={**host, "Origin": tauri_origin, **valid_session},
        )
        assert shutdown.status_code == 202
        assert shutdown.json() == {"status": "shutting-down"}
        assert server.should_exit is True


def test_source_shutdown_requires_the_per_launch_token(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DILIGENT_DESKTOP", "false")
    monkeypatch.setenv("DILIGENT_DESKTOP_SESSION_SECRET", "source-session")
    application = FastAPI()
    runtime = DesktopRuntimeService(DesktopSessionSecurity())
    server = _DesktopServerStub()
    runtime.attach_server(server)
    application.state.desktop_runtime = runtime
    application.include_router(desktop_router, prefix="/api")

    with TestClient(application) as client:
        assert client.post("/api/desktop/shutdown").status_code == 401
        assert client.post(
            "/api/desktop/shutdown",
            headers={DESKTOP_SESSION_HEADER: "wrong"},
        ).status_code == 401
        response = client.post(
            "/api/desktop/shutdown",
            headers={DESKTOP_SESSION_HEADER: "source-session"},
        )

    assert response.status_code == 202
    assert server.should_exit is True
