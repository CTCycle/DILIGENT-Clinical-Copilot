# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import app as server_app_module
from common.security.desktop import DesktopSessionSecurity
from services.runtime.desktop import DesktopRuntimeService


def test_backend_serves_static_spa_and_keeps_api_not_found_responses(
    monkeypatch, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    client_root = tmp_path / "browser"
    asset_root = client_root / "assets"
    asset_root.mkdir(parents=True)
    index_file = client_root / "index.html"
    index_file.write_text('<html><script src="main-build.js"></script></html>', encoding="utf-8")
    (client_root / "main-build.js").write_text("window.appReady = true;", encoding="utf-8")
    (asset_root / "styles.css").write_text("body { color: teal; }", encoding="utf-8")

    monkeypatch.setattr(server_app_module, "CLIENT_DIST_PATH", client_root)
    monkeypatch.setattr(server_app_module, "CLIENT_ASSETS_PATH", asset_root)
    monkeypatch.setattr(server_app_module, "CLIENT_INDEX_FILE_PATH", index_file)
    monkeypatch.setattr(server_app_module, "initialize_settings", lambda: None)
    runtime = DesktopRuntimeService(DesktopSessionSecurity())
    monkeypatch.setattr(server_app_module, "get_desktop_runtime_service", lambda: runtime)
    client = TestClient(server_app_module.create_app())

    root = client.get("/")
    assert root.status_code == 200
    assert root.text == index_file.read_text(encoding="utf-8")

    javascript = client.get("/main-build.js")
    assert javascript.status_code == 200
    assert javascript.headers["content-type"].startswith(
        ("text/javascript", "application/javascript")
    )
    assert javascript.text == "window.appReady = true;"

    stylesheet = client.get("/assets/styles.css")
    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")

    route = client.get("/clinical-sessions/active")
    assert route.status_code == 200
    assert route.text == root.text

    assert client.get("/missing-build.js").status_code == 404
    api_missing = client.get("/api/not-a-route")
    assert api_missing.status_code == 404
    assert api_missing.json() == {"detail": "Not found"}

    with pytest.raises(HTTPException) as traversal:
        server_app_module._resolve_client_file("../secret.txt")
    assert traversal.value.status_code == 404
