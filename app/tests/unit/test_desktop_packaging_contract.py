# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


###############################################################################
def test_desktop_runtime_archive_excludes_frontend_assets() -> None:
    payload = json.loads(
        (ROOT / "app/desktop/build/runtime_payload.json").read_text(encoding="utf-8")
    )
    sources = {entry["source"] for entry in payload["entries"]}
    assert not any(source.startswith("app/client/dist") for source in sources)

    validator = (ROOT / "app/desktop/build/validate_runtime_archive.py").read_text(
        encoding="utf-8"
    )
    assert 'FORBIDDEN_PREFIXES = ("app/client/dist/browser/",)' in validator


###############################################################################
def test_tauri_release_bundles_frontend_and_uses_a_local_session_api() -> None:
    launcher = (ROOT / "start_on_windows.ps1").read_text(encoding="utf-8")
    tauri_main = (ROOT / "app/desktop/src-tauri/src/main.rs").read_text(
        encoding="utf-8"
    )
    backend = (ROOT / "app/desktop/src-tauri/src/backend.rs").read_text(
        encoding="utf-8"
    )
    configuration = json.loads(
        (ROOT / "app/desktop/src-tauri/tauri.conf.json").read_text(encoding="utf-8")
    )

    assert "New-TauriFrontendAssets" in launcher
    assert "$configuration.build.frontendDist" in launcher
    assert "splash.html" in launcher
    assert "http://tauri.localhost/?apiOrigin=" in tauri_main
    assert "Some(&self.session_secret)" in backend
    assert "X-DILIGENT-Desktop-Session" in backend
    assert "http://tauri.localhost" in backend
    assert "http://127.0.0.1:*" in configuration["app"]["security"]["csp"]


###############################################################################
def test_packaged_csp_keeps_runtime_component_styles_and_avoids_inline_handlers() -> None:
    # Tauri's nonce injection would make WebView2 ignore 'unsafe-inline' and block
    # Angular's runtime <style> tags; critical-CSS inlining adds an inline onload
    # handler that the packaged CSP also blocks, leaving the UI unstyled.
    configuration = json.loads(
        (ROOT / "app/desktop/src-tauri/tauri.conf.json").read_text(encoding="utf-8")
    )
    angular = json.loads(
        (ROOT / "app/client/angular.json").read_text(encoding="utf-8")
    )
    production = angular["projects"]["diligent-client"]["architect"]["build"][
        "configurations"
    ]["production"]

    assert configuration["app"]["security"]["dangerousDisableAssetCspModification"] == [
        "style-src"
    ]
    assert production["optimization"]["styles"]["inlineCritical"] is False
