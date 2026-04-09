from __future__ import annotations

from DILIGENT.server.configurations.settings import get_app_settings


###############################################################################
def tauri_mode_enabled() -> bool:
    return bool(get_app_settings().diligent_tauri_mode)
