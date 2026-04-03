from __future__ import annotations

import os

from DILIGENT.server.common.constants import FALSY_ENV_VALUES, TRUTHY_ENV_VALUES


###############################################################################
def tauri_mode_enabled() -> bool:
    value = os.getenv("DILIGENT_TAURI_MODE", "false").strip().lower()
    return value in TRUTHY_ENV_VALUES


###############################################################################
def cloud_mode_enabled() -> bool:
    value = os.getenv("DILIGENT_CLOUD_MODE", "false").strip().lower()
    if value in TRUTHY_ENV_VALUES:
        return True
    if value in FALSY_ENV_VALUES:
        return False
    return False
