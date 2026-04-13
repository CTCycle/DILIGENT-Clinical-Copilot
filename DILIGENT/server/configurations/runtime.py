from __future__ import annotations

import os

from DILIGENT.server.common.constants import TRUTHY_ENV_VALUES


###############################################################################
def tauri_mode_enabled() -> bool:
    value = (os.getenv("DILIGENT_TAURI_MODE") or "").strip().lower()
    return value in TRUTHY_ENV_VALUES
