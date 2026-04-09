from __future__ import annotations

from DILIGENT.server.configurations.settings import get_app_settings


def load_environment():
    return get_app_settings()
