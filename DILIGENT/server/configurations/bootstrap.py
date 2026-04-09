from __future__ import annotations

from DILIGENT.server.configurations.settings import (
    AppSettingsProxy,
    get_app_settings,
    reset_app_settings_cache,
)

server_settings = AppSettingsProxy()
environment_settings = AppSettingsProxy()


def get_server_settings() -> AppSettingsProxy:
    return server_settings


def initialize_settings() -> None:
    get_app_settings()


__all__ = [
    "environment_settings",
    "get_app_settings",
    "get_server_settings",
    "initialize_settings",
    "reset_app_settings_cache",
    "server_settings",
]
