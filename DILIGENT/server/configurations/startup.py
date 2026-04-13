from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from DILIGENT.server.common.constants import ENV_FILE_PATH, TRUTHY_ENV_VALUES
from DILIGENT.server.configurations.environment import (
    ensure_environment_loaded,
    reset_environment_bootstrap_for_tests,
)
from DILIGENT.server.configurations.management import ConfigurationManagement


###############################################################################
def initialize_environment() -> Path | None:
    return ensure_environment_loaded()


def get_app_settings():
    return ConfigurationManagement.get_app_settings()


def get_server_settings(config_path: str | None = None):
    return ConfigurationManagement.get_server_settings(config_path=config_path)


def get_configuration_block(block_name: str) -> dict[str, Any]:
    return ConfigurationManagement.get_configuration_block(block_name)


def get_configuration_value(block_name: str, key: str, default: Any = None) -> Any:
    return ConfigurationManagement.get_configuration_value(block_name, key, default)


def reload_settings_for_tests():
    return ConfigurationManagement.reload_settings_for_tests()


def reset_app_settings_cache() -> None:
    ConfigurationManagement.reset_app_settings_cache()


def tauri_mode_enabled() -> bool:
    value = (os.getenv("DILIGENT_TAURI_MODE") or "").strip().lower()
    return value in TRUTHY_ENV_VALUES


class _ServerSettingsProxy:
    def __getattr__(self, item: str) -> Any:
        return getattr(get_server_settings(), item)

    def __repr__(self) -> str:
        return repr(get_server_settings())


server_settings = _ServerSettingsProxy()
environment_settings = server_settings


def initialize_settings() -> None:
    get_app_settings()


__all__ = [
    "ENV_FILE_PATH",
    "ensure_environment_loaded",
    "environment_settings",
    "get_app_settings",
    "get_configuration_block",
    "get_configuration_value",
    "get_server_settings",
    "initialize_environment",
    "initialize_settings",
    "reload_settings_for_tests",
    "reset_app_settings_cache",
    "reset_environment_bootstrap_for_tests",
    "server_settings",
    "tauri_mode_enabled",
]
