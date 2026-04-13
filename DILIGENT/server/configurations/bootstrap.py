from __future__ import annotations

from pathlib import Path
from typing import Any

from DILIGENT.server.common.constants import ENV_FILE_PATH
from DILIGENT.server.configurations import settings as settings_module
from DILIGENT.server.configurations.environment_bootstrap import (
    ensure_environment_loaded,
    reset_environment_bootstrap_for_tests,
)


###############################################################################
def initialize_environment() -> Path | None:
    return ensure_environment_loaded()


# -----------------------------------------------------------------------------
def get_app_settings():
    return settings_module.get_app_settings()


# -----------------------------------------------------------------------------
def get_server_settings(config_path: str | None = None):
    return settings_module.get_server_settings(config_path=config_path)


# -----------------------------------------------------------------------------
def get_configuration_block(block_name: str) -> dict[str, Any]:
    return settings_module.get_configuration_block(block_name)


# -----------------------------------------------------------------------------
def get_configuration_value(block_name: str, key: str, default: Any = None) -> Any:
    return settings_module.get_configuration_value(block_name, key, default)


# -----------------------------------------------------------------------------
def reload_settings_for_tests():
    return settings_module.reload_settings_for_tests()


# -----------------------------------------------------------------------------
def reset_app_settings_cache() -> None:
    settings_module.reset_app_settings_cache()


class _ServerSettingsProxy:
    def __getattr__(self, item: str) -> Any:
        return getattr(get_server_settings(), item)

    def __repr__(self) -> str:
        return repr(get_server_settings())


server_settings = _ServerSettingsProxy()
environment_settings = server_settings


# -----------------------------------------------------------------------------
def initialize_settings() -> None:
    get_app_settings()


__all__ = [
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
]

