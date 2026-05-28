from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

from common import constants
from common.constants import ENV_FILE_PATH, TRUTHY_ENV_VALUES
from configurations.environment import (
    ensure_environment_loaded,
    reset_environment_bootstrap_for_tests,
)
from configurations.management import ConfigurationManager
from domain.settings.configuration import ServerSettings


class _ConfigurationRuntimeState:
    def __init__(self) -> None:
        self.lock = RLock()
        self.manager: ConfigurationManager | None = None


@lru_cache(maxsize=1)
def _runtime_state() -> _ConfigurationRuntimeState:
    return _ConfigurationRuntimeState()


###############################################################################
def initialize_environment() -> Path | None:
    return ensure_environment_loaded()


def _build_settings_manager(config_path: str | None = None) -> ConfigurationManager:
    ensure_environment_loaded()
    return ConfigurationManager(config_path=config_path)


def get_configuration_manager(config_path: str | None = None) -> ConfigurationManager:
    if config_path:
        return _build_settings_manager(config_path=config_path)

    state = _runtime_state()
    default_path = Path(constants.CONFIGURATIONS_FILE)
    with state.lock:
        if state.manager is None or state.manager.config_path != default_path:
            state.manager = _build_settings_manager(config_path=str(default_path))
        return state.manager


def get_server_settings(config_path: str | None = None) -> ServerSettings:
    manager = get_configuration_manager(config_path=config_path)
    return manager.server_settings


def get_configuration_block(block_name: str) -> dict[str, Any]:
    return get_configuration_manager().get_block(block_name)


def get_configuration_value(block_name: str, key: str, default: Any = None) -> Any:
    return get_configuration_manager().get_value(block_name, key, default)


def reload_settings_for_tests(config_path: str | None = None) -> ServerSettings:
    if config_path is None:
        reset_app_settings_cache()
    return get_server_settings(config_path=config_path)


def reset_app_settings_cache() -> None:
    state = _runtime_state()
    with state.lock:
        state.manager = None


def tauri_mode_enabled() -> bool:
    value = (os.getenv("DILIGENT_TAURI_MODE") or "").strip().lower()
    return value in TRUTHY_ENV_VALUES


def initialize_settings() -> None:
    get_server_settings()


__all__ = [
    "ENV_FILE_PATH",
    "ensure_environment_loaded",
    "get_configuration_manager",
    "get_configuration_block",
    "get_configuration_value",
    "get_server_settings",
    "initialize_environment",
    "initialize_settings",
    "reload_settings_for_tests",
    "reset_app_settings_cache",
    "reset_environment_bootstrap_for_tests",
    "tauri_mode_enabled",
]
