from __future__ import annotations

from functools import lru_cache
from threading import RLock
from typing import Any

from common.utils.logger import configure_logging
from configurations.environment import ensure_environment_loaded
from configurations.management import ConfigurationManager
from domain.settings.configuration import ServerSettings

###############################################################################
class _ConfigurationRuntimeState:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.lock = RLock()
        self.manager: ConfigurationManager | None = None

###############################################################################
@lru_cache(maxsize=1)
def _runtime_state() -> _ConfigurationRuntimeState:
    return _ConfigurationRuntimeState()

###############################################################################
def _build_settings_manager(
    persisted_payload: dict[str, Any] | None = None,
) -> ConfigurationManager:
    ensure_environment_loaded()
    return ConfigurationManager(persisted_payload=persisted_payload)

###############################################################################
def get_configuration_manager() -> ConfigurationManager:
    state = _runtime_state()
    with state.lock:
        if state.manager is None:
            state.manager = _build_settings_manager()
        return state.manager

###############################################################################
def get_server_settings() -> ServerSettings:
    manager = get_configuration_manager()
    return manager.server_settings

###############################################################################
def get_configuration_block(block_name: str) -> dict[str, Any]:
    return get_configuration_manager().get_block(block_name)

###############################################################################
def get_configuration_value(block_name: str, key: str, default: Any = None) -> Any:
    return get_configuration_manager().get_value(block_name, key, default)

###############################################################################
def reload_persisted_settings() -> ServerSettings:
    manager = get_configuration_manager()
    return manager.reload_from_database()

###############################################################################
def reload_settings_for_tests(
    persisted_payload: dict[str, Any] | None = None,
) -> ServerSettings:
    reset_app_settings_cache()
    return _build_settings_manager(persisted_payload=persisted_payload).server_settings

###############################################################################
def reset_app_settings_cache() -> None:
    state = _runtime_state()
    with state.lock:
        state.manager = None

###############################################################################
def initialize_settings() -> None:
    configure_logging()
    get_server_settings()


__all__ = [
    "ensure_environment_loaded",
    "get_configuration_manager",
    "get_configuration_block",
    "get_configuration_value",
    "get_server_settings",
    "initialize_settings",
    "reload_persisted_settings",
    "reload_settings_for_tests",
    "reset_app_settings_cache",
]
