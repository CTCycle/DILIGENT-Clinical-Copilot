from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from DILIGENT.server.common.constants import ENV_FILE_PATH
from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.domain.bootstrap import EnvironmentBootstrapState


# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _bootstrap_state() -> EnvironmentBootstrapState:
    return EnvironmentBootstrapState()


###############################################################################
def ensure_environment_loaded(*, force: bool = False) -> Path | None:
    state = _bootstrap_state()

    with state.lock:
        env_path = Path(ENV_FILE_PATH)
        if state.bootstrapped and not force:
            return env_path if env_path.exists() else None

        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
        else:
            logger.warning(".env file not found at: %s", env_path)

        state.bootstrapped = True
        return env_path if env_path.exists() else None


###############################################################################
def reset_environment_bootstrap_for_tests() -> None:
    state = _bootstrap_state()
    with state.lock:
        state.bootstrapped = False


# -----------------------------------------------------------------------------
def initialize_environment() -> Path | None:
    return ensure_environment_loaded()


# -----------------------------------------------------------------------------
def get_app_settings():
    from DILIGENT.server.configurations.settings import get_app_settings as _get_app_settings

    return _get_app_settings()


# -----------------------------------------------------------------------------
def get_server_settings(config_path: str | None = None):
    from DILIGENT.server.configurations.settings import get_server_settings as _get_server_settings

    return _get_server_settings(config_path=config_path)


# -----------------------------------------------------------------------------
def reload_settings_for_tests():
    from DILIGENT.server.configurations.settings import (
        reload_settings_for_tests as _reload_settings_for_tests,
    )

    return _reload_settings_for_tests()


# -----------------------------------------------------------------------------
def reset_app_settings_cache() -> None:
    from DILIGENT.server.configurations.settings import (
        reset_app_settings_cache as _reset_app_settings_cache,
    )

    _reset_app_settings_cache()


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
    "get_server_settings",
    "initialize_environment",
    "initialize_settings",
    "reload_settings_for_tests",
    "reset_app_settings_cache",
    "reset_environment_bootstrap_for_tests",
    "server_settings",
]
