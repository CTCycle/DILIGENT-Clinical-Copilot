from __future__ import annotations

from threading import Lock
from typing import ClassVar

from pydantic import ValidationError

from DILIGENT.server.common import constants
from DILIGENT.server.configurations.bootstrap import ensure_environment_loaded
from DILIGENT.server.configurations.runtime_state import LLMRuntimeConfig
from DILIGENT.server.domain.settings.app import AppSettings
from DILIGENT.server.domain.settings.configuration import ServerSettings


_SETTINGS_LOCK = Lock()
_CACHED_APP_SETTINGS: AppSettings | None = None
_CACHED_SIGNATURE: tuple[str, str] | None = None


###############################################################################
def _build_path_scoped_settings_class(config_path: str) -> type[AppSettings]:
    class PathScopedAppSettings(AppSettings):
        _configuration_file: ClassVar[str] = config_path

    return PathScopedAppSettings


###############################################################################
def _load_app_settings(settings_cls: type[AppSettings]) -> AppSettings:
    ensure_environment_loaded()
    try:
        settings = settings_cls()
    except ValidationError as exc:
        raise RuntimeError(f"Invalid application settings: {exc}") from exc
    LLMRuntimeConfig.configure(settings.llm_defaults)
    return settings


###############################################################################
def get_app_settings() -> AppSettings:
    global _CACHED_APP_SETTINGS, _CACHED_SIGNATURE
    signature = (str(constants.CONFIGURATIONS_FILE), str(constants.ENV_FILE_PATH))
    with _SETTINGS_LOCK:
        if _CACHED_APP_SETTINGS is not None and _CACHED_SIGNATURE == signature:
            return _CACHED_APP_SETTINGS
        loaded = _load_app_settings(AppSettings)
        _CACHED_APP_SETTINGS = loaded
        _CACHED_SIGNATURE = signature
        return loaded


###############################################################################
def get_server_settings(config_path: str | None = None) -> ServerSettings:
    if config_path:
        scoped_class = _build_path_scoped_settings_class(config_path=config_path)
        return _load_app_settings(scoped_class).to_server_settings()
    return get_app_settings().to_server_settings()


###############################################################################
def reload_settings_for_tests() -> AppSettings:
    reset_app_settings_cache()
    return get_app_settings()


###############################################################################
def reset_app_settings_cache() -> None:
    global _CACHED_APP_SETTINGS, _CACHED_SIGNATURE
    with _SETTINGS_LOCK:
        _CACHED_APP_SETTINGS = None
        _CACHED_SIGNATURE = None
