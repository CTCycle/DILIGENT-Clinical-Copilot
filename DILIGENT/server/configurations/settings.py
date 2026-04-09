from __future__ import annotations

from functools import lru_cache
from typing import Any

from DILIGENT.server.configurations.environment import initialize_environment
from DILIGENT.server.domain.settings.app import AppSettings
from DILIGENT.server.configurations.runtime_state import LLMRuntimeConfig


@lru_cache(maxsize=1)
def get_app_settings() -> AppSettings:
    initialize_environment()
    settings = AppSettings()
    LLMRuntimeConfig.configure(settings.llm_defaults)
    return settings


def reset_app_settings_cache() -> None:
    get_app_settings.cache_clear()


class AppSettingsProxy:
    def __getattr__(self, item: str) -> Any:
        return getattr(get_app_settings(), item)

    def __repr__(self) -> str:
        return repr(get_app_settings())
