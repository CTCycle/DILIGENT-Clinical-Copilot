from __future__ import annotations

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from DILIGENT.server.common.constants import ENV_FILE_PATH
from DILIGENT.server.configurations.sources import (
    CuratedDotenvSource,
    CuratedEnvironmentSource,
    JsonConfigurationSource,
)
from DILIGENT.server.domain.settings.configuration import (
    DatabaseSettings,
    DrugsMatcherSettings,
    ExternalDataSettings,
    FastAPISettings,
    IngestionSettings,
    JobsSettings,
    LLMRuntimeDefaults,
    RagSettings,
)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    fastapi_host: str = Field(default="127.0.0.1", alias="FASTAPI_HOST")
    fastapi_port: int = Field(default=8000, alias="FASTAPI_PORT")
    ui_host: str = Field(default="127.0.0.1", alias="UI_HOST")
    ui_port: int = Field(default=7861, alias="UI_PORT")
    reload: bool = Field(default=False, alias="RELOAD")
    diligent_tauri_mode: bool = Field(default=False, alias="DILIGENT_TAURI_MODE")
    keras_backend: str | None = Field(default=None, alias="KERAS_BACKEND")
    mpl_backend: str | None = Field(default=None, alias="MPLBACKEND")
    optional_dependencies: bool = Field(default=False, alias="OPTIONAL_DEPENDENCIES")
    ollama_url: str | None = Field(default=None, alias="OLLAMA_URL")
    ollama_host: str | None = Field(default=None, alias="OLLAMA_HOST")
    ollama_port: int | None = Field(default=None, alias="OLLAMA_PORT")
    ollama_prefetch_usage_window_s: float = Field(default=120.0, alias="OLLAMA_PREFETCH_USAGE_WINDOW_S")
    ollama_prefetch_transition_window_s: float = Field(default=60.0, alias="OLLAMA_PREFETCH_TRANSITION_WINDOW_S")
    ollama_prefetch_cooldown_s: float = Field(default=20.0, alias="OLLAMA_PREFETCH_COOLDOWN_S")
    ollama_ram_safety_ratio: float = Field(default=0.75, alias="OLLAMA_RAM_SAFETY_RATIO")
    ollama_vram_safety_ratio: float = Field(default=0.85, alias="OLLAMA_VRAM_SAFETY_RATIO")
    ollama_dual_resident_keep_alive: str = Field(default="4h", alias="OLLAMA_DUAL_RESIDENT_KEEP_ALIVE")
    ollama_single_resident_keep_alive: str = Field(default="30m", alias="OLLAMA_SINGLE_RESIDENT_KEEP_ALIVE")

    fastapi: FastAPISettings
    jobs: JobsSettings
    database: DatabaseSettings
    drugs_matcher: DrugsMatcherSettings
    rag: RagSettings
    external_data: ExternalDataSettings
    ingestion: IngestionSettings
    llm_defaults: LLMRuntimeDefaults

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        return (
            init_settings,
            env_settings,
            CuratedEnvironmentSource(settings_cls),
            dotenv_settings,
            CuratedDotenvSource(settings_cls),
            JsonConfigurationSource(settings_cls),
            file_secret_settings,
        )
