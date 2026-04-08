from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from DILIGENT.server.common.constants import ENV_FILE_PATH


class EnvironmentSettings(BaseSettings):
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

    ollama_url: str | None = Field(default=None, alias="OLLAMA_URL")
    ollama_host: str | None = Field(default=None, alias="OLLAMA_HOST")
    ollama_port: int | None = Field(default=None, alias="OLLAMA_PORT")
