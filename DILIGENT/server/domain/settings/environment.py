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
    ui_port: int = Field(default=5173, alias="UI_PORT")
    reload: bool = Field(default=False, alias="RELOAD")
    diligent_cloud_mode: bool = Field(default=False, alias="DILIGENT_CLOUD_MODE")
    diligent_tauri_mode: bool = Field(default=False, alias="DILIGENT_TAURI_MODE")

    db_embedded: bool = Field(default=True, alias="DB_EMBEDDED")
    db_engine: str | None = Field(default=None, alias="DB_ENGINE")
    db_host: str | None = Field(default=None, alias="DB_HOST")
    db_port: int | None = Field(default=None, alias="DB_PORT")
    db_name: str | None = Field(default=None, alias="DB_NAME")
    db_user: str | None = Field(default=None, alias="DB_USER")
    db_password: str | None = Field(default=None, alias="DB_PASSWORD")
    db_ssl: bool = Field(default=False, alias="DB_SSL")
    db_ssl_ca: str | None = Field(default=None, alias="DB_SSL_CA")
    db_connect_timeout: int = Field(default=10, alias="DB_CONNECT_TIMEOUT")
    db_insert_batch_size: int | None = Field(default=None, alias="DB_INSERT_BATCH_SIZE")

    ollama_url: str | None = Field(default=None, alias="OLLAMA_URL")
    ollama_host: str | None = Field(default=None, alias="OLLAMA_HOST")
    ollama_port: int | None = Field(default=None, alias="OLLAMA_PORT")
