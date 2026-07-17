from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

###############################################################################
class DatabaseEnvironmentSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    embedded_database: str | None = None
    backend: str | None = None
    url: str | None = None
    sqlite_path: str | None = None
    connect_timeout: str | None = None
    write_batch_size: str | None = None
    read_page_size: str | None = None

###############################################################################
class EnvironmentSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    ollama_url: str | None
    ollama_host: str | None
    ollama_port: int | None
    database: DatabaseEnvironmentSnapshot = Field(
        default_factory=DatabaseEnvironmentSnapshot
    )
