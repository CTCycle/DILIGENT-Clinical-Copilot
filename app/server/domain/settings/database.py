# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


###############################################################################
class DatabaseConfigurationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["sqlite", "postgresql"]
    sqlite_path: str | None = None
    engine: Literal["postgresql+psycopg", "postgres+psycopg"] = "postgresql+psycopg"
    host: str | None = None
    port: int = Field(default=5432, ge=1, le=65535)
    database_name: str | None = None
    username: str | None = None
    password: str | None = Field(default=None, repr=False)
    keep_password: bool = True
    ssl: bool = False
    ssl_ca: str | None = None
    connect_timeout: int = Field(default=10, ge=1, le=300)
    write_batch_size: int = Field(default=1000, ge=1, le=100000)
    read_page_size: int = Field(default=1000, ge=100, le=100000)

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_selected_backend(self) -> DatabaseConfigurationRequest:
        if self.backend == "sqlite":
            if not self.sqlite_path or not self.sqlite_path.strip():
                raise ValueError("A SQLite database path is required.")
        elif not self.host or not self.host.strip():
            raise ValueError("A PostgreSQL host is required.")
        elif not self.database_name or not self.database_name.strip():
            raise ValueError("A PostgreSQL database name is required.")
        elif not self.username or not self.username.strip():
            raise ValueError("A PostgreSQL username is required.")
        return self


###############################################################################
class DatabaseConfigurationResponse(BaseModel):
    backend: Literal["sqlite", "postgresql"]
    sqlite_path: str | None = None
    engine: str | None = None
    host: str | None = None
    port: int | None = None
    database_name: str | None = None
    username: str | None = None
    password_configured: bool = False
    ssl: bool = False
    ssl_ca: str | None = None
    connect_timeout: int
    write_batch_size: int
    read_page_size: int


###############################################################################
class DatabaseConfigurationStateResponse(BaseModel):
    configuration: DatabaseConfigurationResponse
    active_backend: Literal["sqlite", "postgresql"]
    status: Literal["active"] = "active"
    runtime_switching: bool = True


###############################################################################
class DatabaseConnectionTestResponse(BaseModel):
    reachable: bool
    schema_ready: bool
    message: str


###############################################################################
class DatabaseSwitchResponse(BaseModel):
    active_backend: Literal["sqlite", "postgresql"]
    settings_reloaded: bool = True
    message: str
