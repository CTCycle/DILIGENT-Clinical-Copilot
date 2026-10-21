# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import base64
import ctypes
import ctypes.wintypes
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from common import paths
from domain.settings.configuration import DatabaseSettings

DATABASE_CONFIG_FILE_PATH = paths.DATA_PATH / "database_config.json"
DATABASE_CONFIG_SCHEMA_VERSION = 1


###############################################################################
class DatabaseConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal["sqlite", "postgresql"] = "sqlite"
    sqlite_path: str | None = None
    engine: str = "postgresql+psycopg"
    host: str | None = None
    port: int = Field(default=5432, ge=1, le=65535)
    database_name: str | None = None
    username: str | None = None
    password_protected: str | None = None
    ssl: bool = False
    ssl_ca: str | None = None
    connect_timeout: int = Field(default=10, ge=1, le=300)
    write_batch_size: int = Field(default=1000, ge=1, le=100000)
    read_page_size: int = Field(default=1000, ge=100, le=100000)

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_backend_fields(self) -> DatabaseConfiguration:
        if self.backend == "sqlite":
            if not self.sqlite_path or not self.sqlite_path.strip():
                raise ValueError("A SQLite database path is required.")
        else:
            if not self.host or not self.host.strip():
                raise ValueError("A PostgreSQL host is required.")
            if not self.database_name or not self.database_name.strip():
                raise ValueError("A PostgreSQL database name is required.")
            if not self.username or not self.username.strip():
                raise ValueError("A PostgreSQL username is required.")
            if self.engine not in {"postgresql+psycopg", "postgres+psycopg"}:
                raise ValueError("PostgreSQL must use the psycopg driver.")
        return self


###############################################################################
class DatabaseConfigurationFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = DATABASE_CONFIG_SCHEMA_VERSION
    database: DatabaseConfiguration


###############################################################################
def _dpapi_transform(value: bytes, *, protect: bool) -> bytes:
    if os.name != "nt":
        raise RuntimeError("Database credentials require Windows DPAPI.")

    ###############################################################################
    class DataBlob(ctypes.Structure):
        _fields_ = [
            ("cbData", ctypes.wintypes.DWORD),
            ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
        ]

    buffer = ctypes.create_string_buffer(value)
    source = DataBlob(
        len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
    )
    destination = DataBlob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptProtectData.restype = ctypes.wintypes.BOOL
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(DataBlob),
        ctypes.wintypes.LPCWSTR,
        ctypes.POINTER(DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.wintypes.DWORD,
        ctypes.POINTER(DataBlob),
    ]
    crypt32.CryptUnprotectData.restype = ctypes.wintypes.BOOL
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(DataBlob),
        ctypes.POINTER(ctypes.wintypes.LPWSTR),
        ctypes.POINTER(DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.wintypes.DWORD,
        ctypes.POINTER(DataBlob),
    ]
    kernel32.LocalFree.restype = ctypes.c_void_p
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    if protect:
        ok = crypt32.CryptProtectData(
            ctypes.byref(source),
            "DILIGENT database credential",
            None,
            None,
            None,
            1,
            ctypes.byref(destination),
        )
    else:
        ok = crypt32.CryptUnprotectData(
            ctypes.byref(source),
            None,
            None,
            None,
            None,
            1,
            ctypes.byref(destination),
        )
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(destination.pbData, destination.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(destination.pbData, ctypes.c_void_p))


###############################################################################
def protect_database_password(password: str) -> str:
    return base64.b64encode(_dpapi_transform(password.encode("utf-8"), protect=True)).decode(
        "ascii"
    )


###############################################################################
def unprotect_database_password(protected: str | None) -> str | None:
    if not protected:
        return None
    try:
        payload = base64.b64decode(protected, validate=True)
        return _dpapi_transform(payload, protect=False).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError("The saved database credential cannot be decrypted.") from exc


###############################################################################
def default_database_configuration() -> DatabaseConfiguration:
    return DatabaseConfiguration(
        backend="sqlite",
        sqlite_path=str(paths.DATA_PATH / "database.db"),
    )


###############################################################################
def _atomic_write(path: Path, content: bytes, *, secure: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        if secure and os.name != "nt":
            temporary_path.chmod(0o600)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


###############################################################################
def load_database_configuration() -> DatabaseConfiguration:
    path = DATABASE_CONFIG_FILE_PATH
    if path.is_file():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            stored = DatabaseConfigurationFile.model_validate(parsed)
            # Decryption here fails closed before any database initialization.
            unprotect_database_password(stored.database.password_protected)
            return stored.database
        except (OSError, json.JSONDecodeError, ValidationError, RuntimeError) as exc:
            raise RuntimeError(
                "Database configuration is invalid or cannot be decrypted. "
                f"Repair {path} before starting the application."
            ) from exc

    defaults = default_database_configuration()
    save_database_configuration(defaults)
    return defaults


###############################################################################
def save_database_configuration(configuration: DatabaseConfiguration) -> None:
    payload = DatabaseConfigurationFile(database=configuration).model_dump(mode="json")
    serialized = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    _atomic_write(DATABASE_CONFIG_FILE_PATH, serialized, secure=True)


###############################################################################
def database_settings(configuration: DatabaseConfiguration) -> DatabaseSettings:
    password = unprotect_database_password(configuration.password_protected)
    return DatabaseSettings(
        backend=configuration.backend,
        url=None,
        sqlite_path=configuration.sqlite_path,
        write_batch_size=configuration.write_batch_size,
        read_page_size=configuration.read_page_size,
        embedded_database=configuration.backend == "sqlite",
        engine=configuration.engine if configuration.backend == "postgresql" else None,
        host=configuration.host if configuration.backend == "postgresql" else None,
        port=configuration.port if configuration.backend == "postgresql" else None,
        database_name=configuration.database_name if configuration.backend == "postgresql" else None,
        username=configuration.username if configuration.backend == "postgresql" else None,
        password=password if configuration.backend == "postgresql" else None,
        ssl=configuration.ssl if configuration.backend == "postgresql" else False,
        ssl_ca=configuration.ssl_ca if configuration.backend == "postgresql" else None,
        connect_timeout=configuration.connect_timeout,
        insert_batch_size=configuration.write_batch_size,
        insert_commit_interval=5,
        select_page_size=configuration.read_page_size,
    )


###############################################################################
def configuration_from_request(payload: dict[str, Any], current: DatabaseConfiguration) -> DatabaseConfiguration:
    data = dict(payload)
    password = data.pop("password", None)
    keep_password = bool(data.pop("keep_password", True))
    if data.get("backend") == "postgresql":
        data["sqlite_path"] = None
        if password:
            data["password_protected"] = protect_database_password(str(password))
        elif keep_password and current.backend == "postgresql":
            data["password_protected"] = current.password_protected
        else:
            data["password_protected"] = None
    else:
        data["engine"] = "postgresql+psycopg"
        data["host"] = None
        data["port"] = 5432
        data["database_name"] = None
        data["username"] = None
        data["ssl"] = False
        data["ssl_ca"] = None
        data["password_protected"] = None
    data.setdefault("sqlite_path", current.sqlite_path)
    data.setdefault("engine", "postgresql+psycopg")
    data.setdefault("port", 5432)
    data.setdefault("connect_timeout", 10)
    data.setdefault("write_batch_size", current.write_batch_size)
    data.setdefault("read_page_size", current.read_page_size)
    return DatabaseConfiguration.model_validate(data)


###############################################################################
def public_database_configuration(configuration: DatabaseConfiguration) -> dict[str, Any]:
    return {
        "backend": configuration.backend,
        "sqlite_path": configuration.sqlite_path if configuration.backend == "sqlite" else None,
        "engine": configuration.engine if configuration.backend == "postgresql" else None,
        "host": configuration.host if configuration.backend == "postgresql" else None,
        "port": configuration.port if configuration.backend == "postgresql" else None,
        "database_name": configuration.database_name if configuration.backend == "postgresql" else None,
        "username": configuration.username if configuration.backend == "postgresql" else None,
        "password_configured": bool(configuration.password_protected),
        "ssl": configuration.ssl if configuration.backend == "postgresql" else False,
        "ssl_ca": configuration.ssl_ca if configuration.backend == "postgresql" else None,
        "connect_timeout": configuration.connect_timeout,
        "write_batch_size": configuration.write_batch_size,
        "read_page_size": configuration.read_page_size,
    }
