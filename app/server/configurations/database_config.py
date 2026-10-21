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
from urllib.parse import unquote, urlparse

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from common import paths
from domain.settings.configuration import DatabaseSettings

DATABASE_CONFIG_FILE_PATH = paths.DATA_PATH / "database_config.json"
DATABASE_CONFIG_SCHEMA_VERSION = 1
LEGACY_DATABASE_ENVIRONMENT_KEYS = frozenset(
    {
        "EMBEDDED_DATABASE",
        "DATABASE_BACKEND",
        "DATABASE_URL",
        "DATABASE_SQLITE_PATH",
        "DILIGENT_SQLITE_PATH",
        "DATABASE_ENGINE",
        "DATABASE_HOST",
        "DATABASE_PORT",
        "DATABASE_NAME",
        "DATABASE_USERNAME",
        "DATABASE_PASSWORD",
        "DATABASE_SSL",
        "DATABASE_SSL_CA",
        "DATABASE_CONNECT_TIMEOUT",
        "DATABASE_WRITE_BATCH_SIZE",
        "DATABASE_READ_PAGE_SIZE",
    }
)


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
def _legacy_values() -> dict[str, str]:
    result: dict[str, str] = {}
    if paths.ENV_FILE_PATH.is_file():
        for key, value in dotenv_values(paths.ENV_FILE_PATH).items():
            if key in LEGACY_DATABASE_ENVIRONMENT_KEYS and value is not None:
                result[key] = str(value)
    for key in LEGACY_DATABASE_ENVIRONMENT_KEYS:
        value = os.environ.get(key)
        if value is not None:
            result[key] = value
    return result


###############################################################################
def _legacy_database_configuration(values: dict[str, str]) -> DatabaseConfiguration:
    embedded = values.get("EMBEDDED_DATABASE", "").strip().lower()
    backend = values.get("DATABASE_BACKEND", "").strip().lower()
    if embedded:
        if embedded not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
            raise RuntimeError("Legacy EMBEDDED_DATABASE must be a boolean value.")
        backend = "sqlite" if embedded in {"true", "1", "yes", "on"} else "postgresql"
    if not backend:
        backend = "postgresql" if values.get("DATABASE_URL") else "sqlite"
    if backend not in {"sqlite", "postgresql"}:
        raise RuntimeError("Legacy database backend is not supported.")

    url = urlparse(values.get("DATABASE_URL", ""))
    sqlite_path = (
        values.get("DILIGENT_SQLITE_PATH")
        or values.get("DATABASE_SQLITE_PATH")
        or str(paths.DATA_PATH / "database.db")
    )
    if backend == "sqlite":
        return DatabaseConfiguration(
            backend="sqlite",
            sqlite_path=str(Path(sqlite_path).expanduser().resolve()),
            connect_timeout=_legacy_int(values.get("DATABASE_CONNECT_TIMEOUT"), 10, 1, 300),
            write_batch_size=_legacy_int(values.get("DATABASE_WRITE_BATCH_SIZE"), 1000, 1, 100000),
            read_page_size=_legacy_int(values.get("DATABASE_READ_PAGE_SIZE"), 1000, 100, 100000),
        )

    password = values.get("DATABASE_PASSWORD")
    if password is None and url.password is not None:
        password = unquote(url.password)
    protected = protect_database_password(password) if password else None
    scheme = values.get("DATABASE_ENGINE") or url.scheme or "postgresql+psycopg"
    if scheme in {"postgres", "postgresql"}:
        scheme = "postgresql+psycopg"
    return DatabaseConfiguration(
        backend="postgresql",
        sqlite_path=None,
        engine=scheme,
        host=values.get("DATABASE_HOST") or url.hostname,
        port=_legacy_int(values.get("DATABASE_PORT"), url.port or 5432, 1, 65535),
        database_name=values.get("DATABASE_NAME") or (url.path.lstrip("/") or None),
        username=values.get("DATABASE_USERNAME") or (unquote(url.username) if url.username else None),
        password_protected=protected,
        ssl=values.get("DATABASE_SSL", "").strip().lower() in {"true", "1", "yes", "on"},
        ssl_ca=values.get("DATABASE_SSL_CA") or None,
        connect_timeout=_legacy_int(values.get("DATABASE_CONNECT_TIMEOUT"), 10, 1, 300),
        write_batch_size=_legacy_int(values.get("DATABASE_WRITE_BATCH_SIZE"), 1000, 1, 100000),
        read_page_size=_legacy_int(values.get("DATABASE_READ_PAGE_SIZE"), 1000, 100, 100000),
    )


###############################################################################
def _legacy_int(value: str | None, default: int, minimum: int, maximum: int) -> int:
    try:
        resolved = int(value) if value else default
    except (TypeError, ValueError):
        resolved = default
    return min(max(resolved, minimum), maximum)


###############################################################################
def _scrub_legacy_environment_file() -> None:
    path = paths.ENV_FILE_PATH
    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [
        line
        for line in lines
        if line.split("=", maxsplit=1)[0].strip().removeprefix("export ")
        not in LEGACY_DATABASE_ENVIRONMENT_KEYS
    ]
    if len(kept) == len(lines):
        return
    _atomic_write(path, "".join(kept).encode("utf-8"), secure=False)


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
            _scrub_legacy_environment_file()
            return stored.database
        except (OSError, json.JSONDecodeError, ValidationError, RuntimeError) as exc:
            raise RuntimeError(
                "Database configuration is invalid or cannot be decrypted. "
                f"Repair {path} before starting the application."
            ) from exc

    legacy = _legacy_database_configuration(_legacy_values())
    save_database_configuration(legacy)
    _scrub_legacy_environment_file()
    return legacy


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
