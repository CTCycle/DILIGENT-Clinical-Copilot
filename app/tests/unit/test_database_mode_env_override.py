from __future__ import annotations

import json

from configurations.management import (
    build_settings_payload_from_json,
    environment_snapshot_from_os_env,
    load_configuration_data,
)
from domain.settings.environment import (
    DatabaseEnvironmentSnapshot,
    EnvironmentSnapshot,
)

###############################################################################
def _write_config(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")

###############################################################################
def _base_payload() -> dict:
    return {
        "database": {
            "embedded_database": False,
            "engine": "postgresql+psycopg",
            "host": "json-host",
            "port": 6432,
            "database_name": "json_db",
            "username": "json_user",
            "password": "json_secret",
            "ssl": True,
            "ssl_ca": "/json/ca.crt",
            "connect_timeout": 25,
            "insert_batch_size": 500,
            "insert_commit_interval": 5,
            "select_page_size": 2000,
        }
    }

###############################################################################
def test_database_settings_ignore_json_values_and_use_environment_snapshot(
    tmp_path,
) -> None:
    config_path = tmp_path / "configurations.json"
    _write_config(config_path, _base_payload())
    environment = EnvironmentSnapshot(
        ollama_url=None,
        ollama_host=None,
        ollama_port=None,
        database=DatabaseEnvironmentSnapshot(
            backend="postgresql",
            url="postgresql+psycopg://env_user:env_secret@env-host:5433/env_db",
            connect_timeout="18",
            write_batch_size="1200",
            read_page_size="2400",
        ),
    )

    payload = build_settings_payload_from_json(
        load_configuration_data(config_path),
        environment,
    )

    assert payload["database"] == {
        "backend": "postgresql",
        "url": "postgresql+psycopg://env_user:env_secret@env-host:5433/env_db",
        "sqlite_path": None,
        "write_batch_size": 1200,
        "read_page_size": 2400,
        "embedded_database": False,
        "engine": "postgresql+psycopg",
        "host": "env-host",
        "port": 5433,
        "database_name": "env_db",
        "username": "env_user",
        "password": "env_secret",
        "ssl": False,
        "ssl_ca": None,
        "connect_timeout": 18,
        "insert_batch_size": 1200,
        "insert_commit_interval": 5,
        "select_page_size": 2400,
    }

###############################################################################
def test_canonical_sqlite_database_environment_contract(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DATABASE_BACKEND", "sqlite")
    monkeypatch.setenv("DATABASE_SQLITE_PATH", "C:/data/diligent.db")
    monkeypatch.setenv("DATABASE_WRITE_BATCH_SIZE", "250")
    monkeypatch.setenv("DATABASE_READ_PAGE_SIZE", "500")

    payload = build_settings_payload_from_json(
        {},
        environment_snapshot_from_os_env(),
    )

    assert payload["database"]["backend"] == "sqlite"
    assert payload["database"]["sqlite_path"] == "C:/data/diligent.db"
    assert payload["database"]["write_batch_size"] == 250
    assert payload["database"]["read_page_size"] == 500

###############################################################################
def test_environment_snapshot_from_os_env_uses_domain_models(monkeypatch) -> None:
    env_names = [
        "OLLAMA_URL",
        "OLLAMA_HOST",
        "OLLAMA_PORT",
        "DATABASE_BACKEND",
        "DATABASE_URL",
        "DATABASE_CONNECT_TIMEOUT",
        "DATABASE_SQLITE_PATH",
        "DATABASE_WRITE_BATCH_SIZE",
        "DATABASE_READ_PAGE_SIZE",
    ]
    for name in env_names:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1")
    monkeypatch.setenv("OLLAMA_PORT", "11435")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@db.local:5432/appdb")
    monkeypatch.setenv("DATABASE_BACKEND", "postgresql")

    snapshot = environment_snapshot_from_os_env()

    assert isinstance(snapshot, EnvironmentSnapshot)
    assert isinstance(snapshot.database, DatabaseEnvironmentSnapshot)
    assert snapshot.ollama_url == "http://127.0.0.1:11434"
    assert snapshot.ollama_host == "127.0.0.1"
    assert snapshot.ollama_port == 11435
    assert snapshot.database.backend == "postgresql"
    assert snapshot.database.url == "postgresql://user:pass@db.local:5432/appdb"
    assert snapshot.database.connect_timeout is None
    assert snapshot.database.write_batch_size is None
    assert snapshot.database.read_page_size is None
