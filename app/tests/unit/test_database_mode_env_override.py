from __future__ import annotations

import json

from configurations.management import (
    build_settings_payload_from_json,
    environment_snapshot_from_os_env,
    load_configuration_data,
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
def test_database_settings_are_loaded_from_json_without_env_overlap(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "configurations.json"
    _write_config(config_path, _base_payload())
    monkeypatch.delenv("DATABASE_BACKEND", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_SQLITE_PATH", raising=False)
    monkeypatch.delenv("DATABASE_WRITE_BATCH_SIZE", raising=False)
    monkeypatch.delenv("DATABASE_READ_PAGE_SIZE", raising=False)
    monkeypatch.setenv("EMBEDDED_DATABASE", "false")
    monkeypatch.setenv("DATABASE_ENGINE", "postgresql+psycopg")
    monkeypatch.setenv("DATABASE_HOST", "env-host")
    monkeypatch.setenv("DATABASE_PORT", "5433")
    monkeypatch.setenv("DATABASE_NAME", "env_db")
    monkeypatch.setenv("DATABASE_USERNAME", "env_user")
    monkeypatch.setenv("DATABASE_PASSWORD", "env_secret")
    monkeypatch.setenv("DATABASE_SSL", "true")
    monkeypatch.setenv("DATABASE_SSL_CA", "/env/ca.crt")
    monkeypatch.setenv("DATABASE_CONNECT_TIMEOUT", "18")
    monkeypatch.setenv("DATABASE_INSERT_BATCH_SIZE", "1200")
    monkeypatch.setenv("DATABASE_INSERT_COMMIT_INTERVAL", "7")
    monkeypatch.setenv("DATABASE_SELECT_PAGE_SIZE", "2400")

    payload = build_settings_payload_from_json(
        load_configuration_data(config_path),
        environment_snapshot_from_os_env(),
    )

    assert payload["database"] == {
        "backend": "postgresql",
        "url": None,
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
        "ssl": True,
        "ssl_ca": "/env/ca.crt",
        "connect_timeout": 18,
        "insert_batch_size": 1200,
        "insert_commit_interval": 7,
        "select_page_size": 2400,
    }


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
