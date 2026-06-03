from __future__ import annotations

import json

from configurations.management import (
    build_settings_payload_from_json,
    environment_snapshot_from_os_env,
    load_configuration_data,
)


def _write_config(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


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


def test_database_settings_are_loaded_from_json_without_env_overlap(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "configurations.json"
    _write_config(config_path, _base_payload())
    monkeypatch.setenv("DB_HOST", "os-host")

    for name in [
        "EMBEDDED_DATABASE",
        "DATABASE_URL",
        "DATABASE_ENGINE",
        "DATABASE_HOST",
        "DATABASE_PORT",
        "DATABASE_NAME",
        "DATABASE_USERNAME",
        "DATABASE_PASSWORD",
        "DATABASE_SSL",
        "DATABASE_SSL_CA",
        "DATABASE_CONNECT_TIMEOUT",
        "DATABASE_INSERT_BATCH_SIZE",
        "DATABASE_INSERT_COMMIT_INTERVAL",
        "DATABASE_SELECT_PAGE_SIZE",
    ]:
        monkeypatch.delenv(name, raising=False)

    payload = build_settings_payload_from_json(
        load_configuration_data(config_path),
        environment_snapshot_from_os_env(),
    )

    assert payload["database"]["host"] == "json-host"

