from __future__ import annotations

import json

from DILIGENT.server.common import constants
from DILIGENT.server.configurations.bootstrap import get_app_settings, reset_app_settings_cache
from DILIGENT.server.configurations import sources


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


def test_database_env_override_precedence_os_over_dotenv_over_json(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "configurations.json"
    _write_config(config_path, _base_payload())
    monkeypatch.setattr(constants, "CONFIGURATIONS_FILE", str(config_path))
    monkeypatch.setattr(
        sources,
        "_read_dotenv",
        lambda: {"DB_HOST": "dotenv-host"},
    )
    monkeypatch.setenv("DB_HOST", "os-host")

    reset_app_settings_cache()
    settings = get_app_settings()

    assert settings.database.host == "os-host"

    monkeypatch.delenv("DB_HOST", raising=False)
    reset_app_settings_cache()
    settings = get_app_settings()
    assert settings.database.host == "dotenv-host"

    monkeypatch.setattr(sources, "_read_dotenv", lambda: {})
    reset_app_settings_cache()
    settings = get_app_settings()
    assert settings.database.host == "json-host"
