from __future__ import annotations

import json

from DILIGENT.server.common import constants
from DILIGENT.server.configurations import startup as bootstrap
from DILIGENT.server.configurations.startup import get_server_settings, reset_app_settings_cache


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


def test_database_settings_are_loaded_from_json_without_env_overlap(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "configurations.json"
    _write_config(config_path, _base_payload())
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("DB_HOST=dotenv-host\n", encoding="utf-8")
    monkeypatch.setattr(constants, "CONFIGURATIONS_FILE", str(config_path))
    monkeypatch.setattr(constants, "ENV_FILE_PATH", str(dotenv_path))
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", str(dotenv_path))
    monkeypatch.setenv("DB_HOST", "os-host")

    bootstrap.reset_environment_bootstrap_for_tests()
    reset_app_settings_cache()
    settings = get_server_settings()

    assert settings.database.host == "json-host"

    monkeypatch.delenv("DB_HOST", raising=False)
    bootstrap.reset_environment_bootstrap_for_tests()
    reset_app_settings_cache()
    settings = get_server_settings()
    assert settings.database.host == "json-host"

    bootstrap.reset_environment_bootstrap_for_tests()
    reset_app_settings_cache()

    monkeypatch.setattr(constants, "ENV_FILE_PATH", str(tmp_path / ".missing.env"))
    monkeypatch.setattr(bootstrap, "ENV_FILE_PATH", str(tmp_path / ".missing.env"))
    monkeypatch.delenv("DB_HOST", raising=False)
    bootstrap.reset_environment_bootstrap_for_tests()
    reset_app_settings_cache()
    settings = get_server_settings()
    assert settings.database.host == "json-host"


