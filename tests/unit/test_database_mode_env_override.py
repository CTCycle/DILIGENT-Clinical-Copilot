from __future__ import annotations

from DILIGENT.server.configurations.bootstrap import build_database_settings
from DILIGENT.server.configurations.env_loader import load_environment


###############################################################################
def test_db_embedded_mode_comes_from_json(monkeypatch) -> None:
    payload = {
        "embedded_database": True,
        "engine": "postgres",
        "host": "json-host",
        "port": 5432,
        "database_name": "json_db",
        "username": "json_user",
        "password": "json_pass",
        "ssl": True,
        "connect_timeout": 21,
        "insert_batch_size": 42,
    }

    monkeypatch.setenv("DB_EMBEDDED", "false")
    monkeypatch.setenv("DB_HOST", "ignored-host")

    settings = build_database_settings(payload, load_environment())

    assert settings.embedded_database is True
    assert settings.engine is None
    assert settings.host is None
    assert settings.port is None
    assert settings.database_name is None
    assert settings.username is None
    assert settings.password is None
    assert settings.ssl is False
    assert settings.connect_timeout == 21
    assert settings.insert_batch_size == 42


###############################################################################
def test_external_db_fields_come_from_json(monkeypatch) -> None:
    payload = {
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
    }

    monkeypatch.setenv("DB_EMBEDDED", "true")
    monkeypatch.setenv("DB_ENGINE", "sqlite")
    monkeypatch.setenv("DB_HOST", "ignored-host")
    monkeypatch.setenv("DB_PORT", "9999")
    monkeypatch.setenv("DB_NAME", "ignored-db")
    monkeypatch.setenv("DB_USER", "ignored-user")
    monkeypatch.setenv("DB_PASSWORD", "ignored-secret")
    monkeypatch.setenv("DB_SSL", "false")
    monkeypatch.setenv("DB_SSL_CA", "/ignored/ca.crt")
    monkeypatch.setenv("DB_CONNECT_TIMEOUT", "5")
    monkeypatch.setenv("DB_INSERT_BATCH_SIZE", "10")

    settings = build_database_settings(payload, load_environment())

    assert settings.embedded_database is False
    assert settings.engine == "postgresql+psycopg"
    assert settings.host == "json-host"
    assert settings.port == 6432
    assert settings.database_name == "json_db"
    assert settings.username == "json_user"
    assert settings.password == "json_secret"
    assert settings.ssl is True
    assert settings.ssl_ca == "/json/ca.crt"
    assert settings.connect_timeout == 25
    assert settings.insert_batch_size == 500
