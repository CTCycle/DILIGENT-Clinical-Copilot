from __future__ import annotations

from DILIGENT.server.configurations.bootstrap import build_database_settings
from DILIGENT.server.configurations.env_loader import load_environment


###############################################################################
def test_db_embedded_env_overrides_json_default(monkeypatch) -> None:
    payload = {
        "embedded_database": False,
        "engine": "postgres",
        "host": "json-host",
        "port": 5432,
        "database_name": "json_db",
        "username": "json_user",
        "password": "json_pass",
        "ssl": True,
    }

    monkeypatch.setenv("DB_EMBEDDED", "true")

    settings = build_database_settings(payload, load_environment())

    assert settings.embedded_database is True
    assert settings.engine is None
    assert settings.host is None
    assert settings.port is None
    assert settings.database_name is None
    assert settings.username is None
    assert settings.password is None
    assert settings.ssl is False


###############################################################################
def test_external_db_env_fields_are_honored_when_embedded_disabled(monkeypatch) -> None:
    payload = {
        "embedded_database": True,
        "engine": "sqlite",
        "host": "json-host",
        "port": 1111,
        "database_name": "json_db",
        "username": "json_user",
        "password": "json_pass",
        "ssl": False,
        "connect_timeout": 5,
        "insert_batch_size": 10,
    }

    monkeypatch.setenv("DB_EMBEDDED", "false")
    monkeypatch.setenv("DB_ENGINE", "postgresql+psycopg")
    monkeypatch.setenv("DB_HOST", "db.internal")
    monkeypatch.setenv("DB_PORT", "6432")
    monkeypatch.setenv("DB_NAME", "prod_db")
    monkeypatch.setenv("DB_USER", "prod_user")
    monkeypatch.setenv("DB_PASSWORD", "prod_secret")
    monkeypatch.setenv("DB_SSL", "true")
    monkeypatch.setenv("DB_SSL_CA", "/run/secrets/ca.crt")
    monkeypatch.setenv("DB_CONNECT_TIMEOUT", "25")
    monkeypatch.setenv("DB_INSERT_BATCH_SIZE", "500")

    settings = build_database_settings(payload, load_environment())

    assert settings.embedded_database is False
    assert settings.engine == "postgresql+psycopg"
    assert settings.host == "db.internal"
    assert settings.port == 6432
    assert settings.database_name == "prod_db"
    assert settings.username == "prod_user"
    assert settings.password == "prod_secret"
    assert settings.ssl is True
    assert settings.ssl_ca == "/run/secrets/ca.crt"
    assert settings.connect_timeout == 25
    assert settings.insert_batch_size == 500
