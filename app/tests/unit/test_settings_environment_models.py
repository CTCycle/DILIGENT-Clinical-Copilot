from __future__ import annotations

from configurations.management import environment_snapshot_from_os_env
from domain.settings.environment import (
    DatabaseEnvironmentSnapshot,
    EnvironmentSnapshot,
)


###############################################################################
def test_environment_snapshot_from_os_env_uses_domain_models(monkeypatch) -> None:
    env_names = [
        "OLLAMA_URL",
        "OLLAMA_HOST",
        "OLLAMA_PORT",
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
    ]
    for name in env_names:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1")
    monkeypatch.setenv("OLLAMA_PORT", "11435")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@db.local:5432/appdb")
    monkeypatch.setenv("DATABASE_ENGINE", "postgresql")
    monkeypatch.setenv("DATABASE_PORT", "5432")
    monkeypatch.setenv("DATABASE_NAME", "appdb")
    monkeypatch.setenv("DATABASE_USERNAME", "user")
    monkeypatch.setenv("DATABASE_PASSWORD", "pass")
    monkeypatch.setenv("DATABASE_SSL", "require")
    monkeypatch.setenv("DATABASE_INSERT_BATCH_SIZE", "2000")
    monkeypatch.setenv("DATABASE_SELECT_PAGE_SIZE", "3000")

    snapshot = environment_snapshot_from_os_env()

    assert isinstance(snapshot, EnvironmentSnapshot)
    assert isinstance(snapshot.database, DatabaseEnvironmentSnapshot)
    assert snapshot.ollama_url == "http://127.0.0.1:11434"
    assert snapshot.ollama_host == "127.0.0.1"
    assert snapshot.ollama_port == 11435
    assert snapshot.database.embedded_database is None
    assert snapshot.database.url == "postgresql://user:pass@db.local:5432/appdb"
    assert snapshot.database.engine == "postgresql"
    assert snapshot.database.host is None
    assert snapshot.database.port == "5432"
    assert snapshot.database.database_name == "appdb"
    assert snapshot.database.username == "user"
    assert snapshot.database.password == "pass"
    assert snapshot.database.ssl == "require"
    assert snapshot.database.ssl_ca is None
    assert snapshot.database.connect_timeout is None
    assert snapshot.database.insert_batch_size == "2000"
    assert snapshot.database.insert_commit_interval is None
    assert snapshot.database.select_page_size == "3000"
