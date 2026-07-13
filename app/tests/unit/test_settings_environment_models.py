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
