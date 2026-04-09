from __future__ import annotations

from DILIGENT.server.domain.settings.configuration import DatabaseSettings
from DILIGENT.server.repositories.database import initializer


def _sqlite_settings() -> DatabaseSettings:
    return DatabaseSettings(
        embedded_database=True,
        engine="postgres",
        host="127.0.0.1",
        port=5432,
        database_name="diligent",
        username="postgres",
        password="",
        ssl=False,
        ssl_ca=None,
        connect_timeout=10,
        insert_batch_size=1000,
        insert_commit_interval=100,
        select_page_size=1000,
    )


def _postgres_settings() -> DatabaseSettings:
    return DatabaseSettings(
        embedded_database=False,
        engine="postgres",
        host="127.0.0.1",
        port=5432,
        database_name="diligent",
        username="postgres",
        password="",
        ssl=False,
        ssl_ca=None,
        connect_timeout=10,
        insert_batch_size=1000,
        insert_commit_interval=100,
        select_page_size=1000,
    )


def test_run_database_initialization_uses_sqlite_path_when_embedded(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[str] = []
    settings = _sqlite_settings()

    monkeypatch.setattr(initializer, "server_settings", type("S", (), {"database": settings})())
    monkeypatch.setattr(
        initializer,
        "initialize_sqlite_database",
        lambda _settings: calls.append("sqlite"),
    )
    monkeypatch.setattr(
        initializer,
        "ensure_postgres_database",
        lambda _settings: calls.append("postgres"),
    )

    initializer.run_database_initialization()

    assert calls == ["sqlite"]


def test_run_database_initialization_uses_postgres_path_when_external(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[str] = []
    settings = _postgres_settings()

    monkeypatch.setattr(initializer, "server_settings", type("S", (), {"database": settings})())
    monkeypatch.setattr(
        initializer,
        "initialize_sqlite_database",
        lambda _settings: calls.append("sqlite"),
    )
    monkeypatch.setattr(
        initializer,
        "ensure_postgres_database",
        lambda _settings: calls.append("postgres"),
    )

    initializer.run_database_initialization()

    assert calls == ["postgres"]
