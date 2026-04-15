from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect

from DILIGENT.server.domain.settings.configuration import DatabaseSettings
from DILIGENT.server.repositories.database.sqlite import SQLiteRepository


def _build_settings() -> DatabaseSettings:
    return DatabaseSettings(
        embedded_database=True,
        engine=None,
        host=None,
        port=None,
        database_name=None,
        username=None,
        password=None,
        ssl=False,
        ssl_ca=None,
        connect_timeout=10,
        insert_batch_size=1000,
        insert_commit_interval=5,
        select_page_size=2000,
    )


def test_sqlite_repository_initializes_schema_when_db_file_missing(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "DILIGENT.server.repositories.database.sqlite.RESOURCES_PATH",
        str(tmp_path),
    )
    monkeypatch.setattr(
        "DILIGENT.server.repositories.database.sqlite.DATABASE_FILENAME",
        "missing.db",
    )

    repository = SQLiteRepository(_build_settings())
    inspector = inspect(repository.engine)

    assert repository.db_path is not None
    assert Path(repository.db_path).exists()
    assert inspector.has_table("access_keys")
    assert inspector.has_table("model_selections")


def test_sqlite_repository_skips_schema_init_when_db_file_exists(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    db_path = tmp_path / "existing.db"
    db_path.touch()

    monkeypatch.setattr(
        "DILIGENT.server.repositories.database.sqlite.RESOURCES_PATH",
        str(tmp_path),
    )
    monkeypatch.setattr(
        "DILIGENT.server.repositories.database.sqlite.DATABASE_FILENAME",
        "existing.db",
    )

    repository = SQLiteRepository(_build_settings())
    inspector = inspect(repository.engine)

    assert inspector.has_table("access_keys") is False
    assert inspector.has_table("model_selections") is False
