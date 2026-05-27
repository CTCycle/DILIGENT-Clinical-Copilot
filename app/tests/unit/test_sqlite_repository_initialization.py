from __future__ import annotations

from pathlib import Path

from domain.settings.configuration import DatabaseSettings
from repositories.database.sqlite import SQLiteRepository
from repositories.schemas.models import Base, ReferenceCatalogEntry
from sqlalchemy import create_engine, func, inspect, select


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
        "repositories.database.sqlite.RESOURCES_PATH",
        str(tmp_path),
    )
    monkeypatch.setattr(
        "repositories.database.sqlite.DATABASE_FILENAME",
        "missing.db",
    )

    repository = SQLiteRepository(_build_settings())
    inspector = inspect(repository.engine)

    assert repository.db_path is not None
    assert Path(repository.db_path).exists()
    assert inspector.has_table("access_keys")
    assert inspector.has_table("model_selections")


def test_sqlite_repository_does_not_seed_catalogs_during_construction(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    db_path = tmp_path / "existing.db"
    engine = create_engine(f"sqlite+pysqlite:///{db_path}", future=True)
    Base.metadata.create_all(engine)
    engine.dispose()

    monkeypatch.setattr(
        "repositories.database.sqlite.RESOURCES_PATH",
        str(tmp_path),
    )
    monkeypatch.setattr(
        "repositories.database.sqlite.DATABASE_FILENAME",
        "existing.db",
    )

    repository = SQLiteRepository(_build_settings())
    inspector = inspect(repository.engine)

    assert inspector.has_table("access_keys")
    assert inspector.has_table("model_selections")
    assert inspector.has_table("reference_catalog_entries")
    with repository.session_factory() as db_session:
        catalog_entries = db_session.execute(
            select(func.count()).select_from(ReferenceCatalogEntry)
        ).scalar_one()
    assert int(catalog_entries) == 0


