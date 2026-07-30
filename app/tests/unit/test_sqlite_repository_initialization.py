from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from domain.settings.configuration import DatabaseSettings
from repositories.database.sqlite import SQLiteRepository
from repositories.schemas.base import Base
from repositories.schemas.configuration import (
    ApplicationConfiguration,
    ReferenceCatalogEntry,
)
from sqlalchemy import create_engine, func, inspect, select

###############################################################################
def _build_settings() -> DatabaseSettings:
    return DatabaseSettings(
        backend="sqlite",
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

###############################################################################
def _make_temp_db_root(prefix: str) -> Path:
    temp_root = Path(tempfile.gettempdir()) / f"{prefix}-{uuid.uuid4().hex}"
    temp_root.mkdir(parents=True, exist_ok=True)
    return temp_root

###############################################################################
def test_sqlite_repository_initializes_schema_when_db_file_missing(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    temp_root = _make_temp_db_root("sqlite-init-missing")
    try:
        monkeypatch.setattr(
            "repositories.database.sqlite.DATABASE_FILE_PATH",
            temp_root / "missing.db",
        )

        repository = SQLiteRepository(_build_settings())
        inspector = inspect(repository.engine)

        assert repository.db_path is not None
        assert Path(repository.db_path).exists()
        assert inspector.has_table("access_keys")
        assert inspector.has_table("application_configuration")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

###############################################################################
def test_sqlite_repository_does_not_seed_catalogs_during_construction(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    temp_root = _make_temp_db_root("sqlite-init-existing")
    db_path = temp_root / "existing.db"
    engine = create_engine(f"sqlite+pysqlite:///{db_path}", future=True)
    try:
        Base.metadata.create_all(engine)
        engine.dispose()

        monkeypatch.setattr(
            "repositories.database.sqlite.DATABASE_FILE_PATH",
            db_path,
        )

        repository = SQLiteRepository(_build_settings())
        inspector = inspect(repository.engine)

        assert inspector.has_table("access_keys")
        assert inspector.has_table("application_configuration")
        assert inspector.has_table("reference_catalog_entries")
        with repository.session_factory() as db_session:
            catalog_entries = db_session.execute(
                select(func.count()).select_from(ReferenceCatalogEntry)
            ).scalar_one()
        assert int(catalog_entries) == 0
    finally:
        engine.dispose()
        shutil.rmtree(temp_root, ignore_errors=True)

###############################################################################
def test_sqlite_repository_exposes_orm_session_factory(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "repositories.database.sqlite.DATABASE_FILE_PATH",
        tmp_path / "orm_reads.db",
    )
    repository = SQLiteRepository(_build_settings())

    with repository.session_factory() as db_session:
        db_session.add(
            ApplicationConfiguration(
                payload={
                    "clinical_model": "llama3.1:8b",
                    "text_extraction_model": "llama3.1:8b",
                    "use_cloud_models": True,
                    "cloud_provider": "openai",
                    "cloud_model": "gpt-4.1-mini",
                }
            )
        )
        db_session.commit()

    with repository.session_factory() as db_session:
        loaded = (
            db_session.execute(
                select(ApplicationConfiguration)
            )
            .scalars()
            .all()
        )

    assert len(loaded) == 1
    assert loaded[0].payload["clinical_model"] == "llama3.1:8b"
