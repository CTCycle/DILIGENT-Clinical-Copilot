from __future__ import annotations

from pathlib import Path

from domain.settings.configuration import DatabaseSettings
from repositories.database.sqlite import SQLiteRepository
from repositories.schemas.configuration import ApplicationConfiguration
from sqlalchemy import select

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
