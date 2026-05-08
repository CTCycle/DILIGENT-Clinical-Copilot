from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from domain.settings.configuration import DatabaseSettings
from repositories.database.sqlite import SQLiteRepository
from repositories.schemas.models import ModelSelection


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


def test_sqlite_repository_exposes_orm_session_factory(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "repositories.database.sqlite.RESOURCES_PATH",
        str(tmp_path),
    )
    monkeypatch.setattr(
        "repositories.database.sqlite.DATABASE_FILENAME",
        "orm_reads.db",
    )
    repository = SQLiteRepository(_build_settings())

    with repository.session_factory() as db_session:
        db_session.add_all(
            [
                ModelSelection(
                    role_type="clinical",
                    provider=None,
                    model_name="llama3.1:8b",
                    is_active=True,
                ),
                ModelSelection(
                    role_type="text_extraction",
                    provider=None,
                    model_name="llama3.1:8b",
                    is_active=True,
                ),
                ModelSelection(
                    role_type="cloud",
                    provider="openai",
                    model_name="gpt-4.1-mini",
                    is_active=True,
                ),
            ]
        )
        db_session.commit()

    with repository.session_factory() as db_session:
        loaded = (
            db_session.execute(
                select(ModelSelection).order_by(ModelSelection.role_type.asc())
            )
            .scalars()
            .all()
        )

    assert len(loaded) == 3
    assert [row.role_type for row in loaded] == ["clinical", "cloud", "text_extraction"]


