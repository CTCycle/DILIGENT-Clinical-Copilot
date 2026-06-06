from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from domain.settings.configuration import DatabaseSettings
from repositories.database.sqlite import SQLiteRepository
from repositories.schemas.models import (
    Base,
    ClinicalSession,
    ClinicalSessionResult,
    Patient,
    ReferenceCatalogEntry,
)
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


def _make_temp_db_root(prefix: str) -> Path:
    temp_root = Path(tempfile.gettempdir()) / f"{prefix}-{uuid.uuid4().hex}"
    temp_root.mkdir(parents=True, exist_ok=True)
    return temp_root


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
        assert inspector.has_table("model_selections")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


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
        assert inspector.has_table("model_selections")
        assert inspector.has_table("reference_catalog_entries")
        with repository.session_factory() as db_session:
            catalog_entries = db_session.execute(
                select(func.count()).select_from(ReferenceCatalogEntry)
            ).scalar_one()
        assert int(catalog_entries) == 0
    finally:
        engine.dispose()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_sqlite_repository_additively_upgrades_existing_legacy_schema(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    temp_root = _make_temp_db_root("sqlite-upgrade")
    db_path = temp_root / "legacy.db"
    engine = create_engine(f"sqlite+pysqlite:///{db_path}", future=True)
    try:
        Base.metadata.create_all(
            engine,
            tables=[
                Patient.__table__,
                ClinicalSession.__table__,
                ClinicalSessionResult.__table__,
            ],
        )
        with engine.begin() as connection:
            connection.execute(
                Patient.__table__.insert().values(
                    id=1,
                    name="Legacy Patient",
                )
            )
            connection.execute(
                ClinicalSession.__table__.insert().values(
                    id=1,
                    patient_id=1,
                    version=1,
                    session_status="successful",
                )
            )
            connection.execute(
                ClinicalSessionResult.__table__.insert().values(
                    session_id=1,
                    payload_json='{"report":"Legacy report"}',
                )
            )
        engine.dispose()

        monkeypatch.setattr(
            "repositories.database.sqlite.DATABASE_FILE_PATH",
            db_path,
        )

        repository = SQLiteRepository(_build_settings())
        inspector = inspect(repository.engine)

        assert inspector.has_table("clinical_session_versions")
        assert inspector.has_table("clinical_session_revision_runs")
        assert inspector.has_table("clinical_session_revision_steps")
        assert inspector.has_table("clinical_session_revision_artifacts")
        assert inspector.has_table("clinical_session_revision_entities")
        assert inspector.has_table("clinical_session_revision_reviews")
        assert inspector.has_table("clinical_session_manual_edits")

        with repository.session_factory() as db_session:
            session_count = db_session.execute(
                select(func.count()).select_from(ClinicalSession)
            ).scalar_one()
            result_count = db_session.execute(
                select(func.count()).select_from(ClinicalSessionResult)
            ).scalar_one()

        assert int(session_count) == 1
        assert int(result_count) == 1
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
