# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from domain.settings.configuration import (
    SHARED_RAG_SETTING_KEYS,
    DatabaseSettings,
)
from repositories.agent_repository import AgentRepository
from repositories.context import RepositoryContext
from repositories.database.initializer import initialize_sqlite_database
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
    )


###############################################################################
def _make_temp_db_root(prefix: str) -> Path:
    temp_root = Path(tempfile.gettempdir()) / f"{prefix}-{uuid.uuid4().hex}"
    temp_root.mkdir(parents=True, exist_ok=True)
    return temp_root


###############################################################################
def test_sqlite_initializer_creates_schema_when_db_file_missing(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    temp_root = _make_temp_db_root("sqlite-init-missing")
    try:
        monkeypatch.setattr(
            "repositories.database.sqlite.DATABASE_FILE_PATH",
            temp_root / "missing.db",
        )

        settings = _build_settings()
        initialize_sqlite_database(settings, seed_catalogs=False)
        repository = SQLiteRepository(settings)
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
    settings = _build_settings()
    initialize_sqlite_database(settings, seed_catalogs=False)
    repository = SQLiteRepository(settings)

    with repository.session_factory() as db_session:
        configuration = db_session.get(ApplicationConfiguration, 1)
        assert configuration is not None
        configuration.payload = {
            "clinical_model": "llama3.1:8b",
            "text_extraction_model": "llama3.1:8b",
            "revision_model": "llama3.1:8b",
            "timeline_model": "llama3.1:8b",
            "use_cloud_models": True,
            "cloud_provider": "openai",
            "cloud_model": "gpt-4.1-mini",
        }
        db_session.add(configuration)
        db_session.commit()

    with repository.session_factory() as db_session:
        loaded = db_session.execute(select(ApplicationConfiguration)).scalars().all()

    assert len(loaded) == 1
    assert loaded[0].payload["clinical_model"] == "llama3.1:8b"


###############################################################################
def test_sqlite_initializer_seeds_complete_application_configuration(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    database_path = tmp_path / "complete-config.db"
    monkeypatch.setattr(
        "repositories.database.sqlite.DATABASE_FILE_PATH",
        database_path,
    )

    initialize_sqlite_database(_build_settings(), seed_catalogs=False)
    repository = SQLiteRepository(_build_settings())
    try:
        with repository.session_factory() as db_session:
            configuration = db_session.get(ApplicationConfiguration, 1)
            assert configuration is not None
            payload = dict(configuration.payload)
    finally:
        repository.engine.dispose()

    assert {
        "jobs",
        "rag_settings",
        "runtime",
        "ingestion",
        "session_pipeline",
        "clinical_language_detection",
        "drugs_matcher",
    }.issubset(payload)
    assert set(payload["rag_settings"]) == set(SHARED_RAG_SETTING_KEYS)
    assert "rag" not in payload


###############################################################################
def _read_configuration_and_agents(
    settings: DatabaseSettings,
) -> tuple[dict[str, object], dict[str, dict[str, object]], int]:
    repository = SQLiteRepository(settings)
    try:
        with repository.session_factory() as db_session:
            configuration = db_session.get(ApplicationConfiguration, 1)
            assert configuration is not None
            payload = dict(configuration.payload)
        agents = AgentRepository(
            RepositoryContext.create(
                engine=repository.engine, session_factory=repository.session_factory
            )
        )
        record = agents.get_default()
        default = dict(record["revision"]["configuration"]) if record else {}
        return payload, default, agents.count_agents()
    finally:
        repository.engine.dispose()


###############################################################################
def test_fresh_database_seeds_one_default_agent_once(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "repositories.database.sqlite.DATABASE_FILE_PATH",
        tmp_path / "fresh-agent.db",
    )
    settings = _build_settings()
    initialize_sqlite_database(settings, seed_catalogs=False)
    payload, agent, agent_count = _read_configuration_and_agents(settings)

    assert agent_count == 1
    assert agent["assessment"]["models"]["clinical"]["provider"] == "ollama"
    assert agent["revision"]["models"]["planning"] == agent["assessment"]["models"]["clinical"]
    assert agent["timeline"]["model"] == agent["assessment"]["models"]["extraction"]

    initialize_sqlite_database(settings, seed_catalogs=False)
    second_payload, second_agent, second_count = _read_configuration_and_agents(
        settings
    )
    assert second_payload == payload
    assert second_agent == agent
    assert second_count == 1
