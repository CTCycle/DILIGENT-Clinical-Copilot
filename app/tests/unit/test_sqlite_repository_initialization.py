# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from domain.settings.configuration import (
    DEFAULT_RAG_SETTINGS,
    SHARED_RAG_SETTING_KEYS,
    DatabaseSettings,
)
from repositories.agent_repository import AgentRepository
from repositories.context import RepositoryContext
from repositories.database.initializer import initialize_sqlite_database
from repositories.database.sqlite import SQLiteRepository
from repositories.schemas.base import Base
from repositories.schemas.configuration import (
    AgentDefinition,
    AgentRevision,
    ApplicationConfiguration,
    ReferenceCatalogEntry,
)
from repositories.serialization.model_configs import LEGACY_EXECUTION_KEYS
from sqlalchemy import create_engine, delete, func, inspect, select


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
    assert not LEGACY_EXECUTION_KEYS & set(payload)
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
        defaults = {
            kind: dict(record["revision"]["configuration"])
            for kind in ("dili", "revision")
            if (record := agents.get_default(kind)) is not None
        }
        return payload, defaults, agents.count_agents()
    finally:
        repository.engine.dispose()


###############################################################################
def test_legacy_model_configuration_migrates_into_default_agents_once(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    database_path = tmp_path / "legacy-config.db"
    monkeypatch.setattr(
        "repositories.database.sqlite.DATABASE_FILE_PATH",
        database_path,
    )
    settings = _build_settings()
    initialize_sqlite_database(settings, seed_catalogs=False)

    # Recreate a pre-agent database: legacy execution keys, no agents.
    repository = SQLiteRepository(settings)
    with repository.session_factory() as db_session:
        db_session.execute(delete(AgentRevision))
        db_session.execute(delete(AgentDefinition))
        configuration = db_session.get(ApplicationConfiguration, 1)
        assert configuration is not None
        configuration.payload = {
            **dict(configuration.payload),
            "clinical_model": "gpt-clinical",
            "text_extraction_model": "gpt-parser",
            "revision_model": "gpt-revision",
            "timeline_model": "gpt-timeline",
            "use_cloud_models": True,
            "cloud_provider": "openai",
            "cloud_model": "gpt-clinical",
            "reasoning_level": "high",
            "ollama_seed": 7,
            "rag_settings": {
                **DEFAULT_RAG_SETTINGS,
                "chunk_size": 777,
                "use_reranking": False,
                "reranker_model": "custom-reranker",
                "retrieval_candidate_count": 30,
                "retrieval_selected_count": 5,
            },
        }
        db_session.add(configuration)
        db_session.commit()
    repository.engine.dispose()

    initialize_sqlite_database(settings, seed_catalogs=False)
    payload, agents, agent_count = _read_configuration_and_agents(settings)

    dili, revision = agents["dili"], agents["revision"]
    assert dili["models"]["clinical"] == {
        "provider": "openai",
        "model": "gpt-clinical",
        "reasoning_level": None,
    }
    assert dili["models"]["extraction"]["model"] == "gpt-parser"
    assert dili["models"]["timeline"]["model"] == "gpt-timeline"
    assert dili["reasoning_level"] == "high"
    assert dili["ollama_seed"] == 7
    assert dili["retrieval"]["use_reranking"] is False
    assert dili["retrieval"]["reranker_model"] == "custom-reranker"
    assert dili["retrieval"]["retrieval_candidate_count"] == 30
    assert dili["retrieval"]["retrieval_selected_count"] == 5
    assert revision["models"]["planning"]["provider"] == "openai"
    assert revision["models"]["planning"]["model"] == "gpt-revision"

    assert not LEGACY_EXECUTION_KEYS & set(payload)
    assert set(payload["rag_settings"]) == set(SHARED_RAG_SETTING_KEYS)
    assert payload["rag_settings"]["chunk_size"] == 777
    assert payload["runtime"]["cloud_llm_timeout_cap"] == 1800.0

    initialize_sqlite_database(settings, seed_catalogs=False)
    second_payload, second_agents, second_count = _read_configuration_and_agents(
        settings
    )
    assert second_payload == payload
    assert second_agents == agents
    assert second_count == agent_count == 2
