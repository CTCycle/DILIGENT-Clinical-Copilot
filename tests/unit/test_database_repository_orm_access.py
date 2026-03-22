from __future__ import annotations

from pathlib import Path

import pandas as pd

from DILIGENT.server.configurations.server import DatabaseSettings
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


def test_sqlite_repository_table_reads_use_orm_queries(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "DILIGENT.server.repositories.database.sqlite.RESOURCES_PATH",
        str(tmp_path),
    )
    monkeypatch.setattr(
        "DILIGENT.server.repositories.database.sqlite.DATABASE_FILENAME",
        "orm_reads.db",
    )
    repository = SQLiteRepository(_build_settings())

    payload = pd.DataFrame(
        [
            {
                "role_type": "clinical",
                "provider": None,
                "model_name": "llama3.1:8b",
                "is_active": True,
            },
            {
                "role_type": "text_extraction",
                "provider": None,
                "model_name": "llama3.1:8b",
                "is_active": True,
            },
            {
                "role_type": "cloud",
                "provider": "openai",
                "model_name": "gpt-4.1-mini",
                "is_active": True,
            },
        ]
    )
    repository.upsert_into_database(payload, "model_selections")

    assert repository.count_rows("model_selections") == 3

    loaded = repository.load_from_database("model_selections")
    assert len(loaded.index) == 3
    assert set(loaded["role_type"].tolist()) == {"clinical", "text_extraction", "cloud"}

    paged = repository.load_paginated("model_selections", offset=1, limit=1)
    assert len(paged.index) == 1

    chunks = list(repository.stream_rows("model_selections", page_size=2))
    assert [len(chunk.index) for chunk in chunks] == [2, 1]
    streamed = pd.concat(chunks, ignore_index=True)
    assert streamed["role_type"].tolist() == loaded["role_type"].tolist()
