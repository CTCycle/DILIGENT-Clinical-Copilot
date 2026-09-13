from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from alembic import command
from repositories.database.migrations import (
    build_alembic_config,
    migrate_database,
)
from sqlalchemy import Engine, create_engine, inspect, text

PREVIOUS_RELEASE_HEAD = "202608200003"
CURRENT_RELEASE_HEAD = "202609100001"

###############################################################################
def _engine(path: Path) -> Engine:
    return create_engine(
        f"sqlite+pysqlite:///{path}",
        future=True,
        connect_args={"timeout": 30.0, "autocommit": False},
    )

###############################################################################
def _upgrade_to(engine: Engine, revision: str) -> None:
    config = build_alembic_config()
    with engine.connect() as connection, connection.begin():
        config.attributes["connection"] = connection
        command.upgrade(config, revision)

###############################################################################
def _schema_signature(engine: Engine) -> tuple[tuple[object, ...], ...]:
    inspector = inspect(engine)
    tables: list[tuple[object, ...]] = []
    for table_name in sorted(inspector.get_table_names()):
        columns = tuple(
            (
                column["name"],
                str(column["type"]),
                column["nullable"],
                column["default"],
            )
            for column in inspector.get_columns(table_name)
        )
        indexes = tuple(
            sorted(
                (
                    index.get("name"),
                    tuple(index.get("column_names") or ()),
                    index.get("unique"),
                )
                for index in inspector.get_indexes(table_name)
            )
        )
        foreign_keys = tuple(
            sorted(
                (
                    foreign_key.get("name"),
                    tuple(foreign_key.get("constrained_columns") or ()),
                    foreign_key.get("referred_table"),
                    tuple(foreign_key.get("referred_columns") or ()),
                )
                for foreign_key in inspector.get_foreign_keys(table_name)
            )
        )
        tables.append((table_name, columns, indexes, foreign_keys))
    return tuple(tables)

###############################################################################
def _seed_previous_release_data(engine: Engine) -> None:
    timestamp = datetime(2026, 9, 1, 10, 30, 0, tzinfo=UTC)
    with engine.begin() as connection:
        connection.execute(
            text(
                "insert into application_configuration "
                "(id, revision, payload) values (1, 7, :payload)"
            ),
            {
                "payload": json.dumps(
                    {
                        "clinical_model": "legacy-clinical",
                        "text_extraction_model": "legacy-extraction",
                        "revision_model": "legacy-revision",
                        "timeline_model": "legacy-timeline",
                    }
                )
            },
        )
        connection.execute(
            text(
                "insert into clinical_sessions "
                "(id, patient_name, visit_date, anamnesis, session_status) "
                "values (1, 'Legacy Patient', '2026-09-01', :anamnesis, 'completed')"
            ),
            {"anamnesis": "Persisted v3.3.0 anamnesis"},
        )
        connection.execute(
            text(
                "insert into clinical_session_results "
                "(id, session_id, payload_json) "
                "values (1, 1, :payload)"
            ),
            {"payload": '{"report":"legacy report"}'},
        )
        connection.execute(
            text(
                "insert into clinical_session_versions "
                "(id, session_id, root_session_id, version_number, version_status, "
                "revision_kind, llm_qa_status, clinical_review_status, report_text) "
                "values (1, 1, 1, 1, 'current', 'original', 'not_run', "
                "'not_reviewed', 'Legacy report text')"
            )
        )
        connection.execute(
            text(
                "insert into clinical_session_revision_runs "
                "(id, pipeline_run_id, session_id, root_session_id, source_version_id, "
                "revision_mode, revision_kind, actor_source, actor_confidence, "
                "started_at, status) values "
                "(1, 'legacy-pipeline-1', 1, 1, 1, 'review', 'original', "
                "'system', 'system', :started_at, 'completed')"
            ),
            {"started_at": timestamp},
        )
        connection.execute(
            text(
                "insert into clinical_session_timelines "
                "(id, session_id, generated_at, generation_status, source_kind, "
                "timeline_payload_json) values "
                "(1, 1, :generated_at, 'fallback', 'local', :payload)"
            ),
            {
                "generated_at": timestamp,
                "payload": '{"events":[{"label":"Legacy event"}]}',
            },
        )
        connection.execute(
            text(
                "insert into drugs "
                "(id, canonical_name, canonical_name_norm, livertox_nbk_id) "
                "values (1, 'Legacy Drug', 'legacy drug', 'NBK-legacy')"
            )
        )
        connection.execute(
            text(
                "insert into drug_aliases "
                "(id, drug_id, alias, alias_norm, alias_kind, source) "
                "values (1, 1, 'Legacy Brand', 'legacy brand', 'brand', 'livertox')"
            )
        )
        connection.execute(
            text(
                "insert into livertox_monographs "
                "(id, drug_id, monograph_key, drug_name_norm, nbk_id, excerpt) "
                "values (1, 1, 'NBK-legacy', 'legacy drug', 'NBK-legacy', "
                "'Legacy LiverTox excerpt')"
            )
        )

###############################################################################
def test_v330_database_upgrades_to_current_head_without_losing_data(
    tmp_path: Path,
) -> None:
    previous_database = _engine(tmp_path / "v330.db")
    current_database = _engine(tmp_path / "current.db")
    try:
        _upgrade_to(previous_database, PREVIOUS_RELEASE_HEAD)
        _seed_previous_release_data(previous_database)

        result = migrate_database(previous_database, database_was_empty=False)

        assert result.target_heads == (CURRENT_RELEASE_HEAD,)
        with previous_database.connect() as connection:
            assert (
                connection.execute(text("select version_num from alembic_version"))
                .scalar_one()
                == CURRENT_RELEASE_HEAD
            )
            assert connection.execute(
                text("select patient_name from clinical_sessions where id = 1")
            ).scalar_one() == "Legacy Patient"
            assert connection.execute(
                text("select report_text from clinical_session_versions where id = 1")
            ).scalar_one() == "Legacy report text"
            assert connection.execute(
                text("select pipeline_run_id from clinical_session_revision_runs where id = 1")
            ).scalar_one() == "legacy-pipeline-1"
            assert connection.execute(
                text("select timeline_payload_json from clinical_session_timelines where id = 1")
            ).scalar_one() == '{"events":[{"label":"Legacy event"}]}'
            assert connection.execute(
                text("select canonical_name from drugs where id = 1")
            ).scalar_one() == "Legacy Drug"
            assert connection.execute(
                text("select excerpt from livertox_monographs where id = 1")
            ).scalar_one() == "Legacy LiverTox excerpt"

        upgraded_inspector = inspect(previous_database)
        assert upgraded_inspector.has_table("dilirank_records")
        assert {
            index["name"] for index in upgraded_inspector.get_indexes("dilirank_records")
        } == {
            "ix_dilirank_records_drug_id",
            "ix_dilirank_records_compound_name_norm",
            "ix_dilirank_records_dili_concern",
        }
        assert upgraded_inspector.get_foreign_keys("dilirank_records")[0][
            "referred_table"
        ] == "drugs"

        migrate_database(current_database, database_was_empty=True)
        assert _schema_signature(previous_database) == _schema_signature(current_database)
    finally:
        previous_database.dispose()
        current_database.dispose()
