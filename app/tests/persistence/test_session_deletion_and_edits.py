# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from threading import Event, Thread
from typing import Any

import pytest
from common.exceptions import ServiceConflictError
from repositories.clinical_session_repository import ClinicalSessionRepository
from repositories.database.session import unit_of_work
from repositories.schemas.clinical import ClinicalSession, ClinicalSessionVersion
from repository_fixtures import RepositoryGraph, build_repository_graph
from services.inspection import DataInspectionService
from services.runtime.jobs import JobManager
from sqlalchemy import func, select, text
from sqlalchemy.exc import OperationalError


###############################################################################
def build_service(graph: RepositoryGraph) -> DataInspectionService:
    return DataInspectionService(
        clinical_session_repository=graph.clinical_session_repository,
        drug_catalog_repository=graph.drug_catalog_repository,
        knowledge_repository=graph.knowledge_repository,
        session_timeline_repository=graph.session_timeline_repository,
        session_revision_repository=graph.session_revision_repository,
        jobs=JobManager(),
    )


###############################################################################
def save_completed(graph: RepositoryGraph, name: str, report: str = "Report") -> int:
    session_id = graph.clinical_session_repository.save_clinical_session(
        {
            "patient_name": name,
            "session_status": "completed",
            "final_report": report,
            "session_result_payload": {"report": report},
        }
    )
    assert session_id is not None
    return int(session_id)


###############################################################################
def accept_child(graph: RepositoryGraph, root_id: int) -> int:
    """Persist an accepted revision: a child session holding a lineage version."""
    child_id = save_completed(graph, "Accepted child", "Revised report")
    shell = graph.session_revision_repository.create_revision_version_shell(
        root_id, reviewer_note=None, configuration={}
    )
    assert shell is not None
    finalized = graph.session_revision_repository.finalize_revision_version(
        pipeline_run_id=str(shell["pipeline_run_id"]),
        persisted_session_id=child_id,
    )
    assert finalized is not None
    return child_id


###############################################################################
def snapshot(graph: RepositoryGraph) -> tuple[int, int]:
    with graph.context.session_factory() as db_session:
        sessions = db_session.scalar(select(func.count(ClinicalSession.id)))
        versions = db_session.scalar(select(func.count(ClinicalSessionVersion.id)))
    return int(sessions or 0), int(versions or 0)


###############################################################################
def test_delete_standalone_session_removes_session_and_versions(
    persistence_engine,
) -> None:  # type: ignore[no-untyped-def]
    graph = build_repository_graph(engine=persistence_engine)
    session_id = save_completed(graph, "Standalone")

    assert build_service(graph).delete_session(session_id) is True
    assert snapshot(graph) == (0, 0)


###############################################################################
def test_delete_root_with_accepted_child_is_rejected_and_lineage_preserved(
    persistence_engine,
) -> None:  # type: ignore[no-untyped-def]
    graph = build_repository_graph(engine=persistence_engine)
    root_id = save_completed(graph, "Root")
    accept_child(graph, root_id)
    before = snapshot(graph)

    with pytest.raises(ServiceConflictError, match="child sessions"):
        build_service(graph).delete_session(root_id)

    assert snapshot(graph) == before


###############################################################################
def test_delete_child_then_root_is_permitted(persistence_engine) -> None:  # type: ignore[no-untyped-def]
    graph = build_repository_graph(engine=persistence_engine)
    root_id = save_completed(graph, "Root")
    child_id = accept_child(graph, root_id)
    service = build_service(graph)

    assert service.delete_session(child_id) is True
    with graph.context.session_factory() as db_session:
        remaining = db_session.scalars(
            select(ClinicalSessionVersion.session_id).where(
                ClinicalSessionVersion.root_session_id == root_id
            )
        ).all()
    assert remaining == [root_id]

    assert service.delete_session(root_id) is True
    assert snapshot(graph) == (0, 0)


###############################################################################
def test_failure_in_second_delete_step_rolls_back_revision_deletion(
    persistence_engine, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    graph = build_repository_graph(engine=persistence_engine)
    session_id = save_completed(graph, "Rollback")
    before = snapshot(graph)

    def fail(*_args: Any, **_kwargs: Any) -> bool:
        raise RuntimeError("injected failure")

    monkeypatch.setattr(ClinicalSessionRepository, "delete_session", fail)

    with pytest.raises(RuntimeError, match="injected failure"):
        build_service(graph).delete_session(session_id)

    assert snapshot(graph) == before


###############################################################################
def test_manual_edit_rejects_stale_version_and_accepts_current(
    persistence_engine,
) -> None:  # type: ignore[no-untyped-def]
    graph = build_repository_graph(engine=persistence_engine)
    session_id = save_completed(graph, "Editing", "Original report")
    service = build_service(graph)
    opened = service.get_session_detail(session_id)
    assert opened is not None

    def edit(text: str, expected_version: int) -> dict[str, Any] | None:
        return service.manual_edit_report(
            session_id,
            expected_version=expected_version,
            report_text=text,
            edited_fields=None,
            reviewer_note=None,
            edited_by="Reviewer",
            metadata=None,
        )

    first = edit("Newer report", int(opened["version"]))
    assert first is not None
    assert first["session"]["version"] == int(opened["version"]) + 1

    with pytest.raises(ServiceConflictError, match="changed since you opened"):
        edit("Stale overwrite", int(opened["version"]))

    detail = service.get_session_detail(session_id)
    assert detail is not None
    assert detail["report"] == "Newer report"
    assert len(detail["manual_edit_history"]) == 1


###############################################################################
def test_read_then_write_transaction_survives_concurrent_writer(
    persistence_engine,
) -> None:  # type: ignore[no-untyped-def]
    graph = build_repository_graph(engine=persistence_engine)
    first_id = save_completed(graph, "Reader")
    second_id = save_completed(graph, "Writer")
    factory = graph.context.session_factory
    read_done = Event()
    writer_attempted = Event()
    writer_blocked = Event()
    errors: list[BaseException] = []

    def read_then_write() -> None:
        try:
            with unit_of_work(session_factory=factory, immediate=True) as db_session:
                row = db_session.get(ClinicalSession, first_id)
                assert row is not None
                read_done.set()
                assert writer_attempted.wait(timeout=10)
                row.patient_name = "Reader updated"
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def concurrent_writer() -> None:
        # Fail fast on a held lock instead of waiting, so the test never sleeps.
        try:
            assert read_done.wait(timeout=10)
            with factory() as db_session:
                if db_session.get_bind().dialect.name == "sqlite":
                    db_session.execute(text("PRAGMA busy_timeout=0"))
                row = db_session.get(ClinicalSession, second_id)
                assert row is not None
                row.patient_name = "Writer updated"
                try:
                    db_session.commit()
                except OperationalError:
                    writer_blocked.set()
                    db_session.rollback()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            writer_attempted.set()

    threads = [Thread(target=read_then_write), Thread(target=concurrent_writer)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    # Under SQLite the write lock is taken up front, so the other writer is
    # turned away until the transaction ends; otherwise it commits in between.
    if factory.kw["bind"].dialect.name == "sqlite":
        assert writer_blocked.is_set()
    assert errors == []
    with factory() as db_session:
        reader_name = db_session.scalar(
            select(ClinicalSession.patient_name).where(ClinicalSession.id == first_id)
        )
    assert reader_name == "Reader updated"


###############################################################################
def test_session_report_presence_requires_non_empty_text(persistence_engine) -> None:  # type: ignore[no-untyped-def]
    graph = build_repository_graph(engine=persistence_engine)
    repository = graph.clinical_session_repository
    empty_id = save_completed(graph, "Empty report", "")
    filled_id = save_completed(graph, "Filled report", "Report")
    with graph.context.session_factory() as db_session:
        db_session.query(ClinicalSessionVersion).filter_by(
            session_id=empty_id
        ).update({"report_text": ""})
        db_session.query(ClinicalSessionVersion).filter_by(
            session_id=filled_id
        ).update({"report_text": "Report"})
        db_session.commit()

    state_flags = {
        sid: repository.get_session_state(sid)["has_report"]  # type: ignore[index]
        for sid in (empty_id, filled_id)
    }
    items, _ = repository.list_sessions(search=None, offset=0, limit=10)
    list_flags = {item["session_id"]: item["has_report"] for item in items}

    assert state_flags == {empty_id: False, filled_id: True}
    assert list_flags == state_flags
