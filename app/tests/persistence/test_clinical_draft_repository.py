# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from threading import Barrier
from uuid import uuid4

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import func, select

from domain.clinical.drafts import ClinicalDraftForm
from repositories.clinical_draft_repository import (
    ClinicalDraftRepository,
    DraftConcurrencyError,
)
from repositories.context import RepositoryContext
from repositories.schemas.clinical import (
    ClinicalInputDraft,
    ClinicalSession,
    ClinicalSessionVersion,
)
from repositories.schemas import Base
from services.session.clinical_draft_service import (
    ClinicalDraftService,
    DraftConflictError,
)


def _repository(engine) -> ClinicalDraftRepository:  # type: ignore[no-untyped-def]
    return ClinicalDraftRepository(RepositoryContext.create(engine=engine))


def test_empty_read_is_side_effect_free(persistence_engine) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(persistence_engine)

    snapshot = repository.get_draft()

    assert snapshot.revision == 0
    assert snapshot.draft_key is None
    assert snapshot.form is None
    with RepositoryContext.create(engine=persistence_engine).session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ClinicalInputDraft)) == 0


def test_migrated_schema_matches_metadata_on_configured_backend(
    persistence_engine,
) -> None:  # type: ignore[no-untyped-def]
    with persistence_engine.connect() as connection:
        differences = compare_metadata(MigrationContext.configure(connection), Base.metadata)

    assert differences == []


def test_save_preserves_exact_text_across_repository_instances(persistence_engine) -> None:  # type: ignore[no-untyped-def]
    form = ClinicalDraftForm(
        patient_name="  Synthetic patient  ",
        visit_date=date(2026, 10, 23),
        clinical_input="  First line\r\n\r\n Second line\t ",
    )
    saved = _repository(persistence_engine).save_draft(form, 0, None)
    restored = _repository(persistence_engine).get_draft()

    assert saved.revision == 1
    assert saved.draft_key is not None
    assert restored.form == form
    assert restored.updated_at is not None


def test_revision_and_key_conflicts_do_not_change_saved_data(persistence_engine) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(persistence_engine)
    original = ClinicalDraftForm(patient_name="Original", clinical_input="first")
    saved = repository.save_draft(original, 0, None)

    updated = repository.save_draft(
        ClinicalDraftForm(patient_name="Updated", clinical_input="second"),
        saved.revision,
        saved.draft_key,
    )
    assert updated.revision == 2
    with pytest.raises(DraftConcurrencyError):
        repository.save_draft(original, saved.revision, saved.draft_key)
    with pytest.raises(DraftConcurrencyError):
        repository.save_draft(original, updated.revision, uuid4())

    current = repository.get_draft()
    assert current.revision == updated.revision
    assert current.form == updated.form


def test_concurrent_first_saves_allow_only_one_draft(persistence_engine) -> None:  # type: ignore[no-untyped-def]
    barrier = Barrier(2)

    def save(patient_name: str) -> str:
        repository = _repository(persistence_engine)
        barrier.wait(timeout=10)
        try:
            repository.save_draft(
                ClinicalDraftForm(patient_name=patient_name), 0, None
            )
            return "saved"
        except DraftConcurrencyError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(save, ["Synthetic A", "Synthetic B"]))

    assert sorted(outcomes) == ["conflict", "saved"]
    assert _repository(persistence_engine).get_draft().revision == 1
    with RepositoryContext.create(engine=persistence_engine).session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ClinicalInputDraft)) == 1


def test_discard_tombstone_clears_phi_and_rejects_delayed_save(persistence_engine) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(persistence_engine)
    saved = repository.save_draft(
        ClinicalDraftForm(
            patient_name="Synthetic patient",
            visit_date=date(2026, 10, 23),
            clinical_input="Synthetic clinical text\nsecond line",
        ),
        0,
        None,
    )

    tombstone = repository.discard_draft(saved.revision, saved.draft_key)
    assert tombstone.form is None
    assert tombstone.revision == saved.revision + 1
    assert tombstone.draft_key == saved.draft_key
    with RepositoryContext.create(engine=persistence_engine).session_factory() as session:
        row = session.get(ClinicalInputDraft, 1)
        assert row is not None
        assert row.patient_name is None
        assert row.visit_date is None
        assert row.clinical_input is None
        assert row.updated_at is None

    with pytest.raises(DraftConcurrencyError):
        repository.save_draft(
            ClinicalDraftForm(patient_name="Delayed"),
            saved.revision,
            saved.draft_key,
        )

    replacement = repository.save_draft(
        ClinicalDraftForm(patient_name="New explicit draft"),
        tombstone.revision,
        tombstone.draft_key,
    )
    assert replacement.revision == tombstone.revision + 1
    assert replacement.draft_key is not None
    assert replacement.draft_key != saved.draft_key


def test_different_database_rejects_previous_database_draft_token(
    persistence_engine, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    from sqlalchemy import create_engine

    first_repository = _repository(persistence_engine)
    first = first_repository.save_draft(
        ClinicalDraftForm(patient_name="Synthetic database A"), 0, None
    )

    second_engine = create_engine(
        f"sqlite+pysqlite:///{tmp_path / 'second-database.db'}",
        future=True,
        connect_args={"timeout": 30.0, "autocommit": False},
    )
    try:
        from repositories.database.migrations import migrate_database

        migrate_database(second_engine, database_was_empty=True)
        second_repository = _repository(second_engine)
        with pytest.raises(DraftConcurrencyError):
            second_repository.save_draft(
                ClinicalDraftForm(patient_name="Must not cross database"),
                first.revision,
                first.draft_key,
            )
        assert second_repository.get_draft().form is None
        assert first_repository.get_draft().form is not None
    finally:
        second_engine.dispose()


def test_draft_crud_does_not_change_session_or_version_counts(persistence_engine) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(persistence_engine)
    context = RepositoryContext.create(engine=persistence_engine)
    with context.session_factory() as session:
        before_sessions = session.scalar(select(func.count()).select_from(ClinicalSession))
        before_versions = session.scalar(
            select(func.count()).select_from(ClinicalSessionVersion)
        )

    saved = repository.save_draft(ClinicalDraftForm(clinical_input="Incomplete"), 0, None)
    repository.get_draft()
    repository.discard_draft(saved.revision, saved.draft_key)

    with context.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ClinicalSession)) == before_sessions
        assert session.scalar(
            select(func.count()).select_from(ClinicalSessionVersion)
        ) == before_versions


def test_service_maps_repository_cas_errors_to_domain_conflict(persistence_engine) -> None:  # type: ignore[no-untyped-def]
    service = ClinicalDraftService(_repository(persistence_engine))
    with pytest.raises(DraftConflictError):
        service.save(
            ClinicalDraftForm(),
            expected_revision=1,
            expected_draft_key=uuid4(),
        )
