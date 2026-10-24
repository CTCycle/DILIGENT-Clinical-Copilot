# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError, OperationalError

from domain.clinical.drafts import ClinicalDraftForm, ClinicalDraftSnapshot
from repositories.context import RepositoryContext
from repositories.schemas.clinical import ClinicalInputDraft


class DraftConcurrencyError(RuntimeError):
    """A draft revision or database identity changed since it was read."""


class ClinicalDraftRepository:
    def __init__(self, context: RepositoryContext) -> None:
        self.context = context
        self.session_factory = context.session_factory

    def get_draft(self) -> ClinicalDraftSnapshot:
        with self.session_factory() as db_session:
            row = db_session.get(ClinicalInputDraft, 1)
            return self._snapshot(row)

    def save_draft(
        self,
        form: ClinicalDraftForm,
        expected_revision: int,
        expected_draft_key: UUID | None,
    ) -> ClinicalDraftSnapshot:
        with self.session_factory() as db_session:
            row = db_session.get(ClinicalInputDraft, 1)
            new_key = str(uuid4()) if row is None or not row.is_active else row.draft_key
            if row is None:
                if expected_revision != 0 or expected_draft_key is not None:
                    raise DraftConcurrencyError
                db_session.add(
                    ClinicalInputDraft(
                        id=1,
                        draft_key=new_key,
                        revision=1,
                        is_active=True,
                        patient_name=form.patient_name,
                        visit_date=form.visit_date,
                        clinical_input=form.clinical_input,
                        use_rag=form.use_rag,
                        updated_at=datetime.now(UTC),
                    )
                )
                try:
                    db_session.flush()
                except (IntegrityError, OperationalError) as exc:
                    db_session.rollback()
                    if self._draft_exists(db_session):
                        raise DraftConcurrencyError from exc
                    raise
            else:
                result = db_session.execute(
                    update(ClinicalInputDraft)
                    .where(self._expected_row(expected_revision, expected_draft_key))
                    .values(
                        draft_key=new_key,
                        revision=expected_revision + 1,
                        is_active=True,
                        patient_name=form.patient_name,
                        visit_date=form.visit_date,
                        clinical_input=form.clinical_input,
                        use_rag=form.use_rag,
                        updated_at=datetime.now(UTC),
                    )
                )
                if result.rowcount != 1:
                    db_session.rollback()
                    raise DraftConcurrencyError
            db_session.commit()
            return self._snapshot(db_session.get(ClinicalInputDraft, 1))

    def discard_draft(
        self,
        expected_revision: int,
        expected_draft_key: UUID | None,
    ) -> ClinicalDraftSnapshot:
        with self.session_factory() as db_session:
            row = db_session.get(ClinicalInputDraft, 1)
            if row is None:
                if expected_revision != 0 or expected_draft_key is not None:
                    raise DraftConcurrencyError
                db_session.add(
                    ClinicalInputDraft(
                        id=1,
                        draft_key=None,
                        revision=1,
                        is_active=False,
                        patient_name=None,
                        visit_date=None,
                        clinical_input=None,
                        use_rag=False,
                        updated_at=None,
                    )
                )
                try:
                    db_session.flush()
                except (IntegrityError, OperationalError) as exc:
                    db_session.rollback()
                    if self._draft_exists(db_session):
                        raise DraftConcurrencyError from exc
                    raise
            else:
                result = db_session.execute(
                    update(ClinicalInputDraft)
                    .where(self._expected_row(expected_revision, expected_draft_key))
                    .values(
                        revision=expected_revision + 1,
                        is_active=False,
                        patient_name=None,
                        visit_date=None,
                        clinical_input=None,
                        use_rag=False,
                        updated_at=None,
                    )
                )
                if result.rowcount != 1:
                    db_session.rollback()
                    raise DraftConcurrencyError
            db_session.commit()
            return self._snapshot(db_session.get(ClinicalInputDraft, 1))

    @staticmethod
    def _expected_row(
        expected_revision: int,
        expected_draft_key: UUID | None,
    ):
        key_predicate = (
            ClinicalInputDraft.draft_key.is_(None)
            if expected_draft_key is None
            else ClinicalInputDraft.draft_key == str(expected_draft_key)
        )
        return (
            (ClinicalInputDraft.id == 1)
            & (ClinicalInputDraft.revision == expected_revision)
            & key_predicate
        )

    @staticmethod
    def _snapshot(row: ClinicalInputDraft | None) -> ClinicalDraftSnapshot:
        if row is None:
            return ClinicalDraftSnapshot(
                draft_key=None,
                revision=0,
                updated_at=None,
                form=None,
            )
        draft_key = UUID(row.draft_key) if row.draft_key else None
        form = (
            ClinicalDraftForm(
                patient_name=row.patient_name or "",
                visit_date=row.visit_date,
                clinical_input=row.clinical_input or "",
                use_rag=row.use_rag,
            )
            if row.is_active
            else None
        )
        return ClinicalDraftSnapshot(
            draft_key=draft_key,
            revision=row.revision,
            updated_at=row.updated_at,
            form=form,
        )

    @staticmethod
    def _draft_exists(db_session) -> bool:
        try:
            db_session.rollback()
            return db_session.get(ClinicalInputDraft, 1) is not None
        except OperationalError:
            return False
