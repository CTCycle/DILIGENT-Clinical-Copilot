# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from uuid import UUID

from domain.clinical.drafts import ClinicalDraftForm, ClinicalDraftSnapshot
from repositories.clinical_draft_repository import (
    ClinicalDraftRepository,
    DraftConcurrencyError,
)


class DraftConflictError(RuntimeError):
    """A clinical draft changed or belongs to a different database."""


class ClinicalDraftService:
    def __init__(self, repository: ClinicalDraftRepository) -> None:
        self.repository = repository

    def get(self) -> ClinicalDraftSnapshot:
        return self.repository.get_draft()

    def save(
        self,
        form: ClinicalDraftForm,
        *,
        expected_revision: int,
        expected_draft_key: UUID | None,
    ) -> ClinicalDraftSnapshot:
        try:
            return self.repository.save_draft(
                form,
                expected_revision=expected_revision,
                expected_draft_key=expected_draft_key,
            )
        except DraftConcurrencyError as exc:
            raise DraftConflictError from exc

    def discard(
        self,
        *,
        expected_revision: int,
        expected_draft_key: UUID | None,
    ) -> ClinicalDraftSnapshot:
        try:
            return self.repository.discard_draft(
                expected_revision=expected_revision,
                expected_draft_key=expected_draft_key,
            )
        except DraftConcurrencyError as exc:
            raise DraftConflictError from exc
