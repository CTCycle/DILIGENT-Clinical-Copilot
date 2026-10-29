# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Read-only terminology queries for the Knowledge inspection views."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, aliased

from repositories.context import RepositoryContext
from repositories.schemas.knowledge import Drug, DrugAlias, LiverToxMonograph
from repositories.schemas.terminology import (
    DrugLiverToxLink,
    MedicinalProduct,
    TerminologyConflict,
    TerminologyImportRun,
)
from repositories.terminology_repository import RESOLVABLE_STATUSES

###############################################################################
class TerminologyInspectionRepository:

    # -------------------------------------------------------------------------
    def __init__(self, context: RepositoryContext) -> None:
        self.session_factory = context.session_factory

    # -------------------------------------------------------------------------
    @staticmethod
    def _count(db_session: Session, statement: Any) -> int:
        return int(db_session.execute(statement).scalar_one() or 0)

    # -------------------------------------------------------------------------
    def summary_counts(self) -> dict[str, Any]:
        with self.session_factory() as db_session:
            count = func.count()
            return {
                "revision": db_session.execute(
                    select(func.max(TerminologyImportRun.id)).where(TerminologyImportRun.status == "completed")
                ).scalar_one_or_none(),
                "concepts": self._count(db_session, select(count).select_from(Drug)),
                "aliases": self._count(db_session, select(count).select_from(DrugAlias)),
                "resolvable_aliases": self._count(
                    db_session,
                    select(count).select_from(DrugAlias).where(DrugAlias.validation_status.in_(RESOLVABLE_STATUSES)),
                ),
                "provisional_aliases": self._count(
                    db_session,
                    select(count).select_from(DrugAlias).where(DrugAlias.validation_status == "provisional"),
                ),
                "products": self._count(db_session, select(count).select_from(MedicinalProduct)),
                "verified_products": self._count(
                    db_session,
                    select(count)
                    .select_from(MedicinalProduct)
                    .where(MedicinalProduct.composition_status == "verified"),
                ),
                "livertox_links": self._count(db_session, select(count).select_from(DrugLiverToxLink)),
                "linked_monographs": self._count(
                    db_session, select(func.count(func.distinct(DrugLiverToxLink.monograph_id)))
                ),
                "total_monographs": self._count(db_session, select(count).select_from(LiverToxMonograph)),
                "open_conflicts": self._count(
                    db_session,
                    select(count).select_from(TerminologyConflict).where(TerminologyConflict.status == "open"),
                ),
            }

    # -------------------------------------------------------------------------
    def source_counts(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        with self.session_factory() as db_session:
            for model, field in (
                (DrugAlias, "aliases"),
                (MedicinalProduct, "products"),
            ):
                for source, total in db_session.execute(
                    select(model.source, func.count()).group_by(model.source)
                ).all():
                    counts.setdefault(str(source), {})[field] = int(total)
            for source, total in db_session.execute(
                select(TerminologyConflict.source, func.count())
                .where(TerminologyConflict.status == "open")
                .group_by(TerminologyConflict.source)
            ).all():
                counts.setdefault(str(source), {})["open_conflicts"] = int(total)
        return counts

    # -------------------------------------------------------------------------
    def latest_runs(self) -> dict[str, dict[str, Any]]:
        """Latest run and latest completed run per source."""
        latest: dict[str, dict[str, Any]] = {}
        with self.session_factory() as db_session:
            rows = db_session.execute(
                select(TerminologyImportRun).order_by(TerminologyImportRun.id.desc())
            ).scalars()
            for run in rows:
                entry = latest.setdefault(run.source, {})
                entry.setdefault("last", run)
                if run.status == "completed":
                    entry.setdefault("completed", run)
            return {
                source: {key: self._run_payload(run) for key, run in entry.items()}
                for source, entry in latest.items()
            }

    # -------------------------------------------------------------------------
    @staticmethod
    def _run_payload(run: TerminologyImportRun) -> dict[str, Any]:
        return {
            "id": int(run.id),
            "source": run.source,
            "source_version": run.source_version,
            "mode": run.mode,
            "status": run.status,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "stats": run.stats_json,
            "error": run.error,
        }

    # -------------------------------------------------------------------------
    def list_runs(self, *, limit: int) -> list[dict[str, Any]]:
        with self.session_factory() as db_session:
            runs = db_session.execute(
                select(TerminologyImportRun).order_by(TerminologyImportRun.id.desc()).limit(max(1, limit))
            ).scalars()
            return [self._run_payload(run) for run in runs]

    # -------------------------------------------------------------------------
    def list_conflicts(self, *, status: str | None, offset: int, limit: int) -> tuple[list[dict[str, Any]], int]:
        existing = aliased(Drug)
        proposed = aliased(Drug)
        conditions = [TerminologyConflict.status == status] if status else []
        with self.session_factory() as db_session:
            total = self._count(
                db_session, select(func.count()).select_from(TerminologyConflict).where(*conditions)
            )
            rows = db_session.execute(
                select(TerminologyConflict, existing.canonical_name, proposed.canonical_name)
                .outerjoin(existing, existing.id == TerminologyConflict.existing_drug_id)
                .outerjoin(proposed, proposed.id == TerminologyConflict.proposed_drug_id)
                .where(*conditions)
                .order_by(TerminologyConflict.id.desc())
                .offset(max(0, offset))
                .limit(max(1, limit))
            ).all()
            items = [
                {
                    "id": int(conflict.id),
                    "source": conflict.source,
                    "conflict_type": conflict.conflict_type,
                    "subject": conflict.subject,
                    "existing_drug": existing_name,
                    "proposed_drug": proposed_name,
                    "detail": conflict.detail,
                    "status": conflict.status,
                    "detected_at": conflict.detected_at,
                }
                for conflict, existing_name, proposed_name in rows
            ]
        return items, total

    # -------------------------------------------------------------------------
    def drug_exists(self, drug_id: int) -> bool:
        with self.session_factory() as db_session:
            return db_session.get(Drug, int(drug_id)) is not None
