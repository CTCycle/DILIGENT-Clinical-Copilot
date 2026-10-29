# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from typing import Any, Literal

from repositories.context import RepositoryContext
from repositories.terminology_inspection_repository import TerminologyInspectionRepository
from repositories.terminology_lookup_repository import TerminologyLookupRepository
from services.terminology.factory import build_terminology_resolver
from services.terminology.index import get_terminology_index_cache
from services.terminology.normalization import fold
from services.updater.terminology.factory import (
    TERMINOLOGY_SOURCE_LICENSES,
    enabled_terminology_sources,
)

# User decisions use their own source so no terminology refresh replaces them.
REVIEWER_SOURCE = "reviewer"

###############################################################################
class InspectionTerminologyMixin:
    """Terminology coverage, import reports, conflicts and reviewer decisions."""

    knowledge_repository: Any

    # -------------------------------------------------------------------------
    def _terminology_context(self) -> RepositoryContext:
        context = getattr(self.knowledge_repository, "context", None)
        if not isinstance(context, RepositoryContext):
            raise RuntimeError("Terminology repository is unavailable.")
        return context

    # -------------------------------------------------------------------------
    def get_terminology_summary(self) -> dict[str, Any]:
        repository = TerminologyInspectionRepository(self._terminology_context())
        summary = repository.summary_counts()
        counts = repository.source_counts()
        runs = repository.latest_runs()
        enabled = enabled_terminology_sources()
        sources: list[dict[str, Any]] = []
        for source, (license_text, attribution) in TERMINOLOGY_SOURCE_LICENSES.items():
            last = runs.get(source, {}).get("last") or {}
            completed = runs.get(source, {}).get("completed") or {}
            sources.append(
                {
                    "source": source,
                    "enabled": enabled.get(source, False),
                    "license": license_text,
                    "attribution": attribution,
                    "last_status": last.get("status"),
                    "last_version": last.get("source_version"),
                    "last_finished_at": last.get("finished_at"),
                    "last_completed_version": completed.get("source_version"),
                    "last_completed_at": completed.get("finished_at"),
                    "last_error": last.get("error"),
                    **counts.get(source, {}),
                }
            )
        return {**summary, "index": get_terminology_index_cache().stats(), "sources": sources}

    # -------------------------------------------------------------------------
    def list_terminology_runs(self, *, limit: int) -> list[dict[str, Any]]:
        return TerminologyInspectionRepository(self._terminology_context()).list_runs(limit=limit)

    # -------------------------------------------------------------------------
    def list_terminology_conflicts(
        self,
        *,
        status: str | None,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        items, total = TerminologyInspectionRepository(self._terminology_context()).list_conflicts(
            status=status, offset=offset, limit=limit
        )
        return {"items": items, "total": total, "offset": offset, "limit": limit}

    # -------------------------------------------------------------------------
    def resolve_terminology_name(self, name: str, *, allow_external: bool) -> dict[str, Any]:
        lookup = TerminologyLookupRepository(self._terminology_context())
        resolver = build_terminology_resolver(lookup, allow_external=allow_external)
        return resolver.resolve(name, allow_external=allow_external).model_dump(mode="json")

    # -------------------------------------------------------------------------
    def record_terminology_alias_decision(
        self,
        *,
        alias: str,
        drug_id: int,
        decision: Literal["confirmed", "rejected"],
    ) -> dict[str, Any] | None:
        context = self._terminology_context()
        alias_text = alias.strip()
        alias_norm = fold(alias_text)
        if not alias_norm or not TerminologyInspectionRepository(context).drug_exists(drug_id):
            return None
        TerminologyLookupRepository(context).record_alias(
            drug_id=drug_id,
            alias=alias_text,
            alias_norm=alias_norm,
            alias_kind="synonym",
            source=REVIEWER_SOURCE,
            validation_status=decision,
        )
        get_terminology_index_cache().invalidate()
        return {"alias": alias_text, "drug_id": drug_id, "validation_status": decision}
