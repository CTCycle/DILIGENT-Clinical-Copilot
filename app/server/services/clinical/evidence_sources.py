# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from common.utils.logger import logger
from domain.clinical.entities import (
    EVIDENCE_SOURCE_LABELS,
    EVIDENCE_SOURCES,
    EvidenceSelection,
)
from domain.clinical.robustness import EvidenceSourceStatus, RagReadiness
from repositories.context import RepositoryContext
from repositories.dilirank_repository import DiliRankRepository
from repositories.knowledge_repository import KnowledgeRepository

###############################################################################
def dilirank_repository_for(
    knowledge_repository: KnowledgeRepository,
) -> DiliRankRepository | None:
    context = getattr(knowledge_repository, "context", None)
    return DiliRankRepository(context) if isinstance(context, RepositoryContext) else None


###############################################################################
def check_evidence_sources(
    selection: EvidenceSelection,
    *,
    knowledge_repository: KnowledgeRepository,
    dilirank_repository: DiliRankRepository | None,
    documents_readiness: RagReadiness,
) -> list[EvidenceSourceStatus]:
    """Report local availability of every evidence source without contacting
    external services."""
    statuses: list[EvidenceSourceStatus] = []
    for source in EVIDENCE_SOURCES:
        available, message = _source_availability(
            source,
            knowledge_repository=knowledge_repository,
            dilirank_repository=dilirank_repository,
            documents_readiness=documents_readiness,
        )
        statuses.append(
            EvidenceSourceStatus(
                source=source,
                label=EVIDENCE_SOURCE_LABELS[source],
                selected=source in selection.sources,
                available=available,
                message=message,
            )
        )
    return statuses


###############################################################################
def _source_availability(
    source: str,
    *,
    knowledge_repository: KnowledgeRepository,
    dilirank_repository: DiliRankRepository | None,
    documents_readiness: RagReadiness,
) -> tuple[bool, str | None]:
    try:
        if source == "livertox":
            rows, _ = knowledge_repository.list_livertox_catalog(
                search=None, offset=0, limit=1
            )
            if rows:
                return True, None
            return False, "LiverTox has not been imported. Run the LiverTox update from Knowledge."
        if source == "dilirank":
            if dilirank_repository is not None and dilirank_repository.count_records() > 0:
                return True, None
            return False, "DILIrank has not been imported. Run the DILIrank update from Knowledge."
        return documents_readiness.available, documents_readiness.message
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Evidence source availability check failed: source=%s error_type=%s",
            source,
            type(exc).__name__,
        )
        return False, f"{EVIDENCE_SOURCE_LABELS[source]} could not be inspected."
