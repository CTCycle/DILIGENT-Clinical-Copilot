# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from typing import Any, cast

from domain.agents import AgentExecutionContext, AssessmentConfig
from domain.clinical.entities import ClinicalSessionRequest


###############################################################################
def resolve_dili_agent(
    service: Any,
    request_payload: ClinicalSessionRequest,
) -> tuple[AgentExecutionContext, ClinicalSessionRequest]:
    """Resolve the selected DILI agent revision and bind it to the request.

    The agent's retrieval policy is authoritative for evidence sources, and
    the resolved identity/revision is pinned so a replayed launch cannot
    silently pick up a later edit.
    """

    context: AgentExecutionContext = service.agent_service.resolve_for_execution(
        capability="assessment",
        agent_id=request_payload.agent_id,
        expected_revision=request_payload.agent_revision,
    )
    config = cast(AssessmentConfig, context.configuration)
    bound = request_payload.model_copy(
        update={
            "agent_id": context.agent_id,
            "agent_revision": context.revision_number,
            "evidence_sources": list(config.retrieval.evidence_sources),
        }
    )
    return context, bound
