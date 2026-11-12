# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Path, Query, status

from domain.agents import (
    AgentCreateRequest,
    AgentDuplicateRequest,
    AgentLifecycleRequest,
    AgentListResponse,
    AgentResponse,
    AgentRevisionListResponse,
    AgentUpdateRequest,
    Capability,
)
from services.agents import AgentService

router = APIRouter(prefix="/agents", tags=["agents"])

AgentId = Annotated[str, Path(min_length=1)]


###############################################################################
class AgentsEndpoint:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        router: APIRouter,
        service: AgentService | None = None,
    ) -> None:
        self.router = router
        self.service = service or AgentService()

    # -------------------------------------------------------------------------
    def list_agents(
        self,
        include_archived: Annotated[bool, Query()] = False,
        selectable_for: Annotated[Capability | None, Query()] = None,
    ) -> AgentListResponse:
        return self.service.list_agents(
            include_archived=include_archived, selectable_for=selectable_for
        )

    # -------------------------------------------------------------------------
    def create_agent(
        self,
        payload: AgentCreateRequest = Body(...),
    ) -> AgentResponse:
        return self.service.create_agent(payload)

    # -------------------------------------------------------------------------
    def get_agent(self, agent_id: AgentId) -> AgentResponse:
        return self.service.get_agent(agent_id)

    # -------------------------------------------------------------------------
    def update_agent(
        self,
        agent_id: AgentId,
        payload: AgentUpdateRequest = Body(...),
    ) -> AgentResponse:
        return self.service.update_agent(agent_id, payload)

    # -------------------------------------------------------------------------
    def duplicate_agent(
        self,
        agent_id: AgentId,
        payload: AgentDuplicateRequest | None = Body(default=None),
    ) -> AgentResponse:
        return self.service.duplicate_agent(agent_id, payload or AgentDuplicateRequest())

    # -------------------------------------------------------------------------
    def set_default(self, agent_id: AgentId) -> AgentResponse:
        return self.service.set_default(agent_id)

    # -------------------------------------------------------------------------
    def set_lifecycle(
        self,
        agent_id: AgentId,
        payload: AgentLifecycleRequest = Body(...),
    ) -> AgentResponse:
        return self.service.set_lifecycle(agent_id, payload)

    # -------------------------------------------------------------------------
    def delete_agent(self, agent_id: AgentId) -> None:
        self.service.delete_agent(agent_id)

    # -------------------------------------------------------------------------
    def list_revisions(self, agent_id: AgentId) -> AgentRevisionListResponse:
        return self.service.list_revisions(agent_id)

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "",
            self.list_agents,
            methods=["GET"],
            response_model=AgentListResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "",
            self.create_agent,
            methods=["POST"],
            response_model=AgentResponse,
            status_code=status.HTTP_201_CREATED,
        )
        self.router.add_api_route(
            "/{agent_id}",
            self.get_agent,
            methods=["GET"],
            response_model=AgentResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/{agent_id}",
            self.update_agent,
            methods=["PUT"],
            response_model=AgentResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/{agent_id}/duplicate",
            self.duplicate_agent,
            methods=["POST"],
            response_model=AgentResponse,
            status_code=status.HTTP_201_CREATED,
        )
        self.router.add_api_route(
            "/{agent_id}/default",
            self.set_default,
            methods=["PUT"],
            response_model=AgentResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/{agent_id}/lifecycle",
            self.set_lifecycle,
            methods=["PUT"],
            response_model=AgentResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/{agent_id}",
            self.delete_agent,
            methods=["DELETE"],
            status_code=status.HTTP_204_NO_CONTENT,
        )
        self.router.add_api_route(
            "/{agent_id}/revisions",
            self.list_revisions,
            methods=["GET"],
            response_model=AgentRevisionListResponse,
            status_code=status.HTTP_200_OK,
        )


agents_endpoint = AgentsEndpoint(router=router, service=AgentService())
agents_endpoint.add_routes()


###############################################################################
def rebind_database_services() -> None:
    agents_endpoint.service = AgentService()
