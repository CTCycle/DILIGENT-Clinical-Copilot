from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, status

from DILIGENT.server.domain.research.entities import (
    ResearchRequest,
    ResearchResponse,
)
from DILIGENT.server.services.research.brave import brave_research_service

router = APIRouter(tags=["research"])


###############################################################################
class ResearchEndpoint:
    def __init__(self, *, router: APIRouter) -> None:
        self.router = router

    # -------------------------------------------------------------------------
    async def run_research(
        self,
        payload: ResearchRequest = Body(...),
    ) -> ResearchResponse:
        if not brave_research_service.is_configured():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "No active Brave Search access key configured. "
                    "Add and activate a Brave Search key in the access-key manager."
                ),
            )
        outcome = await brave_research_service.search_sources(
            question=payload.question,
            mode=payload.mode,
            allowed_domains=payload.allowed_domains,
            blocked_domains=payload.blocked_domains,
        )
        answer, citations = await brave_research_service.generate_answer_with_citations(
            question=payload.question,
            sources=outcome.sources,
        )
        return ResearchResponse(
            answer=answer,
            sources=outcome.sources,
            citations=citations,
            message=outcome.message,
        )

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/research",
            self.run_research,
            methods=["POST"],
            response_model=ResearchResponse,
            status_code=status.HTTP_200_OK,
        )


endpoint = ResearchEndpoint(router=router)
endpoint.add_routes()
