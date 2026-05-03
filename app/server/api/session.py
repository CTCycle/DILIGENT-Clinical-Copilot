from __future__ import annotations

from fastapi import APIRouter, Body, status
from fastapi.responses import PlainTextResponse

from domain.clinical.entities import ClinicalSessionRequest
from domain.jobs import (
    JobCancelResponse,
    JobStartResponse,
    JobStatusResponse,
)
from services.runtime.jobs import get_job_manager
from services.session.factory import build_clinical_session_service
from services.session.session_service import ClinicalSessionService

router = APIRouter(tags=["session"])


###############################################################################
class ClinicalSessionEndpoint:
    def __init__(self, *, router: APIRouter, service: ClinicalSessionService) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    async def start_clinical_session(
        self,
        request_payload: ClinicalSessionRequest = Body(...),
    ) -> PlainTextResponse:
        report = await self.service.start_clinical_session(request_payload)
        return PlainTextResponse(content=report, status_code=status.HTTP_202_ACCEPTED)

    # -------------------------------------------------------------------------
    def start_clinical_job(
        self,
        request_payload: ClinicalSessionRequest = Body(...),
    ) -> JobStartResponse:
        return self.service.start_clinical_job(request_payload)

    # -------------------------------------------------------------------------
    def get_clinical_job_status(self, job_id: str) -> JobStatusResponse:
        return self.service.get_clinical_job_status(job_id)

    # -------------------------------------------------------------------------
    def cancel_clinical_job(self, job_id: str) -> JobCancelResponse:
        return self.service.cancel_clinical_job(job_id)

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/clinical",
            self.start_clinical_session,
            methods=["POST"],
            status_code=status.HTTP_202_ACCEPTED,
            response_model=str,
            response_class=PlainTextResponse,
        )
        self.router.add_api_route(
            "/clinical/jobs",
            self.start_clinical_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/clinical/jobs/{job_id}",
            self.get_clinical_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/clinical/jobs/{job_id}",
            self.cancel_clinical_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )


ClinicalSessionEndpoint(
    router=router,
    service=build_clinical_session_service(get_job_manager()),
).add_routes()
