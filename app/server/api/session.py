from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Body, Response, status

from domain.clinical.entities import (
    ClinicalSectionTemplateResponse,
    ClinicalSessionRequest,
)
from domain.clinical.robustness import ClinicalInputPreflightResult
from domain.jobs import (
    JobCancelResponse,
    JobStartResponse,
    JobStatusResponse,
)

from services.clinical.template import get_clinical_section_template
from services.runtime.jobs import get_job_manager
from services.session.factory import build_clinical_session_service
from api.session_validation import validate_clinical_session_request
from services.session.session_service import ClinicalSessionService

router = APIRouter(tags=["session"])

###############################################################################
class ClinicalSessionEndpoint:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        router: APIRouter,
        service: ClinicalSessionService | None = None,
        service_factory: Callable[[], ClinicalSessionService] | None = None,
    ) -> None:
        self.router = router
        self._service = service
        self._service_factory = service_factory

    # -------------------------------------------------------------------------
    @property
    def service(self) -> ClinicalSessionService:
        if self._service is None:
            if self._service_factory is None:
                raise RuntimeError("Clinical session service is not configured.")
            self._service = self._service_factory()
        return self._service

    # -------------------------------------------------------------------------
    @service.setter
    def service(self, value: ClinicalSessionService) -> None:
        self._service = value

    # -------------------------------------------------------------------------
    def start_clinical_job(
        self,
        request_payload: ClinicalSessionRequest = Body(...),
    ) -> JobStartResponse:
        validate_clinical_session_request(request_payload)
        return self.service.start_clinical_job(request_payload)

    # -------------------------------------------------------------------------
    def validate_clinical_input(
        self,
        request_payload: ClinicalSessionRequest = Body(...),
    ) -> ClinicalInputPreflightResult:
        return self.service.validate_clinical_input(request_payload)

    # -------------------------------------------------------------------------
    def get_clinical_job_status(
        self,
        job_id: str,
        response: Response,
    ) -> JobStatusResponse:
        response.headers["Cache-Control"] = (
            "no-store, no-cache, max-age=0, must-revalidate"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return self.service.get_clinical_job_status(job_id)

    # -------------------------------------------------------------------------
    def cancel_clinical_job(self, job_id: str) -> JobCancelResponse:
        return self.service.cancel_clinical_job(job_id)

    # -------------------------------------------------------------------------
    @staticmethod
    def get_section_template() -> ClinicalSectionTemplateResponse:
        return get_clinical_section_template()

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/clinical/section-template",
            self.get_section_template,
            methods=["GET"],
            response_model=ClinicalSectionTemplateResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/clinical/validate-input",
            self.validate_clinical_input,
            methods=["POST"],
            response_model=ClinicalInputPreflightResult,
            status_code=status.HTTP_200_OK,
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
    service_factory=lambda: build_clinical_session_service(get_job_manager()),
).add_routes()
