# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Body, HTTPException, Query, Response, status
from sqlalchemy.exc import SQLAlchemyError
from uuid import UUID

from domain.clinical.entities import (
    ClinicalSectionTemplateResponse,
    ClinicalSessionRequest,
)
from domain.clinical.robustness import ClinicalInputPreflightResult
from domain.clinical.drafts import ClinicalDraftSnapshot, ClinicalDraftWrite
from domain.jobs import (
    JobCancelResponse,
    JobStartResponse,
    JobStatusResponse,
)

from services.clinical.template import get_clinical_section_template
from services.runtime.jobs import get_job_manager
from services.session.factory import (
    build_clinical_draft_service,
    build_clinical_session_service,
)
from api.session_validation import validate_clinical_session_request
from services.session.session_service import ClinicalSessionService
from services.session.clinical_draft_service import (
    ClinicalDraftService,
    DraftConflictError,
)

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


clinical_session_endpoint = ClinicalSessionEndpoint(
    router=router,
    service_factory=lambda: build_clinical_session_service(get_job_manager()),
)
clinical_session_endpoint.add_routes()


###############################################################################
class ClinicalDraftEndpoint:
    def __init__(
        self,
        *,
        router: APIRouter,
        service: ClinicalDraftService | None = None,
        service_factory: Callable[[], ClinicalDraftService] | None = None,
    ) -> None:
        self.router = router
        self._service = service
        self._service_factory = service_factory

    @property
    def service(self) -> ClinicalDraftService:
        if self._service is None:
            if self._service_factory is None:
                raise RuntimeError("Clinical draft service is not configured.")
            self._service = self._service_factory()
        return self._service

    @service.setter
    def service(self, value: ClinicalDraftService) -> None:
        self._service = value

    @staticmethod
    def _no_store(response: Response) -> None:
        response.headers["Cache-Control"] = "no-store"

    @staticmethod
    def _unavailable() -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Draft storage is unavailable. Please retry shortly.",
            headers={"Cache-Control": "no-store"},
        )

    @staticmethod
    def _conflict() -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Saved draft changed. Reload the draft before saving.",
            headers={"Cache-Control": "no-store"},
        )

    def get_draft(self, response: Response) -> ClinicalDraftSnapshot:
        self._no_store(response)
        try:
            return self.service.get()
        except (SQLAlchemyError, RuntimeError) as exc:
            raise self._unavailable() from exc

    def save_draft(
        self,
        request_payload: ClinicalDraftWrite,
        response: Response,
    ) -> ClinicalDraftSnapshot:
        self._no_store(response)
        try:
            return self.service.save(
                request_payload.form,
                expected_revision=request_payload.expected_revision,
                expected_draft_key=request_payload.expected_draft_key,
            )
        except DraftConflictError as exc:
            raise self._conflict() from exc
        except (SQLAlchemyError, RuntimeError) as exc:
            raise self._unavailable() from exc

    def discard_draft(
        self,
        response: Response,
        expected_revision: int = Query(ge=0),
        expected_draft_key: UUID | None = Query(default=None),
    ) -> ClinicalDraftSnapshot:
        self._no_store(response)
        try:
            return self.service.discard(
                expected_revision=expected_revision,
                expected_draft_key=expected_draft_key,
            )
        except DraftConflictError as exc:
            raise self._conflict() from exc
        except (SQLAlchemyError, RuntimeError) as exc:
            raise self._unavailable() from exc

    def add_routes(self) -> None:
        self.router.add_api_route(
            "/clinical/draft",
            self.get_draft,
            methods=["GET"],
            response_model=ClinicalDraftSnapshot,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/clinical/draft",
            self.save_draft,
            methods=["PUT"],
            response_model=ClinicalDraftSnapshot,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/clinical/draft",
            self.discard_draft,
            methods=["DELETE"],
            response_model=ClinicalDraftSnapshot,
            status_code=status.HTTP_200_OK,
        )


clinical_draft_endpoint = ClinicalDraftEndpoint(
    router=router,
    service_factory=build_clinical_draft_service,
)
clinical_draft_endpoint.add_routes()


###############################################################################
def rebind_database_services() -> None:
    clinical_session_endpoint.service = build_clinical_session_service(get_job_manager())
    clinical_draft_endpoint.service = build_clinical_draft_service()
