# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Body, Response, status

from domain.clinical.entities import (
    ClinicalSectionTemplateResponse,
    ClinicalSessionRequest,
)
from domain.clinical.lifecycle import (
    ClinicalSessionInput,
    ClinicalSessionInputUpdate,
    ClinicalSessionInputWrite,
    ClinicalSessionState,
)
from domain.clinical.robustness import ClinicalInputPreflightResult

from services.clinical.template import get_clinical_section_template
from services.runtime.jobs import get_job_manager
from services.session.factory import (
    build_clinical_lifecycle_service,
    build_clinical_session_service,
)
from api.session_validation import validate_clinical_session_request
from services.session.lifecycle import ClinicalSessionLifecycleService
from services.session.session_service import ClinicalSessionService

router = APIRouter(tags=["session"])


###############################################################################
def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store, no-cache, max-age=0, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"


###############################################################################
class ClinicalSessionEndpoint:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        router: APIRouter,
        service_factory: Callable[[], ClinicalSessionService],
        lifecycle_factory: Callable[
            [ClinicalSessionService], ClinicalSessionLifecycleService
        ],
    ) -> None:
        self.router = router
        self._service: ClinicalSessionService | None = None
        self._lifecycle: ClinicalSessionLifecycleService | None = None
        self._service_factory = service_factory
        self._lifecycle_factory = lifecycle_factory

    # -------------------------------------------------------------------------
    @property
    def service(self) -> ClinicalSessionService:
        if self._service is None:
            self._service = self._service_factory()
        return self._service

    # -------------------------------------------------------------------------
    @property
    def lifecycle(self) -> ClinicalSessionLifecycleService:
        if self._lifecycle is None:
            self._lifecycle = self._lifecycle_factory(self.service)
        return self._lifecycle

    # -------------------------------------------------------------------------
    def rebind(self) -> None:
        self._service = None
        self._lifecycle = None

    # -------------------------------------------------------------------------
    @staticmethod
    def get_section_template() -> ClinicalSectionTemplateResponse:
        return get_clinical_section_template()

    # -------------------------------------------------------------------------
    def validate_clinical_input(
        self,
        request_payload: ClinicalSessionRequest = Body(...),
    ) -> ClinicalInputPreflightResult:
        return self.service.validate_clinical_input(request_payload)

    # -------------------------------------------------------------------------
    def create_session(
        self, payload: ClinicalSessionInputWrite, response: Response
    ) -> ClinicalSessionState:
        _no_store(response)
        return self.lifecycle.create_draft(payload)

    # -------------------------------------------------------------------------
    def get_session_input(
        self, session_id: int, response: Response
    ) -> ClinicalSessionInput:
        _no_store(response)
        return self.lifecycle.get_input(session_id)

    # -------------------------------------------------------------------------
    def update_session_input(
        self,
        session_id: int,
        payload: ClinicalSessionInputUpdate,
        response: Response,
    ) -> ClinicalSessionState:
        _no_store(response)
        return self.lifecycle.update_draft(session_id, payload)

    # -------------------------------------------------------------------------
    def launch_session(
        self,
        session_id: int,
        response: Response,
        request_payload: ClinicalSessionRequest = Body(...),
    ) -> ClinicalSessionState:
        _no_store(response)
        validate_clinical_session_request(request_payload)
        return self.lifecycle.launch(session_id, request_payload)

    # -------------------------------------------------------------------------
    def get_session_status(
        self, session_id: int, response: Response
    ) -> ClinicalSessionState:
        _no_store(response)
        return self.lifecycle.get_state(session_id)

    # -------------------------------------------------------------------------
    def cancel_session(
        self, session_id: int, response: Response
    ) -> ClinicalSessionState:
        _no_store(response)
        return self.lifecycle.cancel(session_id)

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        routes = (
            ("/clinical/section-template", self.get_section_template, "GET",
             ClinicalSectionTemplateResponse, status.HTTP_200_OK),
            ("/clinical/validate-input", self.validate_clinical_input, "POST",
             ClinicalInputPreflightResult, status.HTTP_200_OK),
            ("/clinical/sessions", self.create_session, "POST",
             ClinicalSessionState, status.HTTP_201_CREATED),
            ("/clinical/sessions/{session_id}/input", self.get_session_input, "GET",
             ClinicalSessionInput, status.HTTP_200_OK),
            ("/clinical/sessions/{session_id}/input", self.update_session_input, "PUT",
             ClinicalSessionState, status.HTTP_200_OK),
            ("/clinical/sessions/{session_id}/launch", self.launch_session, "POST",
             ClinicalSessionState, status.HTTP_202_ACCEPTED),
            ("/clinical/sessions/{session_id}/status", self.get_session_status, "GET",
             ClinicalSessionState, status.HTTP_200_OK),
            ("/clinical/sessions/{session_id}/cancel", self.cancel_session, "POST",
             ClinicalSessionState, status.HTTP_200_OK),
        )
        for path, endpoint, method, response_model, status_code in routes:
            self.router.add_api_route(
                path,
                endpoint,
                methods=[method],
                response_model=response_model,
                status_code=status_code,
            )


clinical_session_endpoint = ClinicalSessionEndpoint(
    router=router,
    service_factory=lambda: build_clinical_session_service(get_job_manager()),
    lifecycle_factory=build_clinical_lifecycle_service,
)
clinical_session_endpoint.add_routes()


###############################################################################
def rebind_database_services() -> None:
    clinical_session_endpoint.rebind()
