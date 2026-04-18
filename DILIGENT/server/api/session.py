from __future__ import annotations

from fastapi import APIRouter, Body, status
from fastapi.responses import PlainTextResponse
from pydantic import RootModel

from DILIGENT.server.domain.clinical.entities import ClinicalSessionRequest
from DILIGENT.server.domain.jobs import JobCancelResponse, JobStartResponse, JobStatusResponse
from DILIGENT.server.services import session_service as session_service_module
from DILIGENT.server.services.clinical.hepatox import HepatoxConsultation
from DILIGENT.server.services.clinical.job_progress import CLINICAL_PROGRESS_MESSAGES
from DILIGENT.server.services.clinical.preparation import ClinicalKnowledgePreparation
from DILIGENT.server.services.jobs import job_manager
from DILIGENT.server.services.session_service import (
    ClinicalJobCancelled,
    ClinicalSessionService,
    NarrativeBuilder,
    StageProgressFractionCallback,
    build_failed_session_payload,
    execute_clinical_job,
    run_clinical_job,
)


###############################################################################
class ClinicalSessionReportResponse(RootModel[str]):
    pass


###############################################################################
class ClinicalSessionEndpoint:
    def __init__(self, *, router: APIRouter, service: ClinicalSessionService) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    def __getattr__(self, name: str):
        return getattr(self.service, name)

    # -------------------------------------------------------------------------
    def __setattr__(self, name: str, value) -> None:
        if name in {"router", "service"}:
            object.__setattr__(self, name, value)
            return
        service = self.__dict__.get("service")
        if service is not None and hasattr(service, name):
            setattr(service, name, value)
            return
        object.__setattr__(self, name, value)

    # -------------------------------------------------------------------------
    @staticmethod
    def build_structured_clinical_context(*args, **kwargs) -> str:
        return ClinicalSessionService.build_structured_clinical_context(*args, **kwargs)

    # -------------------------------------------------------------------------
    async def process_single_patient(self, payload, *, progress_callback=None, stop_check=None):
        return await self.service.process_single_patient(
            payload,
            progress_callback=progress_callback,
            stop_check=stop_check,
        )

    # -------------------------------------------------------------------------
    async def start_clinical_session(
        self,
        request_payload: ClinicalSessionRequest = Body(...),
    ) -> PlainTextResponse:
        return await self.service.start_clinical_session(request_payload)

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


router = APIRouter(tags=["session"])
service = ClinicalSessionService(
    drugs_parser=session_service_module.drugs_parser,
    disease_extractor=session_service_module.disease_extractor,
    lab_extractor=session_service_module.lab_extractor,
    pattern_analyzer=session_service_module.pattern_analyzer,
    rucam_estimator=session_service_module.rucam_estimator,
    serializer=session_service_module.serializer,
    payload_sanitizer=session_service_module.payload_sanitization_service,
    input_preparator=ClinicalKnowledgePreparation(),
    hepatox_consultation_cls=HepatoxConsultation,
    job_manager=job_manager,
    router=router,
)
endpoint = ClinicalSessionEndpoint(router=router, service=service)


###############################################################################
async def start_clinical_session(
    request_payload: ClinicalSessionRequest = Body(...),
) -> PlainTextResponse:
    return await endpoint.start_clinical_session(request_payload)


###############################################################################
def start_clinical_job(
    request_payload: ClinicalSessionRequest = Body(...),
) -> JobStartResponse:
    return endpoint.start_clinical_job(request_payload)


###############################################################################
def get_clinical_job_status(job_id: str) -> JobStatusResponse:
    return endpoint.get_clinical_job_status(job_id)


###############################################################################
def cancel_clinical_job(job_id: str) -> JobCancelResponse:
    return endpoint.cancel_clinical_job(job_id)


###############################################################################
def report_clinical_job_progress(
    job_id: str,
    *,
    stage: str,
    progress: float,
    message: str | None = None,
) -> None:
    _ = message
    if endpoint.job_manager.should_stop(job_id):
        raise ClinicalJobCancelled("Clinical job stop requested.")
    bounded = min(100.0, max(0.0, float(progress)))
    status_message = CLINICAL_PROGRESS_MESSAGES.get(stage, stage.replace("_", " ").strip())
    endpoint.job_manager.update_progress(job_id, bounded)
    endpoint.job_manager.update_result(
        job_id,
        {
            "progress_stage": stage,
            "progress_message": status_message,
        },
    )


router.add_api_route(
    "/clinical",
    start_clinical_session,
    methods=["POST"],
    response_model=ClinicalSessionReportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    response_class=PlainTextResponse,
)
router.add_api_route(
    "/clinical/jobs",
    start_clinical_job,
    methods=["POST"],
    response_model=JobStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
router.add_api_route(
    "/clinical/jobs/{job_id}",
    get_clinical_job_status,
    methods=["GET"],
    response_model=JobStatusResponse,
    status_code=status.HTTP_200_OK,
)
router.add_api_route(
    "/clinical/jobs/{job_id}",
    cancel_clinical_job,
    methods=["DELETE"],
    response_model=JobCancelResponse,
    status_code=status.HTTP_200_OK,
)

__all__ = [
    "ClinicalSessionEndpoint",
    "NarrativeBuilder",
    "ClinicalJobCancelled",
    "build_failed_session_payload",
    "cancel_clinical_job",
    "endpoint",
    "execute_clinical_job",
    "get_clinical_job_status",
    "report_clinical_job_progress",
    "router",
    "run_clinical_job",
    "StageProgressFractionCallback",
    "start_clinical_job",
    "start_clinical_session",
]
