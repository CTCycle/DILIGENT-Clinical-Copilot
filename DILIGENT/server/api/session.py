from __future__ import annotations

from fastapi import APIRouter, Body, status
from fastapi.responses import PlainTextResponse
from pydantic import RootModel

from DILIGENT.server.domain.clinical.entities import ClinicalSessionRequest
from DILIGENT.server.domain.jobs import JobCancelResponse, JobStartResponse, JobStatusResponse
from DILIGENT.server.services.clinical.hepatox_core import HepatoxConsultation
from DILIGENT.server.services.clinical.job_progress import CLINICAL_PROGRESS_MESSAGES
from DILIGENT.server.services.clinical.preparation import ClinicalKnowledgePreparation
from DILIGENT.server.services.jobs import job_manager
from DILIGENT.server.services.session_service import (
    ClinicalJobCancelled,
    ClinicalSessionService,
    NarrativeBuilder,
    StageProgressFractionCallback,
    build_failed_session_payload,
    disease_extractor,
    drugs_parser,
    execute_clinical_job,
    lab_extractor,
    pattern_analyzer,
    payload_sanitization_service,
    rucam_estimator,
    run_clinical_job,
    serializer,
)


###############################################################################
class ClinicalSessionReportResponse(RootModel[str]):
    pass


router = APIRouter(tags=["session"])
service = ClinicalSessionService(
    drugs_parser=drugs_parser,
    disease_extractor=disease_extractor,
    lab_extractor=lab_extractor,
    pattern_analyzer=pattern_analyzer,
    rucam_estimator=rucam_estimator,
    serializer=serializer,
    payload_sanitizer=payload_sanitization_service,
    input_preparator=ClinicalKnowledgePreparation(),
    hepatox_consultation_cls=HepatoxConsultation,
    job_manager=job_manager,
    router=router,
)


###############################################################################
async def start_clinical_session(
    request_payload: ClinicalSessionRequest = Body(...),
) -> PlainTextResponse:
    return await service.start_clinical_session(request_payload)


###############################################################################
def start_clinical_job(
    request_payload: ClinicalSessionRequest = Body(...),
) -> JobStartResponse:
    return service.start_clinical_job(request_payload)


###############################################################################
def get_clinical_job_status(job_id: str) -> JobStatusResponse:
    return service.get_clinical_job_status(job_id)


###############################################################################
def cancel_clinical_job(job_id: str) -> JobCancelResponse:
    return service.cancel_clinical_job(job_id)


###############################################################################
def report_clinical_job_progress(
    job_id: str,
    *,
    stage: str,
    progress: float,
    message: str | None = None,
) -> None:
    _ = message
    if service.job_manager.should_stop(job_id):
        raise ClinicalJobCancelled("Clinical job stop requested.")
    bounded = min(100.0, max(0.0, float(progress)))
    status_message = CLINICAL_PROGRESS_MESSAGES.get(stage, stage.replace("_", " ").strip())
    service.job_manager.update_progress(job_id, bounded)
    service.job_manager.update_result(
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
    "NarrativeBuilder",
    "ClinicalJobCancelled",
    "build_failed_session_payload",
    "cancel_clinical_job",
    "execute_clinical_job",
    "get_clinical_job_status",
    "report_clinical_job_progress",
    "router",
    "run_clinical_job",
    "StageProgressFractionCallback",
    "start_clinical_job",
    "start_clinical_session",
]
