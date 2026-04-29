from __future__ import annotations

from fastapi import APIRouter, Body, status
from fastapi.responses import PlainTextResponse

from DILIGENT.server.domain.clinical.entities import (
    ClinicalSessionReportResponse,
    ClinicalSessionRequest,
)
from DILIGENT.server.domain.jobs import JobCancelResponse, JobStartResponse, JobStatusResponse
from DILIGENT.server.services.clinical.hepatox_core import HepatoxConsultation
from DILIGENT.server.services.clinical.preparation import ClinicalKnowledgePreparation
from DILIGENT.server.services.runtime.jobs import job_manager
from DILIGENT.server.services.session.session_service import (
    ClinicalSessionService,
    disease_extractor,
    drugs_parser,
    lab_extractor,
    pattern_analyzer,
    payload_sanitization_service,
    rucam_estimator,
    serializer,
)


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
)


###############################################################################
async def start_clinical_session(
    request_payload: ClinicalSessionRequest = Body(...),
) -> PlainTextResponse:
    report = await service.start_clinical_session(request_payload)
    return PlainTextResponse(content=report, status_code=status.HTTP_202_ACCEPTED)


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

