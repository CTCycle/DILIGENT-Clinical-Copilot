from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, status
from fastapi.responses import PlainTextResponse

from DILIGENT.server.domain.clinical.entities import ClinicalSessionRequest
from DILIGENT.server.domain.jobs import JobCancelResponse, JobStartResponse, JobStatusResponse
from DILIGENT.server.services import session_service as session_service_module
from DILIGENT.server.services.session_service import (
    ClinicalJobCancelled,
    NarrativeBuilder,
    StageProgressFractionCallback,
    build_failed_session_payload,
    execute_clinical_job,
    run_clinical_job,
)

job_manager = session_service_module.job_manager
input_preparator = session_service_module.input_preparator
HepatoxConsultation = session_service_module.HepatoxConsultation


class ClinicalSessionEndpoint(session_service_module.ClinicalSessionEndpoint):
    def _sync_compatibility_seams(self) -> None:
        session_service_module.job_manager = job_manager
        session_service_module.input_preparator = input_preparator
        session_service_module.HepatoxConsultation = HepatoxConsultation

    async def process_single_patient(
        self,
        payload,
        *,
        progress_callback=None,
        stop_check=None,
    ):
        self._sync_compatibility_seams()
        return await super().process_single_patient(
            payload,
            progress_callback=progress_callback,
            stop_check=stop_check,
        )

    async def start_clinical_session(
        self,
        request_payload: ClinicalSessionRequest = Body(...),
    ) -> PlainTextResponse:
        self._sync_compatibility_seams()
        return await super().start_clinical_session(request_payload)

    def start_clinical_job(
        self,
        request_payload: ClinicalSessionRequest = Body(...),
    ) -> JobStartResponse:
        self._sync_compatibility_seams()
        return super().start_clinical_job(request_payload)

    def get_clinical_job_status(self, job_id: str) -> JobStatusResponse:
        self._sync_compatibility_seams()
        return super().get_clinical_job_status(job_id)

    def cancel_clinical_job(self, job_id: str) -> JobCancelResponse:
        self._sync_compatibility_seams()
        return super().cancel_clinical_job(job_id)


router = APIRouter(tags=["session"])
endpoint = ClinicalSessionEndpoint(
    drugs_parser=session_service_module.drugs_parser,
    disease_extractor=session_service_module.disease_extractor,
    lab_extractor=session_service_module.lab_extractor,
    pattern_analyzer=session_service_module.pattern_analyzer,
    rucam_estimator=session_service_module.rucam_estimator,
    serializer=session_service_module.serializer,
    payload_sanitizer=session_service_module.payload_sanitization_service,
    router=router,
)


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
    session_service_module.job_manager = job_manager
    if message is None:
        session_service_module.report_clinical_job_progress(
            job_id,
            stage=stage,
            progress=progress,
        )
        return
    session_service_module.report_clinical_job_progress(
        job_id,
        stage=stage,
        progress=progress,
    )


router.add_api_route(
    "/clinical",
    start_clinical_session,
    methods=["POST"],
    response_model=None,
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
    "HepatoxConsultation",
    "input_preparator",
    "job_manager",
    "report_clinical_job_progress",
    "router",
    "run_clinical_job",
    "StageProgressFractionCallback",
    "start_clinical_job",
    "start_clinical_session",
]
