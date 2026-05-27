from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Body, HTTPException, Query, status

from common.utils.logger import logger
from configurations.startup import server_settings
from domain.inspection import (
    CatalogListFilters,
    DateFilterMode,
    DeleteEntityResponse,
    DrugAliasesResponse,
    InspectionLiverToxOverrideRequest,
    InspectionRagOverrideRequest,
    InspectionRxNavOverrideRequest,
    InspectionUpdateConfigResponse,
    LanceVectorStoreSummaryResponse,
    LiverToxCatalogResponse,
    LiverToxExcerptResponse,
    RagDocumentListResponse,
    ReferenceCatalogRuntimeObservationResponse,
    ReferenceCatalogRuntimeObservationUpsertRequest,
    RxNavCatalogResponse,
    SessionCatalogResponse,
    SessionDetailResponse,
    SessionListFilters,
    SessionRevisionRequest,
    SessionStatus,
    SessionUpdateRequest,
)
from domain.jobs import (
    JobCancelResponse,
    JobStartResponse,
    JobStatusResponse,
)
from domain.patient_timeline import (
    PatientTimeline,
    SessionTimelineRegenerateRequest,
)
from services.inspection.service import DataInspectionService
from services.runtime.jobs import get_job_manager

router = APIRouter(prefix="/inspection", tags=["inspection"])


###############################################################################
class DataInspectionEndpoint:
    def __init__(
        self,
        *,
        router: APIRouter,
        service: DataInspectionService,
    ) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    def build_job_start_response(
        self,
        *,
        payload: dict[str, object],
        message: str,
    ) -> JobStartResponse:
        return JobStartResponse(
            job_id=str(payload["job_id"]),
            job_type=str(payload["job_type"]),
            status=str(payload["status"]),
            message=message,
            poll_interval=server_settings.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def start_update_job(
        self,
        *,
        job_type: str,
        message: str,
        overrides: dict[str, object] | None = None,
    ) -> JobStartResponse:
        try:
            payload = self.service.start_update_job(job_type, overrides=overrides)
        except ValueError as exc:
            detail = str(exc)
            error_status = (
                status.HTTP_409_CONFLICT
                if "already running" in detail
                else status.HTTP_422_UNPROCESSABLE_ENTITY
            )
            logger.warning(
                "Inspection update job rejected type=%s detail=%s", job_type, detail
            )
            safe_detail = (
                "An update job is already running."
                if error_status == status.HTTP_409_CONFLICT
                else "Invalid update request."
            )
            raise HTTPException(status_code=error_status, detail=safe_detail) from exc
        except RuntimeError as exc:
            logger.warning(
                "Inspection update job failed to start type=%s error=%s", job_type, exc
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Update job could not start. Please retry.",
            ) from exc
        return self.build_job_start_response(payload=payload, message=message)

    # -------------------------------------------------------------------------
    def get_update_job_status(self, *, job_id: str, job_type: str) -> JobStatusResponse:
        payload = self.service.get_job_status(job_id, expected_type=job_type)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found.",
            )
        return JobStatusResponse(**payload)

    # -------------------------------------------------------------------------
    def cancel_update_job(self, *, job_id: str, job_type: str) -> JobCancelResponse:
        success = self.service.cancel_job(job_id, expected_type=job_type)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found.",
            )
        return JobCancelResponse(
            job_id=job_id,
            success=True,
            message="Cancellation requested",
        )

    # -------------------------------------------------------------------------
    def list_sessions(
        self,
        search: str | None = Query(default=None),
        status_filter: SessionStatus | None = Query(default=None, alias="status"),
        date_mode: DateFilterMode | None = Query(default=None),
        filter_date: date | None = Query(default=None, alias="date"),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=10, ge=1, le=100),
    ) -> SessionCatalogResponse:
        filters = SessionListFilters(
            search=search,
            status=status_filter,
            date_mode=date_mode,
            date=filter_date,
            offset=offset,
            limit=limit,
        )
        payload = self.service.list_sessions(
            search=filters.search,
            status_filter=filters.status,
            date_mode=filters.date_mode,
            filter_date=filters.date,
            offset=filters.offset,
            limit=filters.limit,
        )
        return SessionCatalogResponse(**payload)

    # -------------------------------------------------------------------------
    def get_session_detail(self, session_id: int) -> SessionDetailResponse:
        detail = self.service.get_session_detail(session_id)
        if detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found.",
            )
        return SessionDetailResponse(**detail)

    # -------------------------------------------------------------------------
    def update_session(
        self,
        session_id: int,
        request: SessionUpdateRequest | None = Body(default=None),
    ) -> SessionDetailResponse:
        request = request or SessionUpdateRequest()
        detail = self.service.update_session(
            session_id,
            session_text=request.session_text,
            metadata=request.metadata,
        )
        if detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found.",
            )
        return SessionDetailResponse(**detail)

    # -------------------------------------------------------------------------
    def start_session_revision(
        self,
        session_id: int,
        request: SessionRevisionRequest | None = Body(default=None),
    ) -> JobStartResponse:
        request = request or SessionRevisionRequest()
        try:
            payload = self.service.start_revision_job(
                session_id,
                selected_text=request.selected_text,
                revision_instruction=request.revision_instruction,
                model_overrides=request.model_overrides,
                metadata=request.metadata,
            )
        except ValueError as exc:
            detail = str(exc)
            error_status = (
                status.HTTP_409_CONFLICT
                if "already running" in detail
                else status.HTTP_404_NOT_FOUND
                if "not found" in detail.casefold()
                else status.HTTP_422_UNPROCESSABLE_ENTITY
            )
            raise HTTPException(status_code=error_status, detail=detail) from exc
        return self.build_job_start_response(
            payload=payload,
            message="Session revision job started",
        )

    # -------------------------------------------------------------------------
    def get_session_revision_status(self, job_id: str) -> JobStatusResponse:
        return self.get_update_job_status(
            job_id=job_id,
            job_type=self.service.REVISION_JOB_TYPE,
        )

    # -------------------------------------------------------------------------
    def cancel_session_revision(self, job_id: str) -> JobCancelResponse:
        return self.cancel_update_job(
            job_id=job_id,
            job_type=self.service.REVISION_JOB_TYPE,
        )

    # -------------------------------------------------------------------------
    def get_session_timeline(self, session_id: int) -> PatientTimeline:
        timeline = self.service.get_session_timeline(session_id)
        if timeline is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session timeline not found.",
            )
        return timeline

    # -------------------------------------------------------------------------
    def generate_session_timeline(
        self,
        session_id: int,
        request: SessionTimelineRegenerateRequest | None = Body(default=None),
        force_regenerate_query: bool = Query(default=False, alias="force_regenerate"),
    ) -> PatientTimeline:
        request = request or SessionTimelineRegenerateRequest()
        force_regenerate = bool(force_regenerate_query or request.force_regenerate)
        try:
            timeline = self.service.generate_session_timeline(
                session_id,
                force_regenerate=force_regenerate,
            )
        except RuntimeError as exc:
            detail_message = str(exc)
            lowered_detail = detail_message.casefold()
            if (
                "cooling down" in lowered_detail
                or "already in progress" in lowered_detail
            ):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=detail_message,
                ) from exc
            if "failed to list ollama models" in lowered_detail:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        "Timeline generation requires a reachable local model runtime. "
                        "Start Ollama and retry."
                    ),
                ) from exc
            logger.warning(
                "Session timeline generation failed session_id=%s error=%s",
                session_id,
                exc,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Timeline generation is currently unavailable. Please retry.",
            ) from exc
        if timeline is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found.",
            )
        return timeline

    # -------------------------------------------------------------------------
    def delete_session(self, session_id: int) -> DeleteEntityResponse:
        deleted = self.service.delete_session(session_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found.",
            )
        return DeleteEntityResponse(deleted=True)

    # -------------------------------------------------------------------------
    def list_rxnav_catalog(
        self,
        search: str | None = Query(default=None),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=10, ge=1, le=100),
    ) -> RxNavCatalogResponse:
        filters = CatalogListFilters(search=search, offset=offset, limit=limit)
        payload = self.service.list_rxnav_catalog(
            search=filters.search,
            offset=filters.offset,
            limit=filters.limit,
        )
        return RxNavCatalogResponse(**payload)

    # -------------------------------------------------------------------------
    def get_rxnav_aliases(self, drug_id: int) -> DrugAliasesResponse:
        payload = self.service.get_rxnav_alias_groups(drug_id)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Drug not found.",
            )
        return DrugAliasesResponse(**payload)

    # -------------------------------------------------------------------------
    def delete_rxnav_drug(self, drug_id: int) -> DeleteEntityResponse:
        deleted = self.service.delete_drug(drug_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Drug not found.",
            )
        return DeleteEntityResponse(deleted=True)

    # -------------------------------------------------------------------------
    def get_rxnav_update_config(self) -> InspectionUpdateConfigResponse:
        payload = self.service.build_update_config_response("rxnav")
        return InspectionUpdateConfigResponse(**payload)

    # -------------------------------------------------------------------------
    def start_rxnav_update_job(
        self,
        overrides: InspectionRxNavOverrideRequest | None = Body(default=None),
    ) -> JobStartResponse:
        overrides = overrides or InspectionRxNavOverrideRequest()
        return self.start_update_job(
            job_type=self.service.RXNAV_JOB_TYPE,
            message="RxNav update job started",
            overrides=overrides.model_dump(exclude_none=True),
        )

    # -------------------------------------------------------------------------
    def get_rxnav_update_job_status(self, job_id: str) -> JobStatusResponse:
        return self.get_update_job_status(
            job_id=job_id,
            job_type=self.service.RXNAV_JOB_TYPE,
        )

    # -------------------------------------------------------------------------
    def cancel_rxnav_update_job(self, job_id: str) -> JobCancelResponse:
        return self.cancel_update_job(
            job_id=job_id,
            job_type=self.service.RXNAV_JOB_TYPE,
        )

    # -------------------------------------------------------------------------
    def list_livertox_catalog(
        self,
        search: str | None = Query(default=None),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=10, ge=1, le=100),
    ) -> LiverToxCatalogResponse:
        filters = CatalogListFilters(search=search, offset=offset, limit=limit)
        payload = self.service.list_livertox_catalog(
            search=filters.search,
            offset=filters.offset,
            limit=filters.limit,
        )
        return LiverToxCatalogResponse(**payload)

    # -------------------------------------------------------------------------
    def get_livertox_excerpt(self, drug_id: int) -> LiverToxExcerptResponse:
        payload = self.service.get_livertox_excerpt(drug_id)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="LiverTox excerpt not found.",
            )
        return LiverToxExcerptResponse(**payload)

    # -------------------------------------------------------------------------
    def delete_livertox_drug(self, drug_id: int) -> DeleteEntityResponse:
        deleted = self.service.delete_drug(drug_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Drug not found.",
            )
        return DeleteEntityResponse(deleted=True)

    # -------------------------------------------------------------------------
    def get_livertox_update_config(self) -> InspectionUpdateConfigResponse:
        payload = self.service.build_update_config_response("livertox")
        return InspectionUpdateConfigResponse(**payload)

    # -------------------------------------------------------------------------
    def start_livertox_update_job(
        self,
        overrides: InspectionLiverToxOverrideRequest | None = Body(default=None),
    ) -> JobStartResponse:
        overrides = overrides or InspectionLiverToxOverrideRequest()
        return self.start_update_job(
            job_type=self.service.LIVERTOX_JOB_TYPE,
            message="LiverTox update job started",
            overrides=overrides.model_dump(exclude_none=True),
        )

    # -------------------------------------------------------------------------
    def get_livertox_update_job_status(self, job_id: str) -> JobStatusResponse:
        return self.get_update_job_status(
            job_id=job_id,
            job_type=self.service.LIVERTOX_JOB_TYPE,
        )

    # -------------------------------------------------------------------------
    def cancel_livertox_update_job(self, job_id: str) -> JobCancelResponse:
        return self.cancel_update_job(
            job_id=job_id,
            job_type=self.service.LIVERTOX_JOB_TYPE,
        )

    # -------------------------------------------------------------------------
    def get_rag_update_config(self) -> InspectionUpdateConfigResponse:
        payload = self.service.build_update_config_response("rag")
        return InspectionUpdateConfigResponse(**payload)

    # -------------------------------------------------------------------------
    def list_rag_documents(
        self,
        search: str | None = Query(default=None),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=10, ge=1, le=100),
    ) -> RagDocumentListResponse:
        filters = CatalogListFilters(search=search, offset=offset, limit=limit)
        return RagDocumentListResponse(
            **self.service.list_rag_documents(
                search=filters.search,
                offset=filters.offset,
                limit=filters.limit,
            )
        )

    # -------------------------------------------------------------------------
    def get_rag_vector_store(self) -> LanceVectorStoreSummaryResponse:
        return LanceVectorStoreSummaryResponse(
            **self.service.get_rag_vector_store_summary()
        )

    # -------------------------------------------------------------------------
    def start_rag_update_job(
        self,
        overrides: InspectionRagOverrideRequest | None = Body(default=None),
    ) -> JobStartResponse:
        overrides = overrides or InspectionRagOverrideRequest()
        return self.start_update_job(
            job_type=self.service.RAG_JOB_TYPE,
            message="RAG embeddings update job started",
            overrides=overrides.model_dump(exclude_none=True),
        )

    # -------------------------------------------------------------------------
    def get_rag_update_job_status(self, job_id: str) -> JobStatusResponse:
        return self.get_update_job_status(
            job_id=job_id, job_type=self.service.RAG_JOB_TYPE
        )

    # -------------------------------------------------------------------------
    def cancel_rag_update_job(self, job_id: str) -> JobCancelResponse:
        return self.cancel_update_job(job_id=job_id, job_type=self.service.RAG_JOB_TYPE)

    def list_reference_catalog_runtime_observations(
        self,
    ) -> list[ReferenceCatalogRuntimeObservationResponse]:
        return [
            ReferenceCatalogRuntimeObservationResponse(**row)
            for row in self.service.list_reference_catalog_runtime_observations()
        ]

    def list_reference_catalog_runtime_observations_by_category(
        self, category: str
    ) -> list[ReferenceCatalogRuntimeObservationResponse]:
        return [
            ReferenceCatalogRuntimeObservationResponse(**row)
            for row in self.service.list_reference_catalog_runtime_observations(
                category=category
            )
        ]

    def upsert_reference_catalog_runtime_observation(
        self,
        category: str,
        request: ReferenceCatalogRuntimeObservationUpsertRequest | None = Body(
            default=None
        ),
    ) -> ReferenceCatalogRuntimeObservationResponse:
        request = request or ReferenceCatalogRuntimeObservationUpsertRequest(term="")
        return ReferenceCatalogRuntimeObservationResponse(
            **self.service.upsert_reference_catalog_runtime_observation(
                category=category,
                term=request.term,
                replacement=request.replacement,
                source=request.source,
                is_active=request.is_active,
            )
        )

    def delete_reference_catalog_runtime_observation(
        self, category: str, term: str
    ) -> DeleteEntityResponse:
        deleted = self.service.deactivate_reference_catalog_runtime_observation(
            category=category, term=term
        )
        return DeleteEntityResponse(deleted=deleted)

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/sessions",
            self.list_sessions,
            methods=["GET"],
            response_model=SessionCatalogResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}",
            self.get_session_detail,
            methods=["GET"],
            response_model=SessionDetailResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}",
            self.update_session,
            methods=["PUT"],
            response_model=SessionDetailResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}/revision/jobs",
            self.start_session_revision,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/sessions/revision/jobs/{job_id}",
            self.get_session_revision_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/revision/jobs/{job_id}",
            self.cancel_session_revision,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}/timeline",
            self.get_session_timeline,
            methods=["GET"],
            response_model=PatientTimeline,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}/timeline",
            self.generate_session_timeline,
            methods=["POST"],
            response_model=PatientTimeline,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/sessions/{session_id}",
            self.delete_session,
            methods=["DELETE"],
            response_model=DeleteEntityResponse,
            status_code=status.HTTP_200_OK,
        )

        self.router.add_api_route(
            "/rxnav",
            self.list_rxnav_catalog,
            methods=["GET"],
            response_model=RxNavCatalogResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/rxnav/{drug_id}/aliases",
            self.get_rxnav_aliases,
            methods=["GET"],
            response_model=DrugAliasesResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/rxnav/{drug_id}",
            self.delete_rxnav_drug,
            methods=["DELETE"],
            response_model=DeleteEntityResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/rxnav/update-config",
            self.get_rxnav_update_config,
            methods=["GET"],
            response_model=InspectionUpdateConfigResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/rxnav/jobs",
            self.start_rxnav_update_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/rxnav/jobs/{job_id}",
            self.get_rxnav_update_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/rxnav/jobs/{job_id}",
            self.cancel_rxnav_update_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )

        self.router.add_api_route(
            "/livertox",
            self.list_livertox_catalog,
            methods=["GET"],
            response_model=LiverToxCatalogResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/livertox/{drug_id}/excerpt",
            self.get_livertox_excerpt,
            methods=["GET"],
            response_model=LiverToxExcerptResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/livertox/{drug_id}",
            self.delete_livertox_drug,
            methods=["DELETE"],
            response_model=DeleteEntityResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/livertox/update-config",
            self.get_livertox_update_config,
            methods=["GET"],
            response_model=InspectionUpdateConfigResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/livertox/jobs",
            self.start_livertox_update_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/livertox/jobs/{job_id}",
            self.get_livertox_update_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/livertox/jobs/{job_id}",
            self.cancel_livertox_update_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/reference-catalogs/runtime-observations",
            self.list_reference_catalog_runtime_observations,
            methods=["GET"],
            response_model=list[ReferenceCatalogRuntimeObservationResponse],
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/reference-catalogs/runtime-observations/{category}",
            self.list_reference_catalog_runtime_observations_by_category,
            methods=["GET"],
            response_model=list[ReferenceCatalogRuntimeObservationResponse],
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/reference-catalogs/runtime-observations/{category}",
            self.upsert_reference_catalog_runtime_observation,
            methods=["PUT"],
            response_model=ReferenceCatalogRuntimeObservationResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/reference-catalogs/runtime-observations/{category}/{term}",
            self.delete_reference_catalog_runtime_observation,
            methods=["DELETE"],
            response_model=DeleteEntityResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/rag/update-config",
            self.get_rag_update_config,
            methods=["GET"],
            response_model=InspectionUpdateConfigResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/rag/documents",
            self.list_rag_documents,
            methods=["GET"],
            response_model=RagDocumentListResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/rag/vector-store",
            self.get_rag_vector_store,
            methods=["GET"],
            response_model=LanceVectorStoreSummaryResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/rag/jobs",
            self.start_rag_update_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/rag/jobs/{job_id}",
            self.get_rag_update_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/rag/jobs/{job_id}/cancel",
            self.cancel_rag_update_job,
            methods=["POST"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )


DataInspectionEndpoint(
    router=router,
    service=DataInspectionService(jobs=get_job_manager()),
).add_routes()

