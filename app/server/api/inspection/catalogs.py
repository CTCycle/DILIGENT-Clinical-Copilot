from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Query, status

from api.inspection.common import InspectionJobEndpointMixin
from domain.inspection import (
    CatalogListFilters,
    DeleteEntityResponse,
    DrugAliasesResponse,
    InspectionLiverToxOverrideRequest,
    InspectionRxNavOverrideRequest,
    InspectionUpdateConfigResponse,
    LiverToxCatalogResponse,
    LiverToxExcerptResponse,
    ReferenceCatalogRuntimeObservationResponse,
    ReferenceCatalogRuntimeObservationUpsertRequest,
    RxNavCatalogResponse,
)
from domain.jobs import JobCancelResponse, JobStartResponse, JobStatusResponse
from services.inspection.service import DataInspectionService


###############################################################################
class InspectionCatalogEndpoint(InspectionJobEndpointMixin):

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        router: APIRouter,
        service: DataInspectionService,
    ) -> None:
        super().__init__(router=router, service=service)

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
    def list_reference_catalog_runtime_observations(
        self,
    ) -> list[ReferenceCatalogRuntimeObservationResponse]:
        return [
            ReferenceCatalogRuntimeObservationResponse(**row)
            for row in self.service.list_reference_catalog_runtime_observations()
        ]

    # -------------------------------------------------------------------------
    def list_reference_catalog_runtime_observations_by_category(
        self, category: str
    ) -> list[ReferenceCatalogRuntimeObservationResponse]:
        return [
            ReferenceCatalogRuntimeObservationResponse(**row)
            for row in self.service.list_reference_catalog_runtime_observations(
                category=category
            )
        ]

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def delete_reference_catalog_runtime_observation(
        self, category: str, term: str
    ) -> DeleteEntityResponse:
        deleted = self.service.deactivate_reference_catalog_runtime_observation(
            category=category,
            term=term,
        )
        return DeleteEntityResponse(deleted=deleted)

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
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
