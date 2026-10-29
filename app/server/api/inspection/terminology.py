# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Body, HTTPException, Query, status

from api.inspection.common import InspectionJobEndpointMixin
from domain.inspection import (
    InspectionTerminologyOverrideRequest,
    InspectionUpdateConfigResponse,
)
from domain.jobs import JobCancelResponse, JobStartResponse, JobStatusResponse
from domain.terminology import (
    TerminologyAliasDecisionRequest,
    TerminologyAliasDecisionResponse,
    TerminologyConflictsResponse,
    TerminologyImportRunItem,
    TerminologyImportRunsResponse,
    TerminologyResolution,
    TerminologyResolveRequest,
    TerminologySummaryResponse,
)
from services.inspection.service import DataInspectionService

###############################################################################
class InspectionTerminologyEndpoint(InspectionJobEndpointMixin):

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        router: APIRouter,
        service: DataInspectionService,
    ) -> None:
        super().__init__(router=router, service=service)

    # -------------------------------------------------------------------------
    def get_summary(self) -> TerminologySummaryResponse:
        return TerminologySummaryResponse(**self.service.get_terminology_summary())

    # -------------------------------------------------------------------------
    def list_runs(
        self,
        limit: int = Query(default=20, ge=1, le=200),
    ) -> TerminologyImportRunsResponse:
        return TerminologyImportRunsResponse(
            items=[TerminologyImportRunItem(**item) for item in self.service.list_terminology_runs(limit=limit)]
        )

    # -------------------------------------------------------------------------
    def list_conflicts(
        self,
        conflict_status: Literal["open", "resolved", "dismissed"] | None = Query(default="open", alias="status"),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=20, ge=1, le=100),
    ) -> TerminologyConflictsResponse:
        return TerminologyConflictsResponse(
            **self.service.list_terminology_conflicts(status=conflict_status, offset=offset, limit=limit)
        )

    # -------------------------------------------------------------------------
    def resolve_name(self, request: TerminologyResolveRequest) -> TerminologyResolution:
        return TerminologyResolution(
            **self.service.resolve_terminology_name(request.name, allow_external=request.allow_external)
        )

    # -------------------------------------------------------------------------
    def _decide(
        self,
        request: TerminologyAliasDecisionRequest,
        decision: Literal["confirmed", "rejected"],
    ) -> TerminologyAliasDecisionResponse:
        payload = self.service.record_terminology_alias_decision(
            alias=request.alias, drug_id=request.drug_id, decision=decision
        )
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Drug concept not found.",
            )
        return TerminologyAliasDecisionResponse(**payload)

    # -------------------------------------------------------------------------
    def confirm_alias(self, request: TerminologyAliasDecisionRequest) -> TerminologyAliasDecisionResponse:
        return self._decide(request, "confirmed")

    # -------------------------------------------------------------------------
    def reject_alias(self, request: TerminologyAliasDecisionRequest) -> TerminologyAliasDecisionResponse:
        return self._decide(request, "rejected")

    # -------------------------------------------------------------------------
    def get_update_config(self) -> InspectionUpdateConfigResponse:
        return InspectionUpdateConfigResponse(
            **self.service.build_update_config_response("terminology")
        )

    # -------------------------------------------------------------------------
    def start_terminology_update_job(
        self,
        overrides: InspectionTerminologyOverrideRequest | None = Body(default=None),
    ) -> JobStartResponse:
        overrides = overrides or InspectionTerminologyOverrideRequest()
        return self.start_update_job(
            job_type=self.service.TERMINOLOGY_JOB_TYPE,
            message="Terminology update job started",
            overrides=overrides.model_dump(exclude_none=True),
        )

    # -------------------------------------------------------------------------
    def get_terminology_update_job_status(self, job_id: str) -> JobStatusResponse:
        return self.get_update_job_status(job_id=job_id, job_type=self.service.TERMINOLOGY_JOB_TYPE)

    # -------------------------------------------------------------------------
    def cancel_terminology_update_job(self, job_id: str) -> JobCancelResponse:
        return self.cancel_update_job(job_id=job_id, job_type=self.service.TERMINOLOGY_JOB_TYPE)

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        routes = [
            ("/terminology/summary", self.get_summary, "GET", TerminologySummaryResponse, status.HTTP_200_OK),
            ("/terminology/runs", self.list_runs, "GET", TerminologyImportRunsResponse, status.HTTP_200_OK),
            ("/terminology/conflicts", self.list_conflicts, "GET", TerminologyConflictsResponse, status.HTTP_200_OK),
            ("/terminology/resolve", self.resolve_name, "POST", TerminologyResolution, status.HTTP_200_OK),
            (
                "/terminology/aliases/confirm",
                self.confirm_alias,
                "POST",
                TerminologyAliasDecisionResponse,
                status.HTTP_200_OK,
            ),
            (
                "/terminology/aliases/reject",
                self.reject_alias,
                "POST",
                TerminologyAliasDecisionResponse,
                status.HTTP_200_OK,
            ),
            (
                "/terminology/update-config",
                self.get_update_config,
                "GET",
                InspectionUpdateConfigResponse,
                status.HTTP_200_OK,
            ),
            ("/terminology/jobs", self.start_terminology_update_job, "POST", JobStartResponse, status.HTTP_202_ACCEPTED),
            (
                "/terminology/jobs/{job_id}",
                self.get_terminology_update_job_status,
                "GET",
                JobStatusResponse,
                status.HTTP_200_OK,
            ),
            (
                "/terminology/jobs/{job_id}",
                self.cancel_terminology_update_job,
                "DELETE",
                JobCancelResponse,
                status.HTTP_200_OK,
            ),
        ]
        for path, handler, method, response_model, status_code in routes:
            self.router.add_api_route(
                path,
                handler,
                methods=[method],
                response_model=response_model,
                status_code=status_code,
            )
