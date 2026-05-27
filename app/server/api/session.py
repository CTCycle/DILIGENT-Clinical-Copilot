from __future__ import annotations

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
from services.catalogs.runtime import get_reference_catalog_snapshot
from services.runtime.jobs import get_job_manager
from services.session.factory import build_clinical_session_service
from services.session.session_request_validation import (
    validate_clinical_session_request,
)
from services.session.session_service import ClinicalSessionService

router = APIRouter(tags=["session"])


###############################################################################
class ClinicalSessionEndpoint:
    def __init__(self, *, router: APIRouter, service: ClinicalSessionService) -> None:
        self.router = router
        self.service = service

    def start_clinical_job(
        self,
        request_payload: ClinicalSessionRequest = Body(...),
    ) -> JobStartResponse:
        validate_clinical_session_request(request_payload)
        return self.service.start_clinical_job(request_payload)

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
        response.headers["Cache-Control"] = "no-store, no-cache, max-age=0, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return self.service.get_clinical_job_status(job_id)

    # -------------------------------------------------------------------------
    def cancel_clinical_job(self, job_id: str) -> JobCancelResponse:
        return self.service.cancel_clinical_job(job_id)

    # -------------------------------------------------------------------------
    def get_clinical_section_template(self) -> ClinicalSectionTemplateResponse:
        snapshot = get_reference_catalog_snapshot()
        section_keys = list(
            snapshot.values("clinical_extraction", "clinical_sections", key="default")
        ) or ["anamnesis", "drugs", "laboratory_analysis"]
        anamnesis_aliases = list(
            snapshot.values("clinical_extraction", "section_aliases", key="anamnesis")
        )
        therapy_aliases = list(
            snapshot.values("clinical_extraction", "section_aliases", key="drugs")
        )
        labs_aliases = list(
            snapshot.values(
                "clinical_extraction", "section_aliases", key="laboratory_analysis"
            )
        )
        sections = {
            "anamnesis": anamnesis_aliases,
            "therapy": therapy_aliases,
            "laboratory_history": labs_aliases,
        }
        heading_map = {
            "anamnesis": "Anamnesis",
            "drugs": "Therapy",
            "laboratory_analysis": "Laboratory history",
        }
        ordered_headings = [
            heading_map.get(section_key, section_key.replace("_", " ").title())
            for section_key in section_keys
        ]
        template = "\n".join(
            [
                f"Provide plain text with {len(ordered_headings)} clearly titled and separated sections.",
                "",
                f"## {ordered_headings[0] if len(ordered_headings) > 0 else 'Anamnesis'}",
                "Diagnoses, symptoms, clinical conditions, disease timeline, previous therapies, relevant history.",
                "",
                f"## {ordered_headings[1] if len(ordered_headings) > 1 else 'Therapy'}",
                "Current and past drugs, dosage, route, start date or timing, suspension status. Bullet lists allowed.",
                "",
                f"## {ordered_headings[2] if len(ordered_headings) > 2 else 'Laboratory history'}",
                "ALT, ALP, bilirubin, GGT, AST, INR, dates, units, ULN values, hepatic pattern (if known), RUCAM score (if already calculated).",
                "",
                (
                    "Equivalent headings are accepted. "
                    f"Anamnesis aliases: {', '.join(anamnesis_aliases[:5]) or 'n/a'}. "
                    f"Therapy aliases: {', '.join(therapy_aliases[:5]) or 'n/a'}. "
                    f"Laboratory aliases: {', '.join(labs_aliases[:5]) or 'n/a'}."
                ),
            ]
        )
        return ClinicalSectionTemplateResponse(headings=sections, template=template)

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/clinical/section-template",
            self.get_clinical_section_template,
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
    service=build_clinical_session_service(get_job_manager()),
).add_routes()
