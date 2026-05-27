from __future__ import annotations

from collections.abc import Callable

from services.runtime.jobs import get_job_manager

ClinicalProgressEvent = str

CLINICAL_PROGRESS_MESSAGES: dict[str, str] = {
    "preflight.validated": "Step 1/16: Validating required local data and visit metadata...",
    "sections.loaded": "Step 2/16: Loading parsed ANAMNESIS, DRUGS, and LABORATORY ANALYSIS sections...",
    "assessment.bundle": "Step 3/16: Building the structured assessment bundle...",
    "therapy.extracting": "Step 4/16: Parsing DRUGS section for current medication exposures...",
    "anamnesis.extracting": "Step 5/16: Parsing ANAMNESIS section for historical drug exposures...",
    "drugs.resolving": "Step 6/16: Resolving extracted drug names against local catalogs...",
    "diseases.extracting": "Step 7/16: Extracting disease and competing-cause context...",
    "labs.extracting": "Step 8/16: Extracting laboratory timeline and onset dates...",
    "pattern.assessing": "Step 9/16: Calculating biochemical liver injury pattern...",
    "candidates.selecting": "Step 10/16: Selecting temporally relevant suspect drug candidates...",
    "rucam.initial": "Step 11/16: Estimating preliminary RUCAM scores...",
    "retrieval.query": "Step 12/16: Building retrieval query from structured case facts...",
    "retrieval.evidence": "Step 13/16: Retrieving LiverTox and RAG evidence...",
    "rucam.refined": "Step 14/16: Re-estimating RUCAM scores with retrieved evidence...",
    "report.generating": "Step 15/16: Generating clinical consultation report...",
    "session.saving": "Step 16/16: Auditing artifacts and saving session results...",
    "session_initialization": "Step 1/16: Validating required local data and visit metadata...",
    "therapy_extraction": "Step 4/16: Parsing DRUGS section for current medication exposures...",
    "anamnesis_extraction": "Step 5/16: Parsing ANAMNESIS section for historical drug exposures...",
    "anamnesis_disease_extraction": "Step 7/16: Extracting disease and competing-cause context...",
    "anamnesis_lab_extraction": "Step 8/16: Extracting laboratory timeline and onset dates...",
    "hepatotoxicity_pattern": "Step 9/16: Calculating biochemical liver injury pattern...",
    "rucam_estimation": "Step 11/16: Estimating preliminary RUCAM scores...",
    "rag_query_building": "Step 12/16: Building retrieval query from structured case facts...",
    "livertox_lookup": "Step 13/16: Retrieving LiverTox and RAG evidence...",
    "report_composition": "Step 15/16: Generating clinical consultation report...",
    "finalization": "Step 16/16: Auditing artifacts and saving session results...",
}


class ClinicalJobCancelled(Exception):
    pass


class ClinicalJobProgressCallback:
    def __init__(self, *, job_id: str) -> None:
        self.job_id = job_id

    def __call__(self, stage: str, progress: float, detail: str | None = None) -> None:
        report_clinical_job_progress(self.job_id, stage=stage, progress=progress, detail=detail)


class ClinicalConsultationProgressCallback:
    def __init__(
        self,
        *,
        progress_callback: Callable[[str, float, str | None], None] | None,
    ) -> None:
        self.progress_callback = progress_callback

    def __call__(self, stage: str, fraction: float) -> None:
        if self.progress_callback is None:
            return
        bounded_fraction = min(1.0, max(0.0, float(fraction)))
        if stage == "llm_analysis":
            self.progress_callback("report.generating", 88.0 + (bounded_fraction * 6.0), None)
        elif stage == "report_composition":
            self.progress_callback("report.generating", 94.0 + (bounded_fraction * 5.0), None)


class StageProgressFractionCallback:
    def __init__(
        self,
        *,
        progress_callback: Callable[[str, float, str | None], None],
        stage: str,
        start_value: float,
        end_value: float,
    ) -> None:
        self.progress_callback = progress_callback
        self.stage = stage
        self.lower = min(start_value, end_value)
        self.span = max(0.0, end_value - self.lower)

    def __call__(self, fraction: float) -> None:
        bounded_fraction = min(1.0, max(0.0, float(fraction)))
        self.progress_callback(self.stage, self.lower + (self.span * bounded_fraction), None)


def build_clinical_progress_message(
    stage: str,
    progress: float,
    detail: str | None = None,
) -> str:
    _ = progress
    if detail and detail in CLINICAL_PROGRESS_MESSAGES:
        return CLINICAL_PROGRESS_MESSAGES[detail]
    if stage in CLINICAL_PROGRESS_MESSAGES:
        return CLINICAL_PROGRESS_MESSAGES[stage]
    return stage.replace("_", " ").replace(".", " ").strip()


def ensure_clinical_job_not_cancelled(job_id: str) -> None:
    if get_job_manager().should_stop(job_id):
        raise ClinicalJobCancelled("Clinical job stop requested.")


def report_clinical_job_progress(
    job_id: str,
    *,
    stage: str,
    progress: float,
    detail: str | None = None,
) -> None:
    ensure_clinical_job_not_cancelled(job_id)
    bounded = min(100.0, max(0.0, float(progress)))
    message = build_clinical_progress_message(stage=stage, progress=bounded, detail=detail)
    jobs = get_job_manager()
    jobs.update_progress(job_id, bounded)
    jobs.update_result(
        job_id,
        {
            "progress_stage": stage,
            "progress_message": message,
        },
    )
