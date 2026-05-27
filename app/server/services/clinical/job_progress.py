from __future__ import annotations

from collections.abc import Callable

from services.runtime.jobs import get_job_manager

CLINICAL_PROGRESS_MESSAGES: dict[str, str] = {
    "session_initialization": "Step 1/12 - Initializing session context and validating clinical inputs",
    "therapy_extraction": "Step 2/12 - Parsing THERAPY section to extract active treatment lines",
    "anamnesis_extraction": "Step 3/12 - Parsing ANAMNESIS section to identify historical drug exposures",
    "anamnesis_disease_extraction": "Step 4/12 - Parsing ANAMNESIS section to extract comorbidities and risk context",
    "anamnesis_lab_extraction": "Step 5/12 - Parsing LAB ANALYSIS history to reconstruct longitudinal trends",
    "hepatotoxicity_pattern": "Step 6/12 - Computing hepatotoxicity pattern from laboratory trajectory",
    "rag_query_building": "Step 7/12 - Preparing evidence-retrieval query context",
    "livertox_lookup": "Step 8/12 - Cross-checking candidate drugs against LiverTox evidence",
    "rucam_estimation": "Step 9/12 - Estimating per-drug RUCAM scores",
    "llm_analysis": "Step 10/12 - Performing structured LLM causality assessment per candidate drug",
    "report_composition": "Step 11/12 - Drafting integrated clinical assessment and recommendations",
    "finalization": "Step 12/12 - Final consistency checks and session persistence",
}


class ClinicalJobCancelled(Exception):
    pass


class ClinicalJobProgressCallback:
    def __init__(self, *, job_id: str) -> None:
        self.job_id = job_id

    def __call__(self, stage: str, progress: float) -> None:
        report_clinical_job_progress(self.job_id, stage=stage, progress=progress)


class ClinicalConsultationProgressCallback:
    def __init__(
        self,
        *,
        progress_callback: Callable[[str, float], None] | None,
    ) -> None:
        self.progress_callback = progress_callback

    def __call__(self, stage: str, fraction: float) -> None:
        if self.progress_callback is None:
            return
        bounded_fraction = min(1.0, max(0.0, float(fraction)))
        if stage == "llm_analysis":
            self.progress_callback("llm_analysis", 62.0 + (bounded_fraction * 24.0))
        elif stage == "report_composition":
            self.progress_callback(
                "report_composition", 86.0 + (bounded_fraction * 8.0)
            )


class StageProgressFractionCallback:
    def __init__(
        self,
        *,
        progress_callback: Callable[[str, float], None],
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
        self.progress_callback(self.stage, self.lower + (self.span * bounded_fraction))


def ensure_clinical_job_not_cancelled(job_id: str) -> None:
    if get_job_manager().should_stop(job_id):
        raise ClinicalJobCancelled("Clinical job stop requested.")


def report_clinical_job_progress(job_id: str, *, stage: str, progress: float) -> None:
    ensure_clinical_job_not_cancelled(job_id)
    bounded = min(100.0, max(0.0, float(progress)))
    message = CLINICAL_PROGRESS_MESSAGES.get(stage, stage.replace("_", " ").strip())
    jobs = get_job_manager()
    jobs.update_progress(job_id, bounded)
    jobs.update_result(
        job_id,
        {
            "progress_stage": stage,
            "progress_message": message,
        },
    )

