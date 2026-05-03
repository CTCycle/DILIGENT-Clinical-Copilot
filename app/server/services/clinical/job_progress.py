from __future__ import annotations

from collections.abc import Callable

from services.runtime.jobs import get_job_manager


CLINICAL_PROGRESS_MESSAGES: dict[str, str] = {
    "session_initialization": "Initializing clinical session",
    "hepatotoxicity_pattern": "Calculating hepatotoxicity pattern",
    "therapy_extraction": "Extracting drugs from therapy",
    "anamnesis_extraction": "Extracting drugs from anamnesis",
    "anamnesis_disease_extraction": "Extracting diseases from anamnesis",
    "anamnesis_lab_extraction": "Extracting longitudinal labs from clinical text",
    "rag_query_building": "Building RAG queries",
    "livertox_lookup": "Consulting LiverTox knowledge base",
    "rucam_estimation": "Estimating per-drug RUCAM",
    "llm_analysis": "Running LLM drug-by-drug assessment",
    "report_composition": "Composing final clinical report",
    "finalization": "Finalizing and persisting session",
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

