from __future__ import annotations

from collections.abc import Callable

CLINICAL_PROGRESS_MESSAGES: dict[str, str] = {
    "preflight.validated": "Step 1/15: Validating required local data and visit metadata...",
    "sections.loaded": "Step 2/15: Loading parsed ANAMNESIS, DRUGS, and LABORATORY ANALYSIS sections...",
    "assessment.bundle": "Step 3/15: Building the structured assessment bundle...",
    "drugs.extracting": "Step 4/15: Parsing DRUGS and ANAMNESIS sections for current and historical medication exposures...",
    "drugs.resolving": "Step 5/15: Resolving extracted drug names against local catalogs...",
    "diseases.extracting": "Step 6/15: Extracting disease and competing-cause context...",
    "labs.extracting": "Step 7/15: Extracting laboratory timeline and onset dates...",
    "pattern.assessing": "Step 8/15: Calculating biochemical liver injury pattern...",
    "candidates.selecting": "Step 9/15: Selecting temporally relevant suspect drug candidates...",
    "rucam.initial": "Step 10/15: Estimating preliminary RUCAM scores...",
    "retrieval.query": "Step 11/15: Building retrieval query from structured case facts...",
    "retrieval.evidence": "Step 12/15: Retrieving LiverTox evidence...",
    "rucam.refined": "Step 13/15: Re-estimating RUCAM scores with retrieved evidence...",
    "report.generating": "Step 14/15: Generating clinical consultation report...",
    "session.saving": "Step 15/15: Auditing artifacts and saving session results...",
    "completed": "Clinical analysis completed.",
    "session_initialization": "Step 1/15: Validating required local data and visit metadata...",
    "therapy_extraction": "Step 4/15: Parsing DRUGS and ANAMNESIS sections for current and historical medication exposures...",
    "anamnesis_extraction": "Step 4/15: Parsing DRUGS and ANAMNESIS sections for current and historical medication exposures...",
    "anamnesis_disease_extraction": "Step 6/15: Extracting disease and competing-cause context...",
    "anamnesis_lab_extraction": "Step 7/15: Extracting laboratory timeline and onset dates...",
    "hepatotoxicity_pattern": "Step 8/15: Calculating biochemical liver injury pattern...",
    "rucam_estimation": "Step 10/15: Estimating preliminary RUCAM scores...",
    "rag_query_building": "Step 11/15: Building retrieval query from structured case facts...",
    "livertox_lookup": "Step 12/15: Retrieving LiverTox evidence...",
    "livertox_lookup.rag": "Step 12/15: Retrieving LiverTox and vector evidence...",
    "livertox_lookup.no_rag": "Step 12/15: Retrieving LiverTox evidence (vector retrieval disabled)...",
    "report_composition": "Step 14/15: Generating clinical consultation report...",
    "finalization": "Step 15/15: Auditing artifacts and saving session results...",
}

###############################################################################
class ClinicalJobCancelled(Exception):
    pass

###############################################################################
class ClinicalConsultationProgressCallback:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        progress_callback: Callable[..., None] | None,
    ) -> None:
        self.progress_callback = progress_callback

    # -------------------------------------------------------------------------
    def __call__(self, stage: str, fraction: float) -> None:
        if self.progress_callback is None:
            return
        bounded_fraction = min(1.0, max(0.0, float(fraction)))
        if stage == "llm_analysis":
            self.progress_callback(
                "report.generating", 88.0 + (bounded_fraction * 6.0), None
            )
        elif stage == "report_composition":
            self.progress_callback(
                "report.generating", 94.0 + (bounded_fraction * 5.0), None
            )

###############################################################################
class StageProgressFractionCallback:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        progress_callback: Callable[..., None],
        stage: str,
        start_value: float,
        end_value: float,
    ) -> None:
        self.progress_callback = progress_callback
        self.stage = stage
        self.lower = min(start_value, end_value)
        self.span = max(0.0, end_value - self.lower)

    # -------------------------------------------------------------------------
    def __call__(self, fraction: float) -> None:
        bounded_fraction = min(1.0, max(0.0, float(fraction)))
        self.progress_callback(
            self.stage, self.lower + (self.span * bounded_fraction), None
        )
