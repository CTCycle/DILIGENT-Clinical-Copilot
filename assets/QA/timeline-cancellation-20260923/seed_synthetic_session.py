from __future__ import annotations

from datetime import datetime, timezone

from repositories.clinical_session_repository import ClinicalSessionRepository
from repositories.context import RepositoryContext


def main() -> None:
    repository = ClinicalSessionRepository(RepositoryContext.create())
    report = "Synthetic report for timeline cancellation QA."
    session_id = repository.save_clinical_session(
        {
            "patient_name": "Synthetic Timeline Cancellation QA",
            "session_timestamp": datetime(2025, 1, 17, 9, 0, tzinfo=timezone.utc),
            "session_status": "successful",
            "anamnesis": "Synthetic timeline extraction cancellation fixture.",
            "drugs": "acetaminophen",
            "laboratory_analysis": "2025-01-17 ALT 25 U/L; ALP 70 U/L.",
            "final_report": report,
            "detected_drugs": ["acetaminophen"],
            "session_result_payload": {"report": report, "issues": []},
        }
    )
    if session_id is None:
        raise RuntimeError("Unable to seed the synthetic timeline cancellation session.")
    print(f"Synthetic timeline cancellation session created: {session_id}")


if __name__ == "__main__":
    main()
