from __future__ import annotations

from datetime import UTC, datetime

from repositories.clinical_session_repository import ClinicalSessionRepository
from repositories.context import RepositoryContext


def main() -> None:
    repository = ClinicalSessionRepository(RepositoryContext.create())
    report = "Synthetic report for timeline recovery validation."
    session_id = repository.save_clinical_session(
        {
            "patient_name": "Synthetic Timeline Recovery QA",
            "session_timestamp": datetime(2025, 1, 17, 9, 0, tzinfo=UTC),
            "session_status": "successful",
            "anamnesis": "Symptoms began on 2025-01-17.",
            "drugs": "Acetaminophen was taken in 2025-01.",
            "laboratory_analysis": "ALT was 75 U/L on 2025-01-17.",
            "final_report": report,
            "detected_drugs": ["acetaminophen"],
            "session_result_payload": {"report": report, "issues": []},
        }
    )
    if session_id is None:
        raise RuntimeError("Unable to create the synthetic timeline recovery session.")
    print(f"Synthetic timeline recovery session created: {session_id}")


if __name__ == "__main__":
    main()
