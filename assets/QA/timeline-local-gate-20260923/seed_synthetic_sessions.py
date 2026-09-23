from datetime import UTC, datetime

from repositories.clinical_session_repository import ClinicalSessionRepository
from repositories.context import RepositoryContext

CASES = (
    ("Synthetic Timeline Local 2B", "qwen3.5:2b"),
    ("Synthetic Timeline Local 9B", "qwen3.5:9b"),
)
REPORT = "Synthetic report for the local timeline validation."


def main() -> None:
    repository = ClinicalSessionRepository(RepositoryContext.create())
    for patient_name, model_name in CASES:
        session_id = repository.save_clinical_session(
            {
                "patient_name": patient_name,
                "session_timestamp": datetime(2025, 1, 17, 9, 0, tzinfo=UTC),
                "session_status": "successful",
                "anamnesis": "Symptoms started on 2025-01-17.",
                "drugs": "Acetaminophen was taken in 2025-01.",
                "laboratory_analysis": "ALT was 75 U/L on 2025-01-17.",
                "final_report": REPORT,
                "detected_drugs": ["acetaminophen"],
                "session_result_payload": {"report": REPORT, "issues": []},
            }
        )
        if session_id is None:
            raise RuntimeError(f"Unable to create synthetic case for {model_name}.")
        print(f"Created {patient_name} for {model_name}: session {session_id}")


if __name__ == "__main__":
    main()
