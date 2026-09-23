from __future__ import annotations

from datetime import UTC, datetime

from repositories.clinical_session_repository import ClinicalSessionRepository
from repositories.context import RepositoryContext

CASES = (
    ("Synthetic Timeline Matrix Qwen 2B", "qwen3.5:2b"),
    ("Synthetic Timeline Matrix Qwen 9B", "qwen3.5:9b"),
    ("Synthetic Timeline Matrix DeepSeek V4.1", "deepseek-v4-flash"),
)
SYNTHETIC_REPORT = "Synthetic report for timeline model matrix."


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
                "final_report": SYNTHETIC_REPORT,
                "detected_drugs": ["acetaminophen"],
                "session_result_payload": {"report": SYNTHETIC_REPORT, "issues": []},
            }
        )
        if session_id is None:
            raise RuntimeError(
                f"Unable to create synthetic timeline case for {model_name}."
            )
        print(f"Synthetic timeline case created for {model_name}: session {session_id}")


if __name__ == "__main__":
    main()
