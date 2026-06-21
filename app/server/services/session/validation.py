from __future__ import annotations

import re

from domain.clinical.entities import ClinicalSessionRequest

WORD_PATTERN = re.compile(r"\b[\wÀ-ÖØ-öø-ÿ']+\b", re.UNICODE)


###############################################################################
def count_words(text: str) -> int:
    return len(WORD_PATTERN.findall(text or ""))


###############################################################################
def validate_clinical_session_request(
    request: ClinicalSessionRequest,
) -> list[dict[str, str]]:
    details: list[dict[str, str]] = []

    clinical_input = (request.clinical_input or "").strip()
    providers = [
        item.strip()
        for item in request.selected_model_providers
        if item and item.strip()
    ]

    if request.visit_date is None:
        details.append({"field": "visit_date", "message": "Visit date is required."})
    if not clinical_input:
        details.append(
            {"field": "clinical_input", "message": "Clinical input is required."}
        )
    elif count_words(clinical_input) < 60:
        details.append(
            {
                "field": "clinical_input",
                "message": "Clinical input must contain at least 60 words.",
            }
        )
    if not providers:
        details.append(
            {
                "field": "selected_model_providers",
                "message": "At least one model provider must be selected.",
            }
        )

    return details
