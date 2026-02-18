from __future__ import annotations

import pytest

from DILIGENT.server.entities.clinical import (
    ClinicalPipelineValidationError,
    PatientData,
)
from DILIGENT.server.services.clinical.hepatox import HepatotoxicityPatternAnalyzer


def test_assess_payload_returns_determined_score_when_labs_present() -> None:
    analyzer = HepatotoxicityPatternAnalyzer()
    payload = PatientData(
        drugs="Acetaminophen 500 mg 1-0-0-0",
        alt="100",
        alt_max="50",
        alp="200",
        alp_max="100",
    )

    assessment = analyzer.assess_payload(payload, allow_missing_labs=False)

    assert assessment.status == "ok"
    assert assessment.issues == []
    assert assessment.score.classification == "cholestatic"
    assert assessment.score.r_score == pytest.approx(1.0)


def test_assess_payload_raises_when_labs_missing_and_not_overridden() -> None:
    analyzer = HepatotoxicityPatternAnalyzer()
    payload = PatientData(
        drugs="Acetaminophen 500 mg 1-0-0-0",
        alt=None,
        alt_max="50",
        alp="200",
        alp_max="100",
    )

    with pytest.raises(ClinicalPipelineValidationError) as exc_info:
        analyzer.assess_payload(payload, allow_missing_labs=False)

    assert any(issue.code == "missing_labs" for issue in exc_info.value.issues)


def test_assess_payload_allows_missing_labs_when_overridden() -> None:
    analyzer = HepatotoxicityPatternAnalyzer()
    payload = PatientData(
        drugs="Acetaminophen 500 mg 1-0-0-0",
        alt=None,
        alt_max="50",
        alp="200",
        alp_max="100",
    )

    assessment = analyzer.assess_payload(payload, allow_missing_labs=True)

    assert assessment.status == "undetermined_due_to_missing_labs"
    assert assessment.score.classification == "indeterminate"
    assert any(issue.code == "missing_labs" for issue in assessment.issues)
