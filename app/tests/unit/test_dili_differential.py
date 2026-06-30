from __future__ import annotations

from services.clinical.dili_differential import DiliDifferentialEngine


def _status(source_text: str, cause: str) -> str:
    assessment = DiliDifferentialEngine().assess(source_text)
    matches = [item for item in assessment.causes if item.cause == cause]
    assert matches
    return matches[0].status


def test_missing_competing_cause_evidence_is_not_excluded() -> None:
    assert _status("No relevant history documented.", "ebv_cmv_hsv") == "missing_data"


def test_cause_specific_negative_evidence_excludes_only_that_cause() -> None:
    text = "HAV negative. Autoimmune hepatitis possible; IgG pending."

    assert _status(text, "viral_hepatitis_a_b_c_d_e") == "excluded"
    assert _status(text, "autoimmune_hepatitis") == "unknown"


def test_present_competing_cause_remains_not_excluded() -> None:
    text = "Ultrasound documented biliary obstruction with gallstones."

    assert _status(text, "biliary_obstruction_gallstones") == "not_excluded"


def test_viral_hepatitis_text_does_not_exclude_chronic_liver_disease() -> None:
    text = "HAV negative. HBV negative. HCV negative."

    assert _status(text, "viral_hepatitis_a_b_c_d_e") == "excluded"
    assert _status(text, "pre_existing_chronic_liver_disease") == "missing_data"
