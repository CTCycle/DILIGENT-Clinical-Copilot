from __future__ import annotations

import pytest
from pydantic import ValidationError

from common.prompts.clinical_assessment import (
    EXTERNAL_EVIDENCE_CLINICAL_SYSTEM_PROMPT,
    EXTERNAL_EVIDENCE_CLINICAL_USER_PROMPT,
    EXTERNAL_EVIDENCE_REVISION_SYSTEM_PROMPT,
    LIVERTOX_CLINICAL_SYSTEM_PROMPT,
    LIVERTOX_CLINICAL_USER_PROMPT,
    LIVERTOX_CONCLUSION_SYSTEM_PROMPT,
    LIVERTOX_REVISION_CLINICAL_SYSTEM_PROMPT,
    LIVERTOX_REVISION_CLINICAL_USER_PROMPT,
    LIVERTOX_REVISION_CONCLUSION_SYSTEM_PROMPT,
)
from common.prompts.report_template import (
    apply_template_to_conclusion_system_prompt,
    apply_template_to_drug_system_prompt,
    apply_template_to_drug_user_prompt,
)
from domain.agents import ReportTemplate

TEMPLATE = "# Report\n## Key findings {x}\n- bullet"

DRUG_SYSTEMS = (
    LIVERTOX_CLINICAL_SYSTEM_PROMPT,
    LIVERTOX_REVISION_CLINICAL_SYSTEM_PROMPT,
    EXTERNAL_EVIDENCE_CLINICAL_SYSTEM_PROMPT,
    EXTERNAL_EVIDENCE_REVISION_SYSTEM_PROMPT,
)
CONFLICTING_DEFAULTS = (
    "No wrapper headings",
    "Return only the narrative clinical assessment body",
    "Keep reasoning concise",
)
SAFETY_MARKERS = (
    "Hy's Law",
    "rechallenge",
    "Never state that competing causes were excluded",
    "Do not create bibliography entries",
)


###############################################################################
@pytest.mark.parametrize("system", DRUG_SYSTEMS)
def test_template_replaces_default_drug_format_rules_and_keeps_safety_rules(
    system: str,
) -> None:
    assert all(marker in system for marker in CONFLICTING_DEFAULTS[:2])
    result = apply_template_to_drug_system_prompt(
        system, template_format="md", content=TEMPLATE
    )
    assert not any(marker in result for marker in CONFLICTING_DEFAULTS)
    assert all(marker in result for marker in SAFETY_MARKERS)
    assert result.count("<report_template") == 1
    assert TEMPLATE in result


###############################################################################
@pytest.mark.parametrize(
    "user", (LIVERTOX_CLINICAL_USER_PROMPT, LIVERTOX_REVISION_CLINICAL_USER_PROMPT, EXTERNAL_EVIDENCE_CLINICAL_USER_PROMPT)
)
def test_template_drops_user_prompt_format_caps_but_keeps_evidence_rules(
    user: str,
) -> None:
    assert "<=500 words" in user
    result = apply_template_to_drug_user_prompt(user)
    assert "<=500 words" not in result
    assert "fenced code" not in result
    assert "Return narrative clinical reasoning only" not in result
    assert "Do not invent data." in result
    assert "Hy's Law" in result


###############################################################################
@pytest.mark.parametrize(
    "system",
    (LIVERTOX_CONCLUSION_SYSTEM_PROMPT, LIVERTOX_REVISION_CONCLUSION_SYSTEM_PROMPT),
)
def test_template_replaces_conclusion_word_cap(system: str) -> None:
    result = apply_template_to_conclusion_system_prompt(
        system, template_format="txt", content=TEMPLATE
    )
    assert "<=500 words" not in result
    assert "rechallenge" in result and "Hy's Law" in result
    assert TEMPLATE in result


###############################################################################
def test_report_template_validation() -> None:
    assert ReportTemplate(file_name="a.md", format="md", content="x").format == "md"
    for bad in (
        {"file_name": "a.md", "format": "md", "content": "   "},
        {"file_name": "a.pdf", "format": "pdf", "content": "x"},
        {"file_name": "a.md", "format": "md", "content": "x" * 20001},
    ):
        with pytest.raises(ValidationError):
            ReportTemplate(**bad)
