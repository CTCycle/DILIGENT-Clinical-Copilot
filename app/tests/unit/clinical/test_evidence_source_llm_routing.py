# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from common.prompts.clinical_assessment import (
    EXTERNAL_EVIDENCE_CLINICAL_SYSTEM_PROMPT,
    LIVERTOX_CLINICAL_SYSTEM_PROMPT,
)
from domain.clinical.entities import (
    DrugClinicalAssessment,
    create_drug_suspension_context,
)
from services.clinical.analysis_runner import AnalysisRunner
from services.clinical.drug_analysis import DrugAnalysisService
from services.clinical.exposure_timeline import ExposureTimelineService
from services.clinical.rag_support import RagSupportService


###############################################################################
def _runner() -> Any:
    return SimpleNamespace(
        rag_support=SimpleNamespace(
            select_excerpt=lambda excerpts: RagSupportService.select_excerpt(
                SimpleNamespace(max_excerpt_length=1000), excerpts
            )
        )
    )


###############################################################################
@pytest.mark.parametrize(
    ("evidence_status", "missing_livertox", "excerpts", "expected"),
    [
        # LiverTox selected: a matched excerpt remains mandatory.
        ({"livertox": "retrieved"}, False, ["Excerpt."], ("Excerpt.", "matched")),
        ({"livertox": "no_record"}, True, [], None),
        # LiverTox not selected: accepted identities are assessed without it.
        ({"livertox": "not_selected"}, False, [], ("", "not_selected")),
        ({"livertox": "not_selected"}, True, [], None),
    ],
)
def test_llm_evidence_respects_livertox_selection(
    evidence_status: dict[str, str],
    missing_livertox: bool,
    excerpts: list[str],
    expected: tuple[str, str] | None,
) -> None:
    entry = DrugClinicalAssessment(
        drug_name="Acetaminophen",
        missing_livertox=missing_livertox,
        evidence_status=evidence_status,
    )

    assert AnalysisRunner.resolve_llm_evidence(_runner(), entry, excerpts) == expected


###############################################################################
@pytest.mark.parametrize(
    ("livertox_status", "expected_system"),
    [
        ("matched", LIVERTOX_CLINICAL_SYSTEM_PROMPT),
        ("not_selected", EXTERNAL_EVIDENCE_CLINICAL_SYSTEM_PROMPT),
    ],
)
def test_drug_analysis_prompt_follows_livertox_selection(
    livertox_status: str, expected_system: str
) -> None:
    service = DrugAnalysisService(
        llm_client=None,
        llm_model="test-model",
        exposure_timeline=ExposureTimelineService(),
        retry_attempts=1,
    )
    captured: list[list[dict[str, str]]] = []

    async def fake_chat(**kwargs: Any) -> str:
        captured.append(kwargs["messages"])
        return "Assessment narrative in English."

    service._chat = fake_chat  # type: ignore[method-assign]

    asyncio.run(
        service.request_drug_analysis(
            drug_name="Acetaminophen",
            canonical_name="Acetaminophen",
            origins=["therapy"],
            extraction_metadata=[],
            livertox_status=livertox_status,
            excerpt="LiverTox excerpt text." if livertox_status == "matched" else "",
            rag_context=None,
            clinical_context="Patient context.",
            suspension=create_drug_suspension_context(),
            visit_date=None,
            pattern_summary="hepatocellular",
            metadata=None,
            rucam=None,
            knowledge_prompt="",
        )
    )

    system, user = captured[0]
    assert system["content"] == expected_system
    if livertox_status == "not_selected":
        assert "LiverTox excerpt" not in user["content"]
        assert "No external evidence source was consulted" in user["content"]
    else:
        assert "LiverTox excerpt text." in user["content"]
