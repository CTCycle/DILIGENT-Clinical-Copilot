from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from domain.clinical import PatientData
from services.clinical.labs import ClinicalLabExtractor, LabExtractionPayload
from services.clinical.parser import DrugsParser
from services.runtime.jobs import JobManager
from services.session import factory
from services.session.session_service import ClinicalSessionService

ITALIAN_LAB_TEXT = """
Labor 10.01.2025: ALAT 345 U/L, ALP 1055 U/L, Amilasi P 94 U/L, Lipasi 172 U/L.
Labor 16.01.2025: ALAT 822 U/L, ASAT 344 U/L, ALP 898 U/L, Amilasi P 119 U/L, Lipasi 215 U/L.
Labor 13.02.2025: ALAT 591 U/L, ASAT 205 U/L, ALP 666 U/L, Amilasi P 89 U/L, Lipasi 145 U/L.
"""

ITALIAN_THERAPY_TEXT = """
■Amlodipin axapharm cpr 5 mg 0-0-1-0 per os
■Bilol cpr rivestite 5 mg 1-0-0-0 per os
■Clopidogrel Spirig HC cpr rivestite 75 mg 1-0-0-0 per os
■Aspirin Cardio 100 mg cpr [cpr] 1-0-0-0 per os
■Pantoprazol Helvepharm cpr rivestite 20 mg 1-0-0-0 per os
■Prednison 20 mg cpr [cpr] 2-0-0-0 per os
■Seresta 15 mg cpr [cpr] 0-0-0-0.5 per os
■Venlafaxin ER Sandoz Ret caps 75 mg 1-0-0-0 per os
■Forxiga 10 mg cpr [cpr] 1-0-0-0 per os
■Diovan 80 mg cpr [cpr] 0-0-0-0 per os
■Domperidon axapharm lingual cpr orodisp 10 mg 0-0-0-0 per os
■ Invanz sol iniet [g] 0-0-0-0 i.
"""


class FailingLabClient:
    async def llm_structured_call(self, **kwargs: Any) -> LabExtractionPayload:
        _ = kwargs
        raise RuntimeError("simulated lab extraction failure")


class SlowLabClient:
    async def llm_structured_call(self, **kwargs: Any) -> LabExtractionPayload:
        _ = kwargs
        await asyncio.sleep(1.0)
        return LabExtractionPayload(entries=[], onset_context=None)


def test_clinical_session_factory_uses_configured_parser_timeouts(
    monkeypatch: Any,
) -> None:
    settings = SimpleNamespace(
        runtime=SimpleNamespace(
            parser_llm_timeout=120.0,
            disease_llm_timeout=150.0,
        )
    )
    monkeypatch.setattr(factory, "server_settings", settings)

    service = factory.build_clinical_session_service(JobManager())

    assert service.drugs_parser.timeout_s == 120.0
    assert service.lab_extractor.timeout_s == 120.0
    assert service.disease_extractor.timeout_s == 150.0


def test_italian_therapy_text_parses_multiple_drugs_without_llm() -> None:
    parser = DrugsParser(client=object())

    parsed = asyncio.run(parser.extract_drugs_from_therapy(ITALIAN_THERAPY_TEXT))

    names = {entry.name for entry in parsed.entries}
    assert len(names) >= 10
    assert "Amlodipin axapharm" in names
    assert "Clopidogrel Spirig HC" in names
    assert "Aspirin Cardio" in names


def test_italian_laboratory_text_uses_deterministic_fallback() -> None:
    extractor = ClinicalLabExtractor(client=FailingLabClient())
    payload = PatientData(
        anamnesis="",
        drugs="Invanz sol iniet",
        laboratory_analysis=ITALIAN_LAB_TEXT,
    )

    timeline, _ = asyncio.run(extractor.extract_from_payload(payload))

    observed = {
        (entry.marker_name, entry.sample_date, entry.value) for entry in timeline.entries
    }
    assert ("ALT", "2025-01-10", 345.0) in observed
    assert ("ALP", "2025-01-10", 1055.0) in observed
    assert ("AST", "2025-01-16", 344.0) in observed
    assert ("ALT", "2025-02-13", 591.0) in observed


def test_lab_llm_chunk_timeout_uses_deterministic_fallback(monkeypatch: Any) -> None:
    extractor = ClinicalLabExtractor(client=SlowLabClient(), timeout_s=3600.0)
    monkeypatch.setattr(extractor, "LOCAL_LLM_CHUNK_TIMEOUT_CAP_S", 0.01)
    payload = PatientData(
        anamnesis="",
        drugs="Invanz sol iniet",
        laboratory_analysis="Labor 10.01.2025: ALAT 345 U/L, ALP 1055 U/L.",
    )

    timeline, _ = asyncio.run(extractor.extract_from_payload(payload))

    observed = {
        (entry.marker_name, entry.sample_date, entry.value) for entry in timeline.entries
    }
    assert ("ALT", "2025-01-10", 345.0) in observed
    assert ("ALP", "2025-01-10", 1055.0) in observed


def test_runtime_timeout_resolution_does_not_apply_six_second_parser_cap() -> None:
    timeout = ClinicalSessionService._resolve_runtime_timeout(base_timeout_s=120.0)

    assert timeout > 6.0
