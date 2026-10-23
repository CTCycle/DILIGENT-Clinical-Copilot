# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import pytest
from services.llm.structured import parse_json_object_strict
from services.session.clinical_section_parsers import SectionHeadingMatch, resolve_heading_collisions_with_diagnostics
from services.session.text_section_parser import parse_initial_text_sections
import asyncio
from services.session.preflight import LocalModelBatchPreflightResult, check_parser_batch_capacity

def test_parse_json_object_strict_valid_object() -> None:
    parsed = parse_json_object_strict('{"a": 1, "b": "x"}')
    assert parsed == {"a": 1, "b": "x"}

def test_parse_json_object_strict_fenced_json() -> None:
    parsed = parse_json_object_strict('```json\n{"a": 1}\n```')
    assert parsed == {"a": 1}

def test_parse_json_object_strict_rejects_leading_prose() -> None:
    with pytest.raises(
        ValueError, match="invalid_json_object|leading_prose_not_allowed"
    ):
        parse_json_object_strict('hello {"a":1}')

def test_parse_json_object_strict_rejects_trailing_prose() -> None:
    with pytest.raises(ValueError, match="trailing_prose_not_allowed"):
        parse_json_object_strict('{"a":1} trailing')

def test_parse_json_object_strict_rejects_multiple_objects() -> None:
    with pytest.raises(ValueError, match="trailing_prose_not_allowed"):
        parse_json_object_strict('{"a":1}{"b":2}')

def test_parses_required_sections_with_line_ranges() -> None:
    text = "ANAMNESIS\nhistory text\nDRUGS:\ndrug row\nLABORATORY ANALYSIS\nlab row\n"
    result = parse_initial_text_sections(text)
    assert result.missing_required_sections == []
    assert result.malformed_sections == []
    assert result.sections["anamnesis"].start_line == 1
    assert result.sections["drugs"].start_line == 3
    assert result.sections["laboratory_analysis"].start_line == 5

def test_rejects_missing_anamnesis() -> None:
    text = "DRUGS\nx\nLABORATORY ANALYSIS\ny\n"
    result = parse_initial_text_sections(text)
    assert "anamnesis" in result.missing_required_sections

def test_rejects_empty_required_section() -> None:
    text = "ANAMNESIS\na\nDRUGS\n\nLABORATORY ANALYSIS\nx\n"
    result = parse_initial_text_sections(text)
    assert "empty:drugs" in result.malformed_sections

def test_rejects_duplicate_section_heading() -> None:
    text = "ANAMNESIS\na\nANAMNESIS\nb\nDRUGS\nd\nLABORATORY ANALYSIS\nl\n"
    result = parse_initial_text_sections(text)
    assert "duplicate:anamnesis" in result.malformed_sections

def test_reports_ambiguous_heading_collisions() -> None:
    resolved, diagnostics = resolve_heading_collisions_with_diagnostics(
        [
            SectionHeadingMatch(
                canonical_key="anamnesis",
                raw_heading="Clinical Therapy",
                normalized_heading="clinical therapy",
                score=0.78,
                strategy="semantic_tokens",
                line_start=4,
                line_end=4,
            ),
            SectionHeadingMatch(
                canonical_key="therapy",
                raw_heading="Clinical Therapy",
                normalized_heading="clinical therapy",
                score=0.77,
                strategy="semantic_tokens",
                line_start=4,
                line_end=4,
            ),
        ]
    )
    assert resolved == []
    assert diagnostics == ["ambiguous_heading:4:clinical therapy"]

def test_parses_mixed_language_therapy_heading_from_live_preflight_path() -> None:
    text = (
        "## Anamnesis\n"
        "History text.\n\n"
        "## Terapia farmacologica\n"
        "Fortecortin 4 mg cpr 1-0-0-0\n"
        "De-Ursil 150 mg caps 1-0-1-0 per os\n\n"
        "## Laboratory Analysis\n"
        "ALT 730 U/L, AST 385 U/L, Bil tot 51.6 umol/L.\n"
    )
    result = parse_initial_text_sections(text)
    assert result.missing_required_sections == []
    assert result.malformed_sections == []
    assert "Fortecortin" in result.sections["drugs"].text

class _FakeClient:

    # -------------------------------------------------------------------------
    def __init__(self, *, online: bool = True, models: list[str] | None = None) -> None:
        self._online = online
        self._models = models or []

    # -------------------------------------------------------------------------
    async def is_server_online(self) -> bool:
        return self._online

    # -------------------------------------------------------------------------
    async def list_models(self) -> list[str]:
        return self._models

    # -------------------------------------------------------------------------
    async def get_cached_residency_plan(self, *, force_refresh: bool = False):
        return {"ok": True, "force_refresh": force_refresh}

    # -------------------------------------------------------------------------
    async def close(self) -> None:
        return None

def test_batch_preflight_allows_non_local_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        "services.session.preflight.LLMRuntimeConfig.resolve_provider_and_model",
        lambda purpose: ("openai", "gpt-4.1-mini"),
    )
    result = asyncio.run(check_parser_batch_capacity(task_count=2))
    assert isinstance(result, LocalModelBatchPreflightResult)
    assert result.concurrency_allowed is True
    assert result.provider == "openai"

def test_batch_preflight_denies_unavailable_local_model(monkeypatch) -> None:
    monkeypatch.setattr(
        "services.session.preflight.LLMRuntimeConfig.resolve_provider_and_model",
        lambda purpose: ("ollama", "qwen3:14b"),
    )
    monkeypatch.setattr(
        "services.session.preflight.select_llm_provider",
        lambda **kwargs: _FakeClient(online=True, models=["llama3.1:8b"]),
    )
    result = asyncio.run(check_parser_batch_capacity(task_count=2))
    assert result.concurrency_allowed is False
    assert result.reason is not None

def test_batch_preflight_allows_available_local_model(monkeypatch) -> None:
    monkeypatch.setattr(
        "services.session.preflight.LLMRuntimeConfig.resolve_provider_and_model",
        lambda purpose: ("ollama", "qwen3:14b"),
    )
    monkeypatch.setattr(
        "services.session.preflight.select_llm_provider",
        lambda **kwargs: _FakeClient(online=True, models=["qwen3:14b"]),
    )
    result = asyncio.run(check_parser_batch_capacity(task_count=2))
    assert result.concurrency_allowed is True
