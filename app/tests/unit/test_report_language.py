from __future__ import annotations

import pytest
from domain.clinical.entities import DrugRucamAssessment
from services.clinical.report_language import (
    SUPPORTED_REPORT_LANGUAGE_CODES,
    phrase,
    resolve_report_language,
    rucam_summary_text,
)


def test_required_phrase_keys_exist_for_supported_languages() -> None:
    keys = {
        'rucam_source_reported',
        'rucam_structured_score',
        'rucam_not_calculated',
        'rucam_insufficient_data',
        'rucam_score_source',
        'rucam_causality_category',
        'rucam_limitations',
        'livertox_missing',
        'livertox_ambiguous',
        'unresolved_mentions',
        'evidence_quality',
        'report_section_summary',
        'report_section_per_drug',
    }
    for lang in SUPPORTED_REPORT_LANGUAGE_CODES:
        for key in keys:
            if key == "rucam_structured_score":
                assert phrase(key, lang, score=6, category="probable")
            else:
                assert phrase(key, lang)


def test_rucam_summary_text_returns_localized_or_safe_text() -> None:
    assessment = DrugRucamAssessment(drug_name='A', total_score=6, causality_category='probable')
    for lang in ('it', 'es', 'fr', 'de', 'pt'):
        text = rucam_summary_text(assessment, lang)
        assert '6' in text


def test_unsupported_language_code_resolves_to_english() -> None:
    assert resolve_report_language('xx') == 'en'


def test_missing_phrase_key_raises_deterministic_error() -> None:
    with pytest.raises(KeyError):
        phrase('missing_key', 'en')

