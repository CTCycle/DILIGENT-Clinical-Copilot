from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any

from DILIGENT.server.common.constants import CONFIGURATIONS_FILE
from DILIGENT.server.common.utils.languages import (
    LANGUAGE_DIACRITIC_HINTS,
    LANGUAGE_FUNCTION_HINTS,
    LANGUAGE_HINTS,
    LANGUAGE_PHRASE_HINTS,
    SUPPORTED_REPORT_LANGUAGES,
    TOKEN_PATTERN,
    resolve_supported_language_code,
)
from DILIGENT.server.domain.clinical.entities import PatientData
from DILIGENT.server.domain.clinical.language import LanguageDetectionResult


class ClinicalLanguageDetector:
    DEFAULT_THRESHOLDS: dict[str, float] = {
        "min_best_score": 2.0,
        "high_confidence_min_score": 8.0,
        "high_confidence_min_margin": 3.0,
        "moderate_confidence_min_score": 4.0,
        "moderate_confidence_min_margin": 1.0,
    }

    @staticmethod
    def tokenize(value: str) -> list[str]:
        return [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(value)]

    @classmethod
    def score_text_by_language(cls, text: str) -> dict[str, float]:
        scores: dict[str, float] = dict.fromkeys(SUPPORTED_REPORT_LANGUAGES, 0.0)
        tokens = cls.tokenize(text)
        if not tokens:
            return scores

        unique_tokens = set(tokens)
        lowered_text = text.casefold()
        length_norm = 1.0 + math.log10(max(10, len(tokens)))

        for lang_code in SUPPORTED_REPORT_LANGUAGES:
            hints = LANGUAGE_HINTS.get(lang_code, set())
            function_hints = LANGUAGE_FUNCTION_HINTS.get(lang_code, set())
            phrase_hints = LANGUAGE_PHRASE_HINTS.get(lang_code, ())
            diacritic_hints = LANGUAGE_DIACRITIC_HINTS.get(lang_code, set())

            exact_hint_hits = sum(1 for token in tokens if token in hints)
            unique_hint_hits = sum(1 for token in unique_tokens if token in hints)
            function_hint_hits = sum(1 for token in tokens if token in function_hints)
            phrase_hits = sum(1 for phrase in phrase_hints if phrase in lowered_text)
            diacritic_hits = sum(
                1
                for token in tokens
                if diacritic_hints and any(char in token for char in diacritic_hints)
            )

            raw_score = (
                (exact_hint_hits * 2.2)
                + (unique_hint_hits * 0.9)
                + (function_hint_hits * 0.35)
                + (phrase_hits * 3.0)
                + (diacritic_hits * 1.5)
            )
            scores[lang_code] = raw_score / length_norm
        return scores

    @classmethod
    @lru_cache(maxsize=1)
    def load_thresholds(cls) -> dict[str, float]:
        thresholds = dict(cls.DEFAULT_THRESHOLDS)
        try:
            with open(CONFIGURATIONS_FILE, encoding="utf-8") as handle:
                payload = json.load(handle)
        except OSError, TypeError, ValueError:
            return thresholds

        config = payload.get("clinical_language_detection")
        if not isinstance(config, dict):
            return thresholds
        for key, default_value in cls.DEFAULT_THRESHOLDS.items():
            thresholds[key] = cls.coerce_non_negative_float(
                config.get(key),
                default=default_value,
            )
        return thresholds

    @staticmethod
    def coerce_non_negative_float(value: Any, *, default: float) -> float:
        if isinstance(value, bool):
            return default
        if isinstance(value, int | float):
            parsed = float(value)
            return parsed if parsed >= 0.0 else default
        if isinstance(value, str):
            try:
                parsed = float(value.strip())
            except ValueError:
                return default
            return parsed if parsed >= 0.0 else default
        return default

    @staticmethod
    def default_result() -> LanguageDetectionResult:
        return LanguageDetectionResult(
            detected_input_language="en",
            report_language="en",
            confidence="low",
        )

    @classmethod
    def detect(cls, payload: PatientData) -> LanguageDetectionResult:
        full_text = " ".join(
            section.strip()
            for section in (
                payload.anamnesis,
                payload.laboratory_analysis,
                payload.drugs,
            )
            if section and section.strip()
        )
        if not full_text:
            return cls.default_result()

        scores = cls.score_text_by_language(full_text)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            return cls.default_result()
        best_language_raw, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - second_score
        thresholds = cls.load_thresholds()

        if best_score < thresholds["min_best_score"]:
            return cls.default_result()
        if (
            best_score >= thresholds["high_confidence_min_score"]
            and margin >= thresholds["high_confidence_min_margin"]
        ):
            confidence = "high"
        elif (
            best_score >= thresholds["moderate_confidence_min_score"]
            and margin >= thresholds["moderate_confidence_min_margin"]
        ):
            confidence = "moderate"
        else:
            confidence = "low"

        best_language = resolve_supported_language_code(best_language_raw)
        return LanguageDetectionResult(
            detected_input_language=best_language,
            report_language=best_language,
            confidence=confidence,
        )


detect_clinical_language = ClinicalLanguageDetector.detect
