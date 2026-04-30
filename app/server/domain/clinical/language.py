from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageDetectionResult:
    detected_input_language: str
    report_language: str
    confidence: str
