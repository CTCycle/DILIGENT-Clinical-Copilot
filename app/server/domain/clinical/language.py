# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from dataclasses import dataclass

###############################################################################
@dataclass(frozen=True)
class LanguageDetectionResult:
    detected_input_language: str
    report_language: str
    confidence: str
