from __future__ import annotations

from typing import Literal

ClinicalSectionKey = Literal["anamnesis", "drugs", "laboratory_analysis"]

SECTION_FRAGMENT_JOINER = "\n\n"
SECTION_KEYS: tuple[ClinicalSectionKey, ...] = (
    "anamnesis",
    "drugs",
    "laboratory_analysis",
)

SECTION_DISPLAY_NAMES: dict[ClinicalSectionKey, str] = {
    "anamnesis": "Anamnesis",
    "drugs": "Drugs",
    "laboratory_analysis": "Lab analysis",
}

SECTION_KEY_BY_INDEX: dict[int, ClinicalSectionKey] = {
    1: "anamnesis",
    2: "drugs",
    3: "laboratory_analysis",
}
