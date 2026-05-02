from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ClinicalSectionKey = Literal["anamnesis", "drugs", "laboratory_analysis"]

SECTION_FRAGMENT_JOINER = "\n\n"


@dataclass(frozen=True)
class ClinicalSectionDefinition:
    key: ClinicalSectionKey
    display_name: str
    canonical_labels: tuple[str, ...]
    xml_tags: tuple[str, ...]
    json_keys: tuple[str, ...]
    indexed_position: int


CLINICAL_SECTION_DEFINITIONS: tuple[ClinicalSectionDefinition, ...] = (
    ClinicalSectionDefinition(
        key="anamnesis",
        display_name="Anamnesis",
        canonical_labels=("anamnesis",),
        xml_tags=("anamnesis",),
        json_keys=("anamnesis",),
        indexed_position=1,
    ),
    ClinicalSectionDefinition(
        key="drugs",
        display_name="Current therapy",
        canonical_labels=("current therapy", "current_therapy", "therapy", "drugs"),
        xml_tags=("current_therapy", "drugs"),
        json_keys=("current_therapy", "drugs"),
        indexed_position=2,
    ),
    ClinicalSectionDefinition(
        key="laboratory_analysis",
        display_name="Laboratory analysis",
        canonical_labels=(
            "laboratory analysis",
            "laboratory_analysis",
            "labs",
            "lab results",
            "lab_results",
        ),
        xml_tags=("laboratory_analysis", "labs", "lab_results"),
        json_keys=("laboratory_analysis", "labs", "lab_results"),
        indexed_position=3,
    ),
)

SECTION_KEYS: tuple[ClinicalSectionKey, ...] = tuple(
    definition.key for definition in CLINICAL_SECTION_DEFINITIONS
)

SECTION_BY_KEY: dict[ClinicalSectionKey, ClinicalSectionDefinition] = {
    definition.key: definition for definition in CLINICAL_SECTION_DEFINITIONS
}

SECTION_KEY_BY_LABEL: dict[str, ClinicalSectionKey] = {
    label: definition.key
    for definition in CLINICAL_SECTION_DEFINITIONS
    for label in definition.canonical_labels
}

SECTION_KEY_BY_XML_TAG: dict[str, ClinicalSectionKey] = {
    tag: definition.key
    for definition in CLINICAL_SECTION_DEFINITIONS
    for tag in definition.xml_tags
}

SECTION_KEY_BY_JSON_KEY: dict[str, ClinicalSectionKey] = {
    key: definition.key
    for definition in CLINICAL_SECTION_DEFINITIONS
    for key in definition.json_keys
}

SECTION_KEY_BY_INDEX: dict[int, ClinicalSectionKey] = {
    definition.indexed_position: definition.key
    for definition in CLINICAL_SECTION_DEFINITIONS
}
