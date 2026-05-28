from __future__ import annotations

from domain.clinical.entities import ClinicalSectionTemplateResponse
from services.catalogs.runtime import get_reference_catalog_snapshot


###############################################################################
def get_clinical_section_template() -> ClinicalSectionTemplateResponse:
    snapshot = get_reference_catalog_snapshot()
    section_keys = list(
        snapshot.values("clinical_extraction", "clinical_sections", key="default")
    ) or ["anamnesis", "drugs", "laboratory_analysis"]
    anamnesis_aliases = list(
        snapshot.values("clinical_extraction", "section_aliases", key="anamnesis")
    )
    therapy_aliases = list(
        snapshot.values("clinical_extraction", "section_aliases", key="drugs")
    )
    labs_aliases = list(
        snapshot.values(
            "clinical_extraction", "section_aliases", key="laboratory_analysis"
        )
    )
    sections = {
        "anamnesis": anamnesis_aliases,
        "therapy": therapy_aliases,
        "laboratory_history": labs_aliases,
    }
    heading_map = {
        "anamnesis": "Anamnesis",
        "drugs": "Therapy",
        "laboratory_analysis": "Laboratory history",
    }
    ordered_headings = [
        heading_map.get(section_key, section_key.replace("_", " ").title())
        for section_key in section_keys
    ]
    template = "\n".join(
        [
            f"Provide plain text with {len(ordered_headings)} clearly titled and separated sections.",
            "",
            f"## {ordered_headings[0] if len(ordered_headings) > 0 else 'Anamnesis'}",
            "Diagnoses, symptoms, clinical conditions, disease timeline, previous therapies, relevant history.",
            "",
            f"## {ordered_headings[1] if len(ordered_headings) > 1 else 'Therapy'}",
            "Current and past drugs, dosage, route, start date or timing, suspension status. Bullet lists allowed.",
            "",
            f"## {ordered_headings[2] if len(ordered_headings) > 2 else 'Laboratory history'}",
            "ALT, ALP, bilirubin, GGT, AST, INR, dates, units, ULN values, hepatic pattern (if known), RUCAM score (if already calculated).",
            "",
            (
                "Equivalent headings are accepted. "
                f"Anamnesis aliases: {', '.join(anamnesis_aliases[:5]) or 'n/a'}. "
                f"Therapy aliases: {', '.join(therapy_aliases[:5]) or 'n/a'}. "
                f"Laboratory aliases: {', '.join(labs_aliases[:5]) or 'n/a'}."
            ),
        ]
    )
    return ClinicalSectionTemplateResponse(headings=sections, template=template)
