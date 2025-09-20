from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, status
from pydantic import ValidationError

from Pharmagent.app.api.schemas.clinical import PatientData
from Pharmagent.app.api.schemas.regex import CUTOFF_IN_PAREN_RE, NUMERIC_RE
from Pharmagent.app.constants import TASKS_PATH
from Pharmagent.app.logger import logger
from Pharmagent.app.utils.services.parser import PatientCase

router = APIRouter(tags=["agent"])

patient_case = PatientCase()

ALT_LABELS = {"ALT", "ALAT"}
ALP_LABELS = {"ALP"}


# [HELPERS]
###############################################################################
def _sanitize_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None

# -----------------------------------------------------------------------------
def _extract_sections(raw_text: str, name: str | None) -> dict[str, str]:
    cleaned = patient_case.clean_patient_info(raw_text)
    return patient_case.split_text_by_tags(cleaned, name)

# -----------------------------------------------------------------------------
def _format_marker_value(value: str | None, unit: str | None) -> str | None:
    if not value:
        return None
    unit_part = unit.strip() if unit else ""
    return f"{value} {unit_part}".strip()

# -----------------------------------------------------------------------------
def _extract_cutoff(paren_text: str | None) -> str | None:
    if not paren_text:
        return None
    cutoff_match = CUTOFF_IN_PAREN_RE.search(paren_text)
    if cutoff_match:
        return cutoff_match.group(1)
    max_match = re.search(r"max[: ]*([0-9]+(?:[.,][0-9]+)?)", paren_text, re.IGNORECASE)
    if max_match:
        return max_match.group(1)
    return None

# -----------------------------------------------------------------------------
def _parse_hepatic_markers(section: str | None) -> dict[str, Any]:
    markers: dict[str, Any] = {
        "alt": None,
        "alt_max": None,
        "alp": None,
        "alp_max": None,
    }
    if not section:
        return markers

    for match in NUMERIC_RE.finditer(section):
        raw_name = (match.group("name") or "").replace(":", "").strip().upper()
        normalized = raw_name.replace(" ", "")
        formatted_value = _format_marker_value(
            match.group("value"), match.group("unit")
        )
        cutoff_value = _extract_cutoff(match.group("paren"))
        if normalized in ALT_LABELS:
            markers["alt"] = formatted_value
            markers["alt_max"] = cutoff_value
        elif normalized in ALP_LABELS:
            markers["alp"] = formatted_value
            markers["alp_max"] = cutoff_value

    return markers


# [ENPOINTS]
###############################################################################
async def process_single_patient(single_payload: PatientData) -> dict[str, Any]:
    logger.info(
        "Launching placeholder LLM workflow for patient: %s",
        single_payload.name or "Unknown",
    )
    # Placeholder for LLM-driven workflow. Will be replaced with concrete logic.
    return {
        "name": single_payload.name or "Unknown",
        "anamnesis": single_payload.anamnesis,
        "drugs": single_payload.drugs,
        "exams": single_payload.exams,
        "alt": single_payload.alt,
        "alt_max": single_payload.alt_max,
        "alp": single_payload.alp,
        "alp_max": single_payload.alp_max,
        "symptoms": single_payload.symptoms,
        "note": "LLM workflow pending implementation.",
    }


# -----------------------------------------------------------------------------
@router.post("/agent", response_model=None, status_code=status.HTTP_202_ACCEPTED)
async def start_single_clinical_agent(
    name: str | None = Body(default=None),
    anamnesis: str | None = Body(default=None),
    drugs: str | None = Body(default=None),
    exams: str | None = Body(default=None),
    alt: str | None = Body(default=None),
    alt_max: str | None = Body(default=None),
    alp: str | None = Body(default=None),
    alp_max: str | None = Body(default=None),
    symptoms: list[str] | None = Body(default=None),
) -> dict[str, Any]:
    logger.info(
        "Starting clinical agent processing for patient: %s",
        _sanitize_text(name) or "Unknown",
    )

    try:
        payload = PatientData(
            name=_sanitize_text(name),
            anamnesis=_sanitize_text(anamnesis),
            drugs=_sanitize_text(drugs),
            exams=_sanitize_text(exams),
            alt=_sanitize_text(alt),
            alt_max=_sanitize_text(alt_max),
            alp=_sanitize_text(alp),
            alp_max=_sanitize_text(alp_max),
            symptoms=symptoms or [],
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.errors()
        ) from exc

    single_result = await process_single_patient(payload)
    return {"status": "success", "processed": 1, "patients": [single_result]}


# -----------------------------------------------------------------------------
@router.post("/batch-agent", response_model=None, status_code=status.HTTP_202_ACCEPTED)
async def start_batch_clinical_agent() -> dict[str, Any]:
    txt_files = [path for path in Path(TASKS_PATH).glob("*.txt") if path.is_file()]

    if not txt_files:
        logger.info(
            "No .txt files found in default path. Add new files and rerun the batch agent."
        )
        return {"status": "success", "processed": 0, "patients": []}

    results: list[dict[str, Any]] = []
    for path in txt_files:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed reading {path}: {exc}")
            continue

        sections = _extract_sections(text, path.stem)
        hepatic_markers = _parse_hepatic_markers(sections.get("blood_tests"))
        anamnesis_section = _sanitize_text(sections.get("anamnesis"))
        drugs_section = _sanitize_text(sections.get("drugs"))
        exams_section = _sanitize_text(sections.get("additional_tests"))

        try:
            patient_payload = PatientData(
                name=_sanitize_text(path.stem),
                anamnesis=anamnesis_section,
                drugs=drugs_section,
                exams=exams_section,
                alt=hepatic_markers["alt"],
                alt_max=hepatic_markers["alt_max"],
                alp=hepatic_markers["alp"],
                alp_max=hepatic_markers["alp_max"],
                symptoms=[],
            )
        except ValidationError as exc:
            logger.error(f"Invalid data for patient {path.stem}: {exc}")
            continue

        case = await process_single_patient(patient_payload)
        results.append(case)

    return {"status": "success", "processed": len(results), "patients": results}
