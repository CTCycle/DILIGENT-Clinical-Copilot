from __future__ import annotations

from typing import Any

from services.inspection.normalization import normalize_text as normalize_text_value
from services.text.normalization import normalize_drug_query_name


def extract_revision_drug_names(payload: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for value in payload.get("detected_drugs", []):
        if isinstance(value, str) and value.strip():
            names.append(value.strip())
    for value in payload.get("matched_drugs", []):
        if not isinstance(value, dict):
            continue
        for key in ("raw_drug_name", "matched_drug_name"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                names.append(candidate.strip())
    unique: list[str] = []
    seen: set[str] = set()
    for name in names:
        normalized = normalize_drug_query_name(name)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(name)
    return unique


def build_revision_section_validation(
    *,
    source_sections: dict[str, Any],
    extracted_sections: dict[str, Any],
    selected_text: str | None,
) -> dict[str, Any]:
    section_keys = ("anamnesis", "drugs", "laboratory_analysis")
    validation: dict[str, dict[str, Any]] = {}
    missing_after_revision: list[str] = []
    changed_after_revision: list[str] = []
    selected_norm = normalize_text_value(selected_text or "")
    for key in section_keys:
        original_text = normalize_text_value(source_sections.get(key))
        extracted_text = normalize_text_value(extracted_sections.get(key))
        original_in_scope = not selected_norm or bool(
            extracted_text
            or (original_text and selected_norm in original_text)
            or (original_text and original_text in selected_norm)
        )
        changed = bool(original_in_scope and original_text != extracted_text)
        if original_in_scope and original_text and not extracted_text:
            missing_after_revision.append(key)
        if changed:
            changed_after_revision.append(key)
        validation[key] = {
            "original_length": len(original_text),
            "revised_length": len(extracted_text),
            "original_in_revision_scope": original_in_scope,
            "present_after_revision": bool(extracted_text),
            "changed_after_revision": changed,
        }
    return {
        "sections": validation,
        "missing_sections_after_revision": missing_after_revision,
        "changed_sections_after_revision": changed_after_revision,
    }
