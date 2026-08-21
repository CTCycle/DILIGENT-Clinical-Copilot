from __future__ import annotations

import hashlib
import json
from typing import Any

from services.llm.context_budget import ContextSegment, build_context_plan

UNKNOWN_CAPACITY_INPUT_BUDGET = 8192

###############################################################################
def _bounded(value: Any, limit: int) -> dict[str, Any]:
    text = str(value or "")
    return {
        "text": text[:limit],
        "truncated": len(text) > limit,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }

###############################################################################
def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

###############################################################################
def build_revision_context(
    *,
    session: dict[str, Any],
    manual_edits: list[dict[str, Any]],
    lineage: list[dict[str, Any]],
    selected_text: str | None,
    instruction: str | None,
    input_budget: int | None = None,
) -> dict[str, Any]:
    payload_value = session.get("result_payload")
    payload: dict[str, Any] = payload_value if isinstance(payload_value, dict) else {}
    artifacts_value = payload.get("pipeline_artifacts")
    artifacts: dict[str, Any] = (
        artifacts_value if isinstance(artifacts_value, dict) else {}
    )

    source_text = _bounded(
        session.get("source_clinical_text") or session.get("session_text"),
        16000,
    )
    sections = {
        key: _bounded(value, 8000)
        for key, value in (session.get("sections") or {}).items()
    }
    official_report = _bounded(
        session.get("official_report_text") or session.get("report"),
        12000,
    )
    final_report = _bounded(
        payload.get("final_report") or payload.get("report"),
        12000,
    )
    selected = _bounded(selected_text, 6000)
    user_instruction = _bounded(instruction, 4000)

    values: dict[str, Any] = {
        "clinical.source_text": source_text,
        "review.official_report": official_report,
        "review.final_report": final_report,
        "user.selected_text": selected,
        "user.instruction": user_instruction,
        "audit.manual_edits": manual_edits,
        "audit.version_lineage": lineage,
        "audit.metadata": session.get("metadata") or {},
    }
    priorities = {
        "clinical.source_text": (100, True, "clinical_source"),
        "review.official_report": (95, True, "official_report"),
        "review.final_report": (90, False, "final_report"),
        "user.selected_text": (90, True, "user_steering"),
        "user.instruction": (110, True, "user_steering"),
        "audit.manual_edits": (50, False, "audit"),
        "audit.version_lineage": (40, False, "audit"),
        "audit.metadata": (30, False, "audit"),
    }

    segments: list[ContextSegment] = []
    for key, value in values.items():
        priority, required, source_kind = priorities[key]
        dedupe_key = None
        if key in {"review.official_report", "review.final_report"}:
            digest = value.get("sha256") if isinstance(value, dict) else None
            if digest:
                dedupe_key = f"report:{digest}"
        segments.append(
            ContextSegment(
                key=key,
                text=_json_text(value),
                priority=priority,
                required=required,
                source_kind=source_kind,
                deduplication_key=dedupe_key,
            )
        )

    for section_name, section_value in sorted(sections.items()):
        key = f"clinical.section.{section_name}"
        values[key] = section_value
        segments.append(
            ContextSegment(
                key=key,
                text=_json_text(section_value),
                priority=80,
                source_kind="clinical_section",
            )
        )

    for field_name, value in (
        (
            "structured_case",
            artifacts.get("structured_case") or payload.get("structured_case"),
        ),
        ("lab_timeline", payload.get("lab_timeline")),
        ("matched_drugs", payload.get("matched_drugs")),
        ("rucam_assessments", payload.get("rucam_assessments")),
        (
            "dili_evidence_bundle",
            artifacts.get("dili_evidence_bundle")
            or payload.get("dili_evidence_bundle"),
        ),
    ):
        if value is None:
            continue
        key = f"clinical.structured.{field_name}"
        values[key] = value
        segments.append(
            ContextSegment(
                key=key,
                text=_json_text(value),
                priority=65,
                source_kind="structured_evidence",
            )
        )

    requested_budget = input_budget
    planning_budget = (
        max(0, input_budget)
        if input_budget is not None
        else UNKNOWN_CAPACITY_INPUT_BUDGET
    )
    plan = build_context_plan(segments, input_budget=planning_budget)
    selected_keys = {segment.key for segment in plan.selected}

    def selected_value(key: str, default: Any = None) -> Any:
        return values[key] if key in selected_keys else default

    selected_sections = {
        name: selected_value(f"clinical.section.{name}")
        for name in sorted(sections)
        if f"clinical.section.{name}" in selected_keys
    }
    structured_fields = {
        name: selected_value(f"clinical.structured.{name}")
        for name in (
            "structured_case",
            "lab_timeline",
            "matched_drugs",
            "rucam_assessments",
            "dili_evidence_bundle",
        )
        if f"clinical.structured.{name}" in selected_keys
    }
    omitted = [segment.key for segment in plan.omitted]
    selection_report = dict(plan.selection_report)
    selection_report.update(
        {
            "capacity_known": requested_budget is not None,
            "planning_input_budget": planning_budget,
            "unknown_capacity_fallback": requested_budget is None,
        }
    )

    clinical_evidence: dict[str, Any] = {
        "source_text": selected_value("clinical.source_text", {"omitted": True}),
        "sections": selected_sections,
    }
    for field_name, value in structured_fields.items():
        clinical_evidence[field_name] = value

    return {
        "provenance": {
            "session_id": session.get("session_id"),
            "source": "persisted_session",
        },
        "clinical_evidence": clinical_evidence,
        "review_target": {
            "official_report": selected_value(
                "review.official_report", {"omitted": True}
            ),
            "final_report": selected_value("review.final_report", {"omitted": True}),
        },
        "audit": {
            "manual_edits": selected_value("audit.manual_edits", []),
            "version_lineage": selected_value("audit.version_lineage", []),
            "metadata": selected_value("audit.metadata", {}),
        },
        "user_steering": {
            "selected_text": selected_value("user.selected_text", {"omitted": True}),
            "instruction": selected_value("user.instruction", {"omitted": True}),
        },
        "context_budget": {
            "status": plan.status,
            "required_overflow": plan.required_overflow,
            "omitted_segments": omitted,
            "selection_report": selection_report,
        },
    }
