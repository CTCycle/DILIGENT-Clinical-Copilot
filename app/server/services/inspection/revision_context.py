from __future__ import annotations

import hashlib
from typing import Any

###############################################################################
def _bounded(value: Any, limit: int) -> dict[str, Any]:
    text = str(value or "")
    return {
        "text": text[:limit],
        "truncated": len(text) > limit,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }

###############################################################################
def build_revision_context(
    *,
    session: dict[str, Any],
    manual_edits: list[dict[str, Any]],
    lineage: list[dict[str, Any]],
    selected_text: str | None,
    instruction: str | None,
) -> dict[str, Any]:
    payload_value = session.get("result_payload")
    payload: dict[str, Any] = payload_value if isinstance(payload_value, dict) else {}
    artifacts_value = payload.get("pipeline_artifacts")
    artifacts: dict[str, Any] = (
        artifacts_value if isinstance(artifacts_value, dict) else {}
    )
    return {
        "provenance": {
            "session_id": session.get("session_id"),
            "source": "persisted_session",
        },
        "clinical_evidence": {
            "source_text": _bounded(
                session.get("source_clinical_text") or session.get("session_text"),
                30000,
            ),
            "sections": {
                key: _bounded(value, 10000)
                for key, value in (session.get("sections") or {}).items()
            },
            "structured_case": artifacts.get("structured_case")
            or payload.get("structured_case"),
            "lab_timeline": payload.get("lab_timeline"),
            "matched_drugs": payload.get("matched_drugs"),
            "rucam_assessments": payload.get("rucam_assessments"),
            "dili_evidence_bundle": artifacts.get("dili_evidence_bundle")
            or payload.get("dili_evidence_bundle"),
        },
        "review_target": {
            "official_report": _bounded(
                session.get("official_report_text") or session.get("report"), 20000
            ),
            "final_report": _bounded(
                payload.get("final_report") or payload.get("report"), 20000
            ),
        },
        "audit": {
            "manual_edits": manual_edits,
            "version_lineage": lineage,
            "metadata": session.get("metadata") or {},
        },
        "user_steering": {
            "selected_text": _bounded(selected_text, 10000),
            "instruction": _bounded(instruction, 4000),
        },
    }
