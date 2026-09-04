from __future__ import annotations

from typing import Any

from domain.clinical.dili import DiliEvidenceBundle
from services.clinical.dili_evidence import DiliEvidenceBuilder


###############################################################################
def audit_revised_dili_report(
    *,
    session: dict[str, Any],
    report_text: str,
) -> list[str]:
    payload_value = session.get("result_payload")
    payload = payload_value if isinstance(payload_value, dict) else {}
    artifacts_value = payload.get("pipeline_artifacts")
    artifacts = artifacts_value if isinstance(artifacts_value, dict) else {}
    bundle_value = artifacts.get("dili_evidence_bundle") or payload.get(
        "dili_evidence_bundle"
    )
    if bundle_value is None:
        return [
            "Structured DILI evidence is unavailable for deterministic revision safety validation."
        ]
    try:
        bundle = DiliEvidenceBundle.model_validate(bundle_value)
    except Exception:
        return [
            "Structured DILI evidence is invalid and the revised report cannot be finalized safely."
        ]

    issues = DiliEvidenceBuilder.audit_generated_narrative(
        clinical_narrative=report_text,
        bundle=bundle,
    )
    return [str(issue.get("message") or issue.get("code") or "Clinical safety issue") for issue in issues]
