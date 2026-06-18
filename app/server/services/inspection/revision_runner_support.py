from __future__ import annotations

from typing import Any


REVISION_STEP_SEQUENCE: list[tuple[str, str]] = [
    ("load_source_version", "Loading selected source version"),
    ("analyze_reviewer_instructions", "Analyzing reviewer instructions"),
    ("prepare_runtime", "Preparing revision runtime"),
    ("preprocess_input", "Preprocessing source clinical text"),
    ("generate_revision", "Generating revised clinical session"),
    ("resolve_revision_extraction", "Resolving revision extraction bundle"),
    ("validate_anamnesis_drugs", "Validating revised anamnesis drugs"),
    (
        "extract_missing_anamnesis_drugs",
        "Extracting missing anamnesis drug candidates",
    ),
    ("revise_labs_timeline", "Revising structured laboratory timeline"),
    ("reconcile_revision_candidates", "Reconciling revision candidate selection"),
    ("merge_revision_snapshot", "Merging revision entity snapshot"),
    ("resolve_livertox_matches", "Resolving revision LiverTox matches"),
    ("rerun_dili_assessments", "Rebuilding revision DILI assessments"),
    ("rebuild_final_report", "Rebuilding revision final report"),
    ("qa_validate_revision", "Validating rebuilt revision output"),
    ("persist_revision", "Persisting revision artifacts"),
    ("finalize_revision_version", "Finalizing revision version state"),
]


###############################################################################
def derive_revision_run_actor_source(metadata: dict[str, Any]) -> str:
    return (
        "manual_entry"
        if str((metadata or {}).get("reviewer") or "").strip()
        else "unknown"
    )


###############################################################################
def report_revision_progress(
    jobs: Any,
    job_id: str | None,
    _stage: str,
    progress: float,
    _detail: str | None = None,
) -> None:
    if job_id:
        jobs.update_progress(job_id, progress)


###############################################################################
def ensure_revision_not_cancelled(jobs: Any, job_id: str | None) -> None:
    if job_id and jobs.should_stop(job_id):
        raise RuntimeError("Revision job was cancelled")


###############################################################################
def derive_revision_qa_outcome(result_payload: dict[str, Any]) -> tuple[str, str]:
    revision_payload = result_payload.get("revision")
    if isinstance(revision_payload, dict):
        qa_validation = revision_payload.get("qa_validation")
        if isinstance(qa_validation, dict):
            version_status = str(qa_validation.get("version_status") or "").strip()
            qa_status = str(qa_validation.get("status") or "").strip()
            if version_status in {
                "llm_qa_passed",
                "qa_failed",
                "requires_human_review",
            } and qa_status in {
                "passed",
                "passed_with_warnings",
                "failed",
                "requires_human_review",
            }:
                return version_status, qa_status
    blocking_issues = result_payload.get("blocking_issues")
    if isinstance(blocking_issues, list) and blocking_issues:
        return "qa_failed", "failed"
    if bool(result_payload.get("manual_review_required")):
        return "requires_human_review", "requires_human_review"
    pipeline_artifacts = result_payload.get("pipeline_artifacts")
    if isinstance(pipeline_artifacts, dict) and isinstance(
        pipeline_artifacts.get("faithfulness_audit"),
        dict,
    ):
        return "llm_qa_passed", "passed"
    return "requires_human_review", "not_run"


###############################################################################
def get_revision_entity_pipeline(
    result_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    revision_payload = result_payload.get("revision")
    if not isinstance(revision_payload, dict):
        return {}
    entity_pipeline = revision_payload.get("entity_pipeline")
    if not isinstance(entity_pipeline, dict):
        return {}
    return {
        str(step_name): payload
        for step_name, payload in entity_pipeline.items()
        if isinstance(step_name, str) and isinstance(payload, dict)
    }


###############################################################################
def summarize_revision_entity_stage_payload(
    step_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if step_name == "validate_anamnesis_drugs":
        return {
            "status": payload.get("status"),
            "deterministic_detected_count": len(
                payload.get("deterministic_detected_names") or []
            ),
            "revised_detected_count": len(payload.get("revised_detected_names") or []),
            "supplemental_detected_count": len(
                payload.get("revised_only_names") or []
            ),
        }
    if step_name == "resolve_revision_extraction":
        return {
            "status": payload.get("status"),
            "therapy_source": payload.get("therapy_source"),
            "anamnesis_source": payload.get("anamnesis_source"),
            "disease_source": payload.get("disease_source"),
            "therapy_structured_count": len(
                payload.get("therapy_structured_names") or []
            ),
            "anamnesis_structured_count": len(
                payload.get("anamnesis_structured_names") or []
            ),
            "disease_deterministic_count": len(
                payload.get("disease_deterministic_names") or []
            ),
        }
    if step_name == "extract_missing_anamnesis_drugs":
        return {
            "status": payload.get("status"),
            "supplemental_drug_count": len(payload.get("supplemental_drug_names") or []),
        }
    if step_name == "revise_labs_timeline":
        return {
            "status": payload.get("status"),
            "lab_entry_count": int(payload.get("lab_entry_count") or 0),
            "marker_count": len(payload.get("marker_names") or []),
        }
    if step_name == "reconcile_revision_candidates":
        return {
            "status": payload.get("status"),
            "analysis_drug_count": len(payload.get("analysis_drug_names") or []),
            "relevant_drug_count": len(payload.get("relevant_drug_names") or []),
            "unresolved_drug_count": len(payload.get("unresolved_drug_names") or []),
        }
    if step_name == "merge_revision_snapshot":
        return {
            "status": payload.get("status"),
            "therapy_drug_count": len(payload.get("therapy_drug_names") or []),
            "anamnesis_drug_count": len(payload.get("anamnesis_drug_names") or []),
            "analysis_drug_count": len(payload.get("analysis_drug_names") or []),
            "rucam_assessment_count": int(payload.get("rucam_assessment_count") or 0),
        }
    return {"status": payload.get("status")}
