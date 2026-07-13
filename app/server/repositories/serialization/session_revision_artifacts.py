from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update

from repositories.schemas.models import (
    ClinicalSessionRevisionArtifact,
)
from repositories.serialization.session_revision_data import (
    REVISION_DRUG_SCHEMA_NAME,
    REVISION_DISEASE_SCHEMA_NAME,
    REVISION_LAB_SCHEMA_NAME,
    REVISION_LIVERTOX_DECISION_SCHEMA_NAME,
    REVISION_DILI_ASSESSMENT_SCHEMA_NAME,
    _create_revision_artifact_row,
    _create_revision_entity_row,
    normalize_text_key,
    serialize_revision_artifact_row,
    serialize_revision_entity_row,
    validate_revised_dili_assessment,
    validate_revised_disease_payload,
    validate_revised_drug_payload,
    validate_revised_lab_payload,
    validate_revision_livertox_decision,
)

###############################################################################
def persist_revision_artifacts(
    self,
    *,
    pipeline_run_id: str,
    revision_version_id: int,
    result_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    db_session = self.session_factory()
    try:
        safe_revision_version_id = int(revision_version_id)
        safe_pipeline_run_id = str(pipeline_run_id)
        db_session.query(ClinicalSessionRevisionArtifact).filter(
            ClinicalSessionRevisionArtifact.revision_version_id
            == safe_revision_version_id
        ).delete()

        created_rows: list[ClinicalSessionRevisionArtifact] = []

        structured_case = result_payload.get("structured_case")
        if isinstance(structured_case, dict):
            for entity_type in (
                "therapy_drugs",
                "anamnesis_drugs",
                "anamnesis_diseases",
            ):
                entries = structured_case.get(entity_type)
                if not isinstance(entries, list):
                    continue
                for index, entry in enumerate(entries):
                    if not isinstance(entry, dict):
                        continue
                    entity_name = entry.get("name") or entry.get("drug_name")
                    row = _create_revision_artifact_row(
                        self,
                        revision_version_id=safe_revision_version_id,
                        pipeline_run_id=safe_pipeline_run_id,
                        artifact_kind="structured_case_entity",
                        artifact_key=f"{entity_type}:{index}",
                        entity_type=entity_type,
                        entity_name=str(entity_name).strip() if entity_name else None,
                        status="derived",
                        payload=entry,
                    )
                    db_session.add(row)
                    created_rows.append(row)

        faithfulness_audit = None
        pipeline_artifacts = result_payload.get("pipeline_artifacts")
        if isinstance(pipeline_artifacts, dict):
            raw_faithfulness_audit = pipeline_artifacts.get("faithfulness_audit")
            if isinstance(raw_faithfulness_audit, dict):
                faithfulness_audit = raw_faithfulness_audit
                row = _create_revision_artifact_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    pipeline_run_id=safe_pipeline_run_id,
                    artifact_kind="llm_qa_output",
                    artifact_key="faithfulness_audit",
                    status=(
                        "failed"
                        if result_payload.get("blocking_issues")
                        else "requires_human_review"
                        if bool(result_payload.get("manual_review_required"))
                        else "passed"
                    ),
                    payload={
                        **faithfulness_audit,
                        "manual_review_required": bool(
                            result_payload.get("manual_review_required")
                        ),
                        "blocking_issues": result_payload.get("blocking_issues") or [],
                    },
                )
                db_session.add(row)
                created_rows.append(row)
            fact_graph_validation = pipeline_artifacts.get("fact_graph_validation")
            if isinstance(fact_graph_validation, dict):
                row = _create_revision_artifact_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    pipeline_run_id=safe_pipeline_run_id,
                    artifact_kind="pipeline_artifact",
                    artifact_key="fact_graph_validation",
                    status="derived",
                    payload=fact_graph_validation,
                )
                db_session.add(row)
                created_rows.append(row)

        report_comparison = result_payload.get("report_comparison")
        if isinstance(report_comparison, dict):
            row = _create_revision_artifact_row(
                self,
                revision_version_id=safe_revision_version_id,
                pipeline_run_id=safe_pipeline_run_id,
                artifact_kind="report_comparison",
                artifact_key="report_comparison",
                status=self.normalize_string(
                    str(report_comparison.get("outcome") or "")
                ),
                payload=report_comparison,
            )
            db_session.add(row)
            created_rows.append(row)

        revision_payload = result_payload.get("revision")
        if isinstance(revision_payload, dict):
            instruction_profile = revision_payload.get("instruction_profile")
            if isinstance(instruction_profile, dict):
                row = _create_revision_artifact_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    pipeline_run_id=safe_pipeline_run_id,
                    artifact_kind="pipeline_artifact",
                    artifact_key="reviewer_instruction_profile",
                    status="derived",
                    payload=instruction_profile,
                )
                db_session.add(row)
                created_rows.append(row)
            instruction_trace = revision_payload.get("instruction_trace")
            if isinstance(instruction_trace, dict):
                row = _create_revision_artifact_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    pipeline_run_id=safe_pipeline_run_id,
                    artifact_kind="pipeline_artifact",
                    artifact_key="reviewer_instruction_trace",
                    status="derived",
                    payload=instruction_trace,
                )
                db_session.add(row)
                created_rows.append(row)
            final_report_rebuild = revision_payload.get("final_report_rebuild")
            if isinstance(final_report_rebuild, dict):
                row = _create_revision_artifact_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    pipeline_run_id=safe_pipeline_run_id,
                    artifact_kind="pipeline_artifact",
                    artifact_key="final_report_rebuild",
                    status=(
                        "warning"
                        if bool(final_report_rebuild.get("warnings"))
                        else "derived"
                    ),
                    payload=final_report_rebuild,
                )
                db_session.add(row)
                created_rows.append(row)
            qa_validation = revision_payload.get("qa_validation")
            if isinstance(qa_validation, dict):
                row = _create_revision_artifact_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    pipeline_run_id=safe_pipeline_run_id,
                    artifact_kind="llm_qa_output",
                    artifact_key="revision_qa_validation",
                    status=self.normalize_string(
                        str(qa_validation.get("status") or "")
                    ),
                    payload=qa_validation,
                )
                db_session.add(row)
                created_rows.append(row)
            entity_pipeline = revision_payload.get("entity_pipeline")
            if isinstance(entity_pipeline, dict):
                row = _create_revision_artifact_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    pipeline_run_id=safe_pipeline_run_id,
                    artifact_kind="pipeline_artifact",
                    artifact_key="revision_entity_pipeline",
                    status="derived",
                    payload=entity_pipeline,
                )
                db_session.add(row)
                created_rows.append(row)
            entity_snapshot_context = revision_payload.get("entity_snapshot_context")
            if (
                isinstance(entity_snapshot_context, str)
                and entity_snapshot_context.strip()
            ):
                row = _create_revision_artifact_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    pipeline_run_id=safe_pipeline_run_id,
                    artifact_kind="pipeline_artifact",
                    artifact_key="revision_entity_snapshot_context",
                    status="derived",
                    payload={"text": entity_snapshot_context.strip()},
                )
                db_session.add(row)
                created_rows.append(row)
            consultation_execution = revision_payload.get("consultation_execution")
            if isinstance(consultation_execution, dict):
                row = _create_revision_artifact_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    pipeline_run_id=safe_pipeline_run_id,
                    artifact_kind="pipeline_artifact",
                    artifact_key="revision_consultation_execution",
                    status="derived",
                    payload=consultation_execution,
                )
                db_session.add(row)
                created_rows.append(row)
            finalization_execution = revision_payload.get("finalization_execution")
            if isinstance(finalization_execution, dict):
                row = _create_revision_artifact_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    pipeline_run_id=safe_pipeline_run_id,
                    artifact_kind="pipeline_artifact",
                    artifact_key="revision_finalization_execution",
                    status="derived",
                    payload=finalization_execution,
                )
                db_session.add(row)
                created_rows.append(row)

        db_session.flush()
        db_session.commit()
        return [serialize_revision_artifact_row(self, row) for row in created_rows]
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()

###############################################################################
def persist_revision_agent_issue_scan(
    self,
    *,
    pipeline_run_id: str,
    revision_version_id: int,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    db_session = self.session_factory()
    try:
        row = _create_revision_artifact_row(
            self,
            revision_version_id=int(revision_version_id),
            pipeline_run_id=str(pipeline_run_id),
            artifact_kind="pipeline_artifact",
            artifact_key="revision_agent_issue_scan",
            status="requires_human_review",
            payload=payload,
        )
        db_session.add(row)
        db_session.flush()
        db_session.commit()
        return [serialize_revision_artifact_row(self, row)]
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()

###############################################################################
def persist_revision_entities(
    self,
    *,
    pipeline_run_id: str,
    revision_version_id: int,
    source_version_id: int | None,
    result_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    db_session = self.session_factory()
    try:
        safe_revision_version_id = int(revision_version_id)
        safe_pipeline_run_id = str(pipeline_run_id)
        now = datetime.now(UTC)
        db_session.execute(
            update(ClinicalSessionRevisionArtifact)
            .where(
                ClinicalSessionRevisionArtifact.revision_version_id
                == safe_revision_version_id,
                ClinicalSessionRevisionArtifact.artifact_kind
                == "structured_case_entity",
                ClinicalSessionRevisionArtifact.status != "superseded",
            )
            .values(status="superseded", updated_at=now)
        )

        created_rows: list[ClinicalSessionRevisionArtifact] = []

        structured_case = result_payload.get("structured_case")
        if isinstance(structured_case, dict):
            for section_name, source_section in (
                ("therapy_drugs", "therapy"),
                ("anamnesis_drugs", "anamnesis"),
            ):
                entries = structured_case.get(section_name)
                if not isinstance(entries, list):
                    continue
                for index, entry in enumerate(entries):
                    if not isinstance(entry, dict):
                        continue
                    validated_entry = validate_revised_drug_payload(entry)
                    serialized_entry = validated_entry.model_dump(exclude_none=True)
                    revised_name = validated_entry.name
                    row = _create_revision_entity_row(
                        self,
                        revision_version_id=safe_revision_version_id,
                        source_version_id=(
                            int(source_version_id)
                            if source_version_id is not None
                            else None
                        ),
                        pipeline_run_id=safe_pipeline_run_id,
                        step_name="generate_revision",
                        entity_type="drug",
                        source_section=source_section,
                        original_entity_id=f"{section_name}:{index}",
                        original_name=revised_name,
                        revised_name=revised_name,
                        normalized_name=normalize_text_key(revised_name),
                        payload=serialized_entry,
                        schema_name=REVISION_DRUG_SCHEMA_NAME,
                        requires_human_review=not bool(revised_name),
                    )
                    db_session.add(row)
                    created_rows.append(row)
            disease_entries = structured_case.get("anamnesis_diseases")
            if isinstance(disease_entries, list):
                for index, entry in enumerate(disease_entries):
                    if not isinstance(entry, dict):
                        continue
                    validated_entry = validate_revised_disease_payload(entry)
                    serialized_entry = validated_entry.model_dump(exclude_none=True)
                    revised_name = validated_entry.name
                    row = _create_revision_entity_row(
                        self,
                        revision_version_id=safe_revision_version_id,
                        source_version_id=(
                            int(source_version_id)
                            if source_version_id is not None
                            else None
                        ),
                        pipeline_run_id=safe_pipeline_run_id,
                        step_name="generate_revision",
                        entity_type="disease",
                        source_section="anamnesis",
                        original_entity_id=f"anamnesis_diseases:{index}",
                        original_name=revised_name,
                        revised_name=revised_name,
                        normalized_name=normalize_text_key(revised_name),
                        payload=serialized_entry,
                        schema_name=REVISION_DISEASE_SCHEMA_NAME,
                        requires_human_review=not bool(revised_name),
                    )
                    db_session.add(row)
                    created_rows.append(row)

        lab_entries = result_payload.get("lab_timeline")
        if isinstance(lab_entries, list):
            for index, entry in enumerate(lab_entries):
                if not isinstance(entry, dict):
                    continue
                validated_entry = validate_revised_lab_payload(entry)
                serialized_entry = validated_entry.model_dump(exclude_none=True)
                revised_name = validated_entry.marker_name
                row = _create_revision_entity_row(
                    self,
                    revision_version_id=safe_revision_version_id,
                    source_version_id=(
                        int(source_version_id)
                        if source_version_id is not None
                        else None
                    ),
                    pipeline_run_id=safe_pipeline_run_id,
                    step_name="generate_revision",
                    entity_type="lab_timeline_entry",
                    source_section="laboratory_analysis",
                    original_entity_id=f"lab_timeline:{index}",
                    original_name=revised_name,
                    revised_name=revised_name,
                    normalized_name=normalize_text_key(revised_name),
                    payload=serialized_entry,
                    schema_name=REVISION_LAB_SCHEMA_NAME,
                    requires_human_review=not bool(revised_name),
                )
                db_session.add(row)
                created_rows.append(row)

        revision_payload = result_payload.get("revision")
        if isinstance(revision_payload, dict):
            livertox_decisions = revision_payload.get("livertox_revision_decisions")
            if isinstance(livertox_decisions, list):
                for index, entry in enumerate(livertox_decisions):
                    if not isinstance(entry, dict):
                        continue
                    validated_entry = validate_revision_livertox_decision(entry)
                    serialized_entry = validated_entry.model_dump(exclude_none=True)
                    revised_name = validated_entry.drug_name
                    row = _create_revision_entity_row(
                        self,
                        revision_version_id=safe_revision_version_id,
                        source_version_id=(
                            int(source_version_id)
                            if source_version_id is not None
                            else None
                        ),
                        pipeline_run_id=safe_pipeline_run_id,
                        step_name="resolve_livertox_matches",
                        entity_type="livertox_match",
                        source_section="therapy",
                        original_entity_id=validated_entry.decision_id
                        or f"livertox:{index}",
                        original_name=revised_name,
                        revised_name=revised_name,
                        normalized_name=normalize_text_key(revised_name),
                        payload=serialized_entry,
                        schema_name=REVISION_LIVERTOX_DECISION_SCHEMA_NAME,
                        entity_revision_status=validated_entry.decision,
                        requires_human_review=validated_entry.requires_human_review,
                    )
                    db_session.add(row)
                    created_rows.append(row)
            revised_dili_assessments = revision_payload.get("revised_dili_assessments")
            if isinstance(revised_dili_assessments, list):
                for index, entry in enumerate(revised_dili_assessments):
                    if not isinstance(entry, dict):
                        continue
                    validated_entry = validate_revised_dili_assessment(entry)
                    serialized_entry = validated_entry.model_dump(exclude_none=True)
                    revised_name = validated_entry.drug_name
                    row = _create_revision_entity_row(
                        self,
                        revision_version_id=safe_revision_version_id,
                        source_version_id=(
                            int(source_version_id)
                            if source_version_id is not None
                            else None
                        ),
                        pipeline_run_id=safe_pipeline_run_id,
                        step_name="rerun_dili_assessments",
                        entity_type="dili_assessment",
                        source_section="therapy",
                        original_entity_id=validated_entry.revised_drug_entry_id
                        or f"dili:{index}",
                        original_name=revised_name,
                        revised_name=revised_name,
                        normalized_name=normalize_text_key(revised_name),
                        payload=serialized_entry,
                        schema_name=REVISION_DILI_ASSESSMENT_SCHEMA_NAME,
                        entity_revision_status="active",
                        requires_human_review=validated_entry.requires_human_review,
                    )
                    db_session.add(row)
                    created_rows.append(row)

        db_session.flush()
        db_session.commit()
        return [serialize_revision_entity_row(self, row) for row in created_rows]
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()

###############################################################################
def list_revision_artifacts_for_version(
    self,
    *,
    revision_version_id: int,
) -> list[dict[str, Any]]:
    db_session = self.session_factory()
    try:
        rows = db_session.execute(
            select(ClinicalSessionRevisionArtifact)
            .where(
                ClinicalSessionRevisionArtifact.revision_version_id
                == int(revision_version_id)
            )
            .order_by(
                ClinicalSessionRevisionArtifact.artifact_kind.asc(),
                ClinicalSessionRevisionArtifact.entity_type.asc(),
                ClinicalSessionRevisionArtifact.entity_name.asc(),
                ClinicalSessionRevisionArtifact.id.asc(),
            )
        ).scalars()
        return [serialize_revision_artifact_row(self, row) for row in rows]
    finally:
        db_session.close()

###############################################################################
def list_revision_entities_for_version(
    self,
    *,
    revision_version_id: int,
) -> list[dict[str, Any]]:
    db_session = self.session_factory()
    try:
        rows = db_session.execute(
            select(ClinicalSessionRevisionArtifact)
            .where(
                ClinicalSessionRevisionArtifact.revision_version_id
                == int(revision_version_id),
                ClinicalSessionRevisionArtifact.artifact_kind
                == "structured_case_entity",
                ClinicalSessionRevisionArtifact.status != "superseded",
            )
            .order_by(
                ClinicalSessionRevisionArtifact.entity_type.asc(),
                ClinicalSessionRevisionArtifact.entity_name.asc(),
                ClinicalSessionRevisionArtifact.id.asc(),
            )
        ).scalars()
        return [serialize_revision_entity_row(self, row) for row in rows]
    finally:
        db_session.close()

###############################################################################
def persist_revision_artifact(
    self,
    *,
    pipeline_run_id: str,
    revision_version_id: int,
    artifact_key: str,
    payload: dict[str, Any],
    status: str = "derived",
) -> list[dict[str, Any]]:
    db_session = self.session_factory()
    try:
        row = _create_revision_artifact_row(
            self,
            revision_version_id=int(revision_version_id),
            pipeline_run_id=str(pipeline_run_id),
            artifact_kind="pipeline_artifact",
            artifact_key=artifact_key,
            status=status,
            payload=payload,
        )
        db_session.add(row)
        db_session.flush()
        db_session.commit()
        return [serialize_revision_artifact_row(self, row)]
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()
