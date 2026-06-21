from __future__ import annotations

import difflib
from typing import Any

from services.inspection.normalization import normalize_text as _normalize_text_value


###############################################################################
def _append_derived_revision_entity(
    *,
    derived: list[dict[str, Any]],
    session_detail: dict[str, Any],
    version_summary: dict[str, Any],
    revision_version_id: int,
    source_version_id: Any,
    pipeline_run_id: str,
    entity_type: str,
    source_section: str,
    original_entity_id: str,
    revised_name: str | None,
    payload: dict[str, Any],
    entity_revision_status: str = "active",
    requires_human_review: bool = False,
    step_name: str = "persisted_session_result",
) -> None:
    normalized_name = _normalize_text_value(revised_name)
    derived.append(
        {
            "revision_version_id": revision_version_id,
            "source_version_id": int(source_version_id)
            if source_version_id is not None
            else None,
            "pipeline_run_id": pipeline_run_id,
            "step_name": step_name,
            "entity_type": entity_type,
            "entity_revision_status": entity_revision_status,
            "source_section": source_section,
            "original_entity_id": original_entity_id,
            "original_name": revised_name,
            "revised_name": revised_name,
            "normalized_name": normalized_name or None,
            "requires_human_review": requires_human_review,
            "human_review_status": (
                "required" if requires_human_review else "not_required"
            ),
            "payload": payload,
            "schema_name": "revision_entity",
            "schema_version": "1",
            "prompt_version": None,
            "parser_version": None,
            "model_provider": None,
            "model_name": None,
            "input_hash": None,
            "output_hash": None,
            "created_at": session_detail.get("session_timestamp")
            or version_summary.get("updated_at"),
            "superseded_at": None,
        }
    )


###############################################################################
class InspectionRevisionDiffMixin:
    # -------------------------------------------------------------------------
    def compare_session_versions(
        self,
        session_id: int,
        *,
        left_version_id: int,
        right_version_id: int,
    ) -> dict[str, Any] | None:
        left_detail = self.get_session_version_detail(
            session_id,
            version_id=left_version_id,
        )
        right_detail = self.get_session_version_detail(
            session_id,
            version_id=right_version_id,
        )
        if left_detail is None or right_detail is None:
            return None

        left_version = left_detail.get("version") or {}
        right_version = right_detail.get("version") or {}
        if int(left_version.get("root_session_id") or 0) != int(
            right_version.get("root_session_id") or 0
        ):
            raise ValueError("Versions do not belong to the same session lineage.")

        left_entities = self._resolve_version_comparison_entities(
            version_id=left_version_id,
            detail=left_detail,
        )
        right_entities = self._resolve_version_comparison_entities(
            version_id=right_version_id,
            detail=right_detail,
        )
        entity_diff = self._build_version_entity_diff(
            left_entities=left_entities,
            right_entities=right_entities,
        )
        return {
            "left_version": left_version,
            "right_version": right_version,
            **entity_diff,
            "report_text_diff": self._build_report_text_diff(
                left_text=self._extract_version_report_text(left_detail),
                right_text=self._extract_version_report_text(right_detail),
            ),
            "qa_summary": self._build_revision_qa_summary(
                left_detail=left_detail,
                right_detail=right_detail,
            ),
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def _extract_version_report_text(detail: dict[str, Any]) -> str:
        session_detail = detail.get("session")
        if not isinstance(session_detail, dict):
            return ""
        report_text = (
            session_detail.get("official_report_text")
            or session_detail.get("report")
            or (session_detail.get("result_payload") or {}).get("report")
            or ""
        )
        return str(report_text).strip()

    # -------------------------------------------------------------------------
    def _resolve_version_comparison_entities(
        self,
        *,
        version_id: int,
        detail: dict[str, Any],
    ) -> list[dict[str, Any]]:
        persisted_entities = self.list_revision_entities(
            revision_version_id=version_id,
        )
        if persisted_entities:
            return persisted_entities
        return self._derive_entities_from_version_detail(detail)

    # -------------------------------------------------------------------------
    def _derive_entities_from_version_detail(
        self,
        detail: dict[str, Any],
    ) -> list[dict[str, Any]]:
        session_detail = detail.get("session")
        version_summary = detail.get("version")
        if not isinstance(session_detail, dict) or not isinstance(
            version_summary, dict
        ):
            return []
        result_payload = session_detail.get("result_payload")
        if not isinstance(result_payload, dict):
            result_payload = {}

        revision_version_id = int(version_summary.get("version_id") or 0)
        source_version_id = version_summary.get("source_version_id")
        pipeline_run_id = str(version_summary.get("pipeline_run_id") or "")
        derived: list[dict[str, Any]] = []

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
                    revised_name = (
                        str(entry.get("name") or entry.get("drug_name") or "").strip()
                        or None
                    )
                    _append_derived_revision_entity(
                        derived=derived,
                        session_detail=session_detail,
                        version_summary=version_summary,
                        revision_version_id=revision_version_id,
                        source_version_id=source_version_id,
                        pipeline_run_id=pipeline_run_id,
                        entity_type="drug",
                        source_section=source_section,
                        original_entity_id=f"{section_name}:{index}",
                        revised_name=revised_name,
                        payload=entry,
                        requires_human_review=not bool(revised_name),
                    )
            diseases = structured_case.get("anamnesis_diseases")
            if isinstance(diseases, list):
                for index, entry in enumerate(diseases):
                    if not isinstance(entry, dict):
                        continue
                    revised_name = str(entry.get("name") or "").strip() or None
                    _append_derived_revision_entity(
                        derived=derived,
                        session_detail=session_detail,
                        version_summary=version_summary,
                        revision_version_id=revision_version_id,
                        source_version_id=source_version_id,
                        pipeline_run_id=pipeline_run_id,
                        entity_type="disease",
                        source_section="anamnesis",
                        original_entity_id=f"anamnesis_diseases:{index}",
                        revised_name=revised_name,
                        payload=entry,
                        requires_human_review=not bool(revised_name),
                    )

        lab_timeline = result_payload.get("lab_timeline")
        if isinstance(lab_timeline, list):
            for index, entry in enumerate(lab_timeline):
                if not isinstance(entry, dict):
                    continue
                revised_name = str(entry.get("marker_name") or "").strip() or None
                _append_derived_revision_entity(
                    derived=derived,
                    session_detail=session_detail,
                    version_summary=version_summary,
                    revision_version_id=revision_version_id,
                    source_version_id=source_version_id,
                    pipeline_run_id=pipeline_run_id,
                    entity_type="lab_timeline_entry",
                    source_section="laboratory_analysis",
                    original_entity_id=f"lab_timeline:{index}",
                    revised_name=revised_name,
                    payload=entry,
                    requires_human_review=not bool(revised_name),
                )

        matched_drugs = result_payload.get("matched_drugs")
        if isinstance(matched_drugs, list):
            for index, entry in enumerate(matched_drugs):
                if not isinstance(entry, dict):
                    continue
                revised_name = (
                    str(
                        entry.get("matched_drug_name")
                        or entry.get("raw_drug_name")
                        or ""
                    ).strip()
                    or None
                )
                _append_derived_revision_entity(
                    derived=derived,
                    session_detail=session_detail,
                    version_summary=version_summary,
                    revision_version_id=revision_version_id,
                    source_version_id=source_version_id,
                    pipeline_run_id=pipeline_run_id,
                    entity_type="livertox_match",
                    source_section="therapy",
                    original_entity_id=f"matched_drug:{index}",
                    revised_name=revised_name,
                    payload=entry,
                    entity_revision_status=str(entry.get("match_status") or "active"),
                    requires_human_review=bool(entry.get("requires_human_review")),
                )

        rucam_assessments = result_payload.get("rucam_assessments")
        if isinstance(rucam_assessments, list):
            for index, entry in enumerate(rucam_assessments):
                if not isinstance(entry, dict):
                    continue
                revised_name = str(entry.get("drug_name") or "").strip() or None
                _append_derived_revision_entity(
                    derived=derived,
                    session_detail=session_detail,
                    version_summary=version_summary,
                    revision_version_id=revision_version_id,
                    source_version_id=source_version_id,
                    pipeline_run_id=pipeline_run_id,
                    entity_type="dili_assessment",
                    source_section="therapy",
                    original_entity_id=f"rucam_assessment:{index}",
                    revised_name=revised_name,
                    payload=entry,
                    requires_human_review=bool(entry.get("requires_human_review")),
                )
        return derived

    # -------------------------------------------------------------------------
    @staticmethod
    def _comparison_entity_key(entity: dict[str, Any]) -> tuple[str, str, str]:
        entity_type = str(entity.get("entity_type") or "").strip()
        normalized_name = str(
            entity.get("normalized_name")
            or entity.get("revised_name")
            or entity.get("original_name")
            or entity.get("original_entity_id")
            or ""
        ).strip()
        source_section = str(entity.get("source_section") or "").strip()
        return entity_type, normalized_name, source_section

    # -------------------------------------------------------------------------
    @classmethod
    def _build_entity_diff_item(
        cls,
        *,
        change_type: str,
        left_entity: dict[str, Any] | None,
        right_entity: dict[str, Any] | None,
    ) -> dict[str, Any]:
        reference = right_entity if right_entity is not None else left_entity or {}
        entity_type = str(reference.get("entity_type") or "").strip()
        normalized_name = str(reference.get("normalized_name") or "").strip() or None
        source_section = str(reference.get("source_section") or "").strip() or None
        revised_name = str(
            reference.get("revised_name") or normalized_name or ""
        ).strip()
        if not revised_name:
            revised_name = entity_type or "entity"
        summary = f"{revised_name} ({entity_type or 'entity'})"
        return {
            "entity_type": entity_type,
            "normalized_name": normalized_name,
            "source_section": source_section,
            "change_type": change_type,
            "summary": summary,
            "requires_human_review": bool(reference.get("requires_human_review")),
            "left_entity": left_entity,
            "right_entity": right_entity,
        }

    # -------------------------------------------------------------------------
    @classmethod
    def _build_version_entity_diff(
        cls,
        *,
        left_entities: list[dict[str, Any]],
        right_entities: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        left_map = {cls._comparison_entity_key(item): item for item in left_entities}
        right_map = {cls._comparison_entity_key(item): item for item in right_entities}

        added_entities: list[dict[str, Any]] = []
        removed_entities: list[dict[str, Any]] = []
        corrected_entities: list[dict[str, Any]] = []
        replaced_entities: list[dict[str, Any]] = []
        unresolved_entities: list[dict[str, Any]] = []
        unchanged_entities: list[dict[str, Any]] = []

        for key in sorted(set(left_map) | set(right_map)):
            left_entity = left_map.get(key)
            right_entity = right_map.get(key)
            if left_entity is None and right_entity is not None:
                added_entities.append(
                    cls._build_entity_diff_item(
                        change_type="added",
                        left_entity=None,
                        right_entity=right_entity,
                    )
                )
                continue
            if right_entity is None and left_entity is not None:
                removed_entities.append(
                    cls._build_entity_diff_item(
                        change_type="removed",
                        left_entity=left_entity,
                        right_entity=None,
                    )
                )
                continue
            if left_entity is None or right_entity is None:
                continue

            left_payload = (
                left_entity.get("payload") if isinstance(left_entity, dict) else None
            )
            right_payload = (
                right_entity.get("payload") if isinstance(right_entity, dict) else None
            )
            payload_changed = left_payload != right_payload
            right_status = (
                str(right_entity.get("entity_revision_status") or "").strip().casefold()
            )
            if right_entity.get("requires_human_review"):
                unresolved_entities.append(
                    cls._build_entity_diff_item(
                        change_type="unresolved",
                        left_entity=left_entity,
                        right_entity=right_entity,
                    )
                )
            elif payload_changed and "replace" in right_status:
                replaced_entities.append(
                    cls._build_entity_diff_item(
                        change_type="replaced",
                        left_entity=left_entity,
                        right_entity=right_entity,
                    )
                )
            elif payload_changed:
                corrected_entities.append(
                    cls._build_entity_diff_item(
                        change_type="corrected",
                        left_entity=left_entity,
                        right_entity=right_entity,
                    )
                )
            else:
                unchanged_entities.append(
                    cls._build_entity_diff_item(
                        change_type="unchanged",
                        left_entity=left_entity,
                        right_entity=right_entity,
                    )
                )

        return {
            "added_entities": added_entities,
            "removed_entities": removed_entities,
            "corrected_entities": corrected_entities,
            "replaced_entities": replaced_entities,
            "unresolved_entities": unresolved_entities,
            "unchanged_entities": unchanged_entities,
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def _build_report_text_diff(
        *,
        left_text: str,
        right_text: str,
    ) -> dict[str, Any]:
        left_lines = left_text.splitlines()
        right_lines = right_text.splitlines()
        diff_lines = list(
            difflib.unified_diff(
                left_lines,
                right_lines,
                fromfile="left_version",
                tofile="right_version",
                lineterm="",
                n=2,
            )
        )
        return {
            "changed": left_text != right_text,
            "left_character_count": len(left_text),
            "right_character_count": len(right_text),
            "left_line_count": len(left_lines),
            "right_line_count": len(right_lines),
            "similarity_ratio": round(
                difflib.SequenceMatcher(a=left_text, b=right_text).ratio(),
                4,
            ),
            "diff_lines": diff_lines[:80],
        }

    # -------------------------------------------------------------------------
    def _resolve_qa_payload(self, detail: dict[str, Any]) -> dict[str, Any]:
        version = detail.get("version")
        if not isinstance(version, dict):
            return {}
        version_id = int(version.get("version_id") or 0)
        for artifact in self.list_revision_artifacts(revision_version_id=version_id):
            if (
                str(artifact.get("artifact_key") or "").strip()
                == "revision_qa_validation"
            ):
                payload = artifact.get("payload")
                if isinstance(payload, dict):
                    return payload
        session_detail = detail.get("session")
        if not isinstance(session_detail, dict):
            return {}
        result_payload = session_detail.get("result_payload")
        if not isinstance(result_payload, dict):
            return {}
        revision_payload = result_payload.get("revision")
        if isinstance(revision_payload, dict):
            qa_validation = revision_payload.get("qa_validation")
            if isinstance(qa_validation, dict):
                return qa_validation
        return {}

    # -------------------------------------------------------------------------
    def _build_revision_qa_summary(
        self,
        *,
        left_detail: dict[str, Any],
        right_detail: dict[str, Any],
    ) -> dict[str, Any]:
        left_version = left_detail.get("version") or {}
        right_version = right_detail.get("version") or {}
        left_payload = self._resolve_qa_payload(left_detail)
        right_payload = self._resolve_qa_payload(right_detail)
        left_warnings = [
            str(item).strip()
            for item in (left_payload.get("warnings") or [])
            if str(item).strip()
        ]
        right_warnings = [
            str(item).strip()
            for item in (right_payload.get("warnings") or [])
            if str(item).strip()
        ]
        left_blocking = [
            str(item).strip()
            for item in (left_payload.get("blocking_issues") or [])
            if str(item).strip()
        ]
        right_blocking = [
            str(item).strip()
            for item in (right_payload.get("blocking_issues") or [])
            if str(item).strip()
        ]
        left_finding_count = int(
            left_payload.get("finding_count") or len(left_warnings) + len(left_blocking)
        )
        right_finding_count = int(
            right_payload.get("finding_count")
            or len(right_warnings) + len(right_blocking)
        )
        return {
            "left_llm_qa_status": str(left_version.get("llm_qa_status") or "not_run"),
            "right_llm_qa_status": str(right_version.get("llm_qa_status") or "not_run"),
            "left_clinical_review_status": str(
                left_version.get("clinical_review_status") or "not_reviewed"
            ),
            "right_clinical_review_status": str(
                right_version.get("clinical_review_status") or "not_reviewed"
            ),
            "left_version_status": str(left_version.get("version_status") or ""),
            "right_version_status": str(right_version.get("version_status") or ""),
            "left_warning_count": len(left_warnings),
            "right_warning_count": len(right_warnings),
            "left_blocking_issue_count": len(left_blocking),
            "right_blocking_issue_count": len(right_blocking),
            "left_finding_count": left_finding_count,
            "right_finding_count": right_finding_count,
            "manual_review_required": bool(left_payload.get("manual_review_required"))
            or bool(right_payload.get("manual_review_required"))
            or bool(left_blocking)
            or bool(right_blocking),
        }
