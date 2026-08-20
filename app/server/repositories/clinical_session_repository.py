from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import and_, delete, exists, func, or_, select
from sqlalchemy.orm import Session

from common.utils.logger import logger
from common.utils.text_utils import normalize_drug_name
from repositories import values as repository_values
from repositories.context import RepositoryContext
from repositories.schemas.clinical import (
    ClinicalDrugMention,
    ClinicalLabObservation,
    ClinicalSession,
    ClinicalSessionResult,
    ClinicalSessionRevisionArtifact,
    ClinicalSessionRevisionReview,
    ClinicalSessionRevisionRun,
    ClinicalSessionRevisionStep,
    ClinicalSessionSection,
    ClinicalSessionTimeline,
    ClinicalSessionVersion,
)
from repositories.serialization import session_result_data

###############################################################################
def _build_search_pattern(value: str | None) -> str | None:
    normalized = repository_values.normalize_string(value)
    if normalized is None:
        return None
    escaped = re.sub(r"([%_\\])", r"\\\1", normalized.casefold())
    return f"%{escaped}%"

###############################################################################
class ClinicalSessionRepository:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        context: RepositoryContext,
    ) -> None:
        self.context = context
        self.engine = context.engine
        self.session_factory = context.session_factory

    # -------------------------------------------------------------------------
    def save_clinical_session(self, session_data: dict[str, Any]) -> int | None:
        if not session_data:
            logger.warning("Skipping clinical session save; payload is empty")
            return None
        db_session = self.session_factory()
        try:
            persisted_session = ClinicalSession(
                patient_name=repository_values.normalize_string(
                    session_data.get("patient_name")
                ),
                visit_date=repository_values.normalize_date_value(
                    session_data.get("patient_visit_date")
                ),
                anamnesis=repository_values.normalize_string(session_data.get("anamnesis")),
                drugs_text=repository_values.normalize_string(session_data.get("drugs")),
                laboratory_analysis=repository_values.normalize_string(
                    session_data.get("laboratory_analysis")
                ),
                patient_image_blob=session_result_data.decode_patient_image(
                    session_data.get("patient_image_base64")
                ),
                session_timestamp=session_result_data.parse_datetime(
                    session_data.get("session_timestamp")
                ),
                hepatic_pattern=repository_values.normalize_string(
                    session_data.get("hepatic_pattern")
                ),
                text_extraction_model=repository_values.normalize_string(
                    session_data.get("text_extraction_model")
                ),
                clinical_model=repository_values.normalize_string(
                    session_data.get("clinical_model")
                ),
                total_duration=repository_values.to_float(session_data.get("total_duration")),
                session_status=repository_values.normalize_session_status(
                    session_data.get("session_status")
                ),
                session_kind=repository_values.normalize_string(session_data.get("session_kind")),
                metadata_json=session_result_data.serialize_json_payload(
                    session_data.get("metadata")
                ),
            )
            db_session.add(persisted_session)
            db_session.flush()
            session_id = int(persisted_session.id)
            if session_data.get("root_session_id") is None:
                db_session.add(
                    ClinicalSessionVersion(
                        session_id=session_id,
                        root_session_id=session_id,
                        source_version_id=None,
                        version_number=1,
                        version_status="current",
                        revision_kind="original",
                        llm_qa_status="not_run",
                        clinical_review_status="not_reviewed",
                        pipeline_run_id=None,
                        model_configuration_json=session_result_data.serialize_json_payload(
                            {
                                "text_extraction_model": persisted_session.text_extraction_model,
                                "clinical_model": persisted_session.clinical_model,
                            }
                        ),
                        completed_at=persisted_session.session_timestamp,
                    )
                )
            self.persist_session_sections(db_session, session_id, session_data)
            self.persist_session_labs(db_session, session_id, session_data)
            self.persist_session_drugs(db_session, session_id, session_data)
            self.persist_session_result_payload(db_session, session_id, session_data)
            db_session.commit()
            return session_id
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    # -------------------------------------------------------------------------
    def list_sessions(
        self,
        *,
        search: str | None,
        status_filter: str | None,
        date_mode: str | None,
        filter_date: date | None,
        offset: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        safe_offset = max(int(offset), 0)
        safe_limit = max(int(limit), 1)
        conditions: list[Any] = []
        search_pattern = _build_search_pattern(search)
        if search_pattern is not None:
            section_match = exists(
                select(1).where(
                    ClinicalSessionSection.session_id == ClinicalSession.id,
                    func.lower(func.coalesce(ClinicalSessionSection.content, "")).like(
                        search_pattern, escape="\\"
                    ),
                )
            )
            result_payload_match = exists(
                select(1).where(
                    ClinicalSessionResult.session_id == ClinicalSession.id,
                    func.lower(func.coalesce(ClinicalSessionResult.payload_json, "")).like(
                        search_pattern, escape="\\"
                    ),
                )
            )
            conditions.append(
                or_(
                    func.lower(func.coalesce(ClinicalSession.patient_name, "")).like(
                        search_pattern, escape="\\"
                    ),
                    section_match,
                    result_payload_match,
                )
            )
        normalized_status_filter = (
            status_filter.casefold() if isinstance(status_filter, str) else None
        )
        if normalized_status_filter in {"successful", "failed"}:
            conditions.append(
                func.lower(func.coalesce(ClinicalSession.session_status, "successful"))
                == normalized_status_filter
            )
        if filter_date is not None and date_mode in {"before", "after", "exact"}:
            day_start = datetime.combine(filter_date, datetime.min.time())
            next_day = day_start + timedelta(days=1)
            if date_mode == "before":
                conditions.append(ClinicalSession.session_timestamp < day_start)
            elif date_mode == "after":
                conditions.append(ClinicalSession.session_timestamp >= next_day)
            else:
                conditions.extend(
                    [
                        ClinicalSession.session_timestamp >= day_start,
                        ClinicalSession.session_timestamp < next_day,
                    ]
                )
        db_session = self.session_factory()
        try:
            report_exists = exists(
                select(1).where(
                    ClinicalSessionVersion.session_id == ClinicalSession.id,
                    ClinicalSessionVersion.report_text.isnot(None),
                )
            )
            timeline_exists = exists(
                select(1).where(ClinicalSessionTimeline.session_id == ClinicalSession.id)
            )
            version_number = (
                select(func.max(ClinicalSessionVersion.version_number))
                .where(ClinicalSessionVersion.session_id == ClinicalSession.id)
                .scalar_subquery()
            )
            sessions_stmt = select(
                ClinicalSession,
                version_number.label("version_number"),
                report_exists.label("has_report"),
                timeline_exists.label("has_timeline"),
            )
            count_stmt = select(func.count()).select_from(ClinicalSession)
            if conditions:
                combined = and_(*conditions)
                sessions_stmt = sessions_stmt.where(combined)
                count_stmt = count_stmt.where(combined)
            total_rows = int(db_session.execute(count_stmt).scalar_one())
            rows = db_session.execute(
                sessions_stmt.order_by(
                    ClinicalSession.session_timestamp.desc(), ClinicalSession.id.desc()
                )
                .offset(safe_offset)
                .limit(safe_limit)
            ).all()
            items = [
                {
                    "session_id": int(session_row.id),
                    "patient_name": repository_values.normalize_string(session_row.patient_name),
                    "session_timestamp": session_row.session_timestamp,
                    "version": int(version_value or 1),
                    "status": repository_values.normalize_session_status(
                        session_row.session_status
                    ),
                    "total_duration": repository_values.to_float(session_row.total_duration),
                    "has_report": bool(has_report),
                    "has_timeline": bool(has_timeline),
                    "can_generate_timeline": bool(
                        repository_values.normalize_string(session_row.anamnesis)
                        or repository_values.normalize_string(session_row.drugs_text)
                        or repository_values.normalize_string(session_row.laboratory_analysis)
                    ),
                }
                for session_row, version_value, has_report, has_timeline in rows
            ]
            return items, total_rows
        finally:
            db_session.close()

    # -------------------------------------------------------------------------
    def get_session_detail(self, session_id: int) -> dict[str, Any] | None:
        safe_session_id = int(session_id)
        db_session = self.session_factory()
        try:
            session_row = db_session.execute(
                select(ClinicalSession).where(ClinicalSession.id == safe_session_id)
            ).scalar_one_or_none()
            if session_row is None:
                return None
            version_number = db_session.execute(
                select(func.max(ClinicalSessionVersion.version_number)).where(
                    ClinicalSessionVersion.session_id == safe_session_id
                )
            ).scalar_one_or_none()
            section_rows = db_session.execute(
                select(ClinicalSessionSection.section_kind, ClinicalSessionSection.content).where(
                    ClinicalSessionSection.session_id == safe_session_id
                )
            ).all()
            sections = {
                str(kind): repository_values.normalize_string(content) or ""
                for kind, content in section_rows
            }
            payload = self.get_session_result_payload(safe_session_id) or {}
            metadata = session_result_data.parse_session_result_payload(
                session_row.metadata_json
            ) or {}
            session_text = repository_values.normalize_string(
                payload.get("original_session_text")
            ) or ""
            official_report_text = repository_values.normalize_string(payload.get("report"))
            return {
                "session_id": safe_session_id,
                "patient_name": repository_values.normalize_string(session_row.patient_name),
                "visit_date": session_row.visit_date,
                "session_timestamp": session_row.session_timestamp,
                "version": int(version_number or 1),
                "status": repository_values.normalize_session_status(session_row.session_status),
                "text_extraction_model": repository_values.normalize_string(
                    session_row.text_extraction_model
                ),
                "clinical_model": repository_values.normalize_string(session_row.clinical_model),
                "metadata": metadata,
                "sections": sections,
                "session_text": session_text,
                "source_clinical_text": session_text,
                "result_payload": payload,
                "report": official_report_text,
                "official_report_text": official_report_text,
            }
        finally:
            db_session.close()

    # -------------------------------------------------------------------------
    def get_session_result_payload(self, session_id: int) -> dict[str, Any] | None:
        with self.session_factory() as db_session:
            payload_json = db_session.execute(
                select(ClinicalSessionResult.payload_json).where(
                    ClinicalSessionResult.session_id == int(session_id)
                )
            ).scalar_one_or_none()
            return session_result_data.parse_session_result_payload(payload_json)

    # -------------------------------------------------------------------------
    def upsert_session_result_payload(
        self, session_id: int, payload: dict[str, Any]
    ) -> bool:
        serialized_payload = session_result_data.serialize_json_payload(payload)
        if serialized_payload is None:
            return False
        with self.session_factory() as db_session:
            existing_session = db_session.get(ClinicalSession, int(session_id))
            if existing_session is None:
                return False
            existing_result = db_session.execute(
                select(ClinicalSessionResult).where(
                    ClinicalSessionResult.session_id == int(session_id)
                )
            ).scalar_one_or_none()
            if existing_result is None:
                db_session.add(
                    ClinicalSessionResult(
                        session_id=int(session_id), payload_json=serialized_payload
                    )
                )
            else:
                existing_result.payload_json = serialized_payload
            db_session.commit()
            return True

    # -------------------------------------------------------------------------
    def update_session_text_and_metadata(
        self,
        session_id: int,
        *,
        session_text: str | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        safe_session_id = int(session_id)
        with self.session_factory() as db_session:
            existing = db_session.get(ClinicalSession, safe_session_id)
            if existing is None:
                return None
            if session_text is not None:
                result_row = db_session.execute(
                    select(ClinicalSessionResult).where(
                        ClinicalSessionResult.session_id == safe_session_id
                    )
                ).scalar_one_or_none()
                payload = (
                    session_result_data.parse_session_result_payload(result_row.payload_json)
                    if result_row is not None
                    else {}
                ) or {}
                payload["original_session_text"] = str(session_text).strip()
                serialized_payload = session_result_data.serialize_json_payload(payload) or "{}"
                if result_row is None:
                    db_session.add(
                        ClinicalSessionResult(
                            session_id=safe_session_id, payload_json=serialized_payload
                        )
                    )
                else:
                    result_row.payload_json = serialized_payload
            if metadata is not None:
                existing.metadata_json = session_result_data.serialize_json_payload(metadata or {})
            db_session.commit()
        return self.get_session_detail(safe_session_id)

    # -------------------------------------------------------------------------
    def update_session_metadata(
        self, session_id: int, *, metadata: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        safe_session_id = int(session_id)
        with self.session_factory() as db_session:
            existing = db_session.get(ClinicalSession, safe_session_id)
            if existing is None:
                return None
            if metadata is not None:
                existing.metadata_json = session_result_data.serialize_json_payload(metadata or {})
            db_session.commit()
        return self.get_session_detail(safe_session_id)

    # -------------------------------------------------------------------------
    def get_next_session_version(self, root_session_id: int) -> int:
        with self.session_factory() as db_session:
            max_version = db_session.execute(
                select(func.max(ClinicalSessionVersion.version_number)).where(
                    ClinicalSessionVersion.root_session_id == int(root_session_id)
                )
            ).scalar_one_or_none()
            return int(max_version or 1) + 1

    # -------------------------------------------------------------------------
    def delete_session(self, session_id: int) -> bool:
        safe_session_id = int(session_id)
        with self.session_factory() as db_session:
            existing = db_session.get(ClinicalSession, safe_session_id)
            if existing is None:
                return False
            root_session_id = db_session.execute(
                select(ClinicalSessionVersion.root_session_id)
                .where(ClinicalSessionVersion.session_id == safe_session_id)
                .order_by(ClinicalSessionVersion.version_number.desc())
                .limit(1)
            ).scalar_one_or_none()
            is_root_session = int(root_session_id or safe_session_id) == safe_session_id
            version_scope = (
                ClinicalSessionVersion.root_session_id == safe_session_id
                if is_root_session
                else ClinicalSessionVersion.session_id == safe_session_id
            )
            version_ids = list(
                db_session.execute(select(ClinicalSessionVersion.id).where(version_scope)).scalars()
            )
            run_scope = (
                ClinicalSessionRevisionRun.root_session_id == safe_session_id
                if is_root_session
                else ClinicalSessionRevisionRun.session_id == safe_session_id
            )
            run_ids = list(
                db_session.execute(select(ClinicalSessionRevisionRun.id).where(run_scope)).scalars()
            )
            if version_ids:
                db_session.execute(
                    delete(ClinicalSessionRevisionReview).where(
                        ClinicalSessionRevisionReview.revision_version_id.in_(version_ids)
                    )
                )
                db_session.execute(
                    delete(ClinicalSessionRevisionArtifact).where(
                        ClinicalSessionRevisionArtifact.revision_version_id.in_(version_ids)
                    )
                )
            if run_ids:
                db_session.execute(
                    delete(ClinicalSessionRevisionStep).where(
                        ClinicalSessionRevisionStep.revision_run_id.in_(run_ids)
                    )
                )
                db_session.execute(
                    delete(ClinicalSessionRevisionArtifact).where(
                        ClinicalSessionRevisionArtifact.revision_run_id.in_(run_ids)
                    )
                )
            db_session.execute(delete(ClinicalSessionRevisionRun).where(run_scope))
            if version_ids:
                db_session.execute(delete(ClinicalSessionVersion).where(version_scope))
            db_session.delete(existing)
            db_session.commit()
            return True

    # -------------------------------------------------------------------------
    def persist_session_sections(
        self, db_session: Session, session_id: int, session_data: dict[str, Any]
    ) -> None:
        issues_raw = session_data.get("issues")
        if isinstance(issues_raw, (list, dict)):
            issues_content: str | None = json.dumps(issues_raw, ensure_ascii=False)
        else:
            issues_content = repository_values.normalize_string(issues_raw)
        payload = {
            "anamnesis": session_data.get("anamnesis"),
            "drugs": session_data.get("drugs"),
            "laboratory_analysis": session_data.get("laboratory_analysis"),
            "final_report": session_data.get("final_report"),
            "issues": issues_content,
        }
        for section_kind, value in payload.items():
            content = repository_values.normalize_string(value)
            if content is not None:
                db_session.add(
                    ClinicalSessionSection(
                        session_id=session_id, section_kind=section_kind, content=content
                    )
                )

    # -------------------------------------------------------------------------
    def persist_session_labs(
        self, db_session: Session, session_id: int, session_data: dict[str, Any]
    ) -> None:
        result_payload = session_data.get("session_result_payload")
        timeline_raw = result_payload.get("lab_timeline") if isinstance(result_payload, dict) else None
        if not isinstance(timeline_raw, list):
            return
        persisted_codes = {
            "ALT": "alt",
            "AST": "ast",
            "ALP": "alp",
            "TBIL": "tbil",
            "DBIL": "dbil",
            "GGT": "ggt",
            "INR": "inr",
            "ALB": "albumin",
        }
        for ordinal, item in enumerate(timeline_raw):
            if not isinstance(item, dict):
                continue
            marker_name = repository_values.normalize_string(item.get("marker_name"))
            lab_code = persisted_codes.get(marker_name.upper()) if marker_name else None
            if lab_code is None:
                continue
            value_raw = repository_values.normalize_string(item.get("value")) or repository_values.normalize_string(item.get("value_text"))
            upper_limit_raw = repository_values.normalize_string(item.get("upper_limit_normal")) or repository_values.normalize_string(item.get("upper_limit_text"))
            if value_raw is None and upper_limit_raw is None:
                continue
            db_session.add(
                ClinicalLabObservation(
                    session_id=session_id,
                    marker_code=lab_code,
                    observation_at=session_result_data.parse_datetime(
                        item.get("observation_at") or item.get("date")
                    ),
                    value_numeric=repository_values.to_float(value_raw),
                    value_text=value_raw,
                    unit=repository_values.normalize_string(item.get("unit")),
                    upper_limit_numeric=repository_values.to_float(upper_limit_raw),
                    source_ordinal=ordinal,
                    metadata_json={
                        key: value
                        for key, value in item.items()
                        if key not in {"value", "value_text", "upper_limit_normal", "upper_limit_text"}
                    },
                )
            )

    # -------------------------------------------------------------------------
    def persist_session_drugs(
        self, db_session: Session, session_id: int, session_data: dict[str, Any]
    ) -> None:
        payload = session_data.get("matched_drugs")
        records: list[dict[str, Any]] = []
        if isinstance(payload, list):
            records.extend(
                item if isinstance(item, dict) else {"raw_drug_name": item}
                for item in payload
                if isinstance(item, (dict, str))
            )
        if not records and isinstance(session_data.get("detected_drugs"), list):
            records.extend(
                {"raw_drug_name": item}
                for item in session_data["detected_drugs"]
                if isinstance(item, str)
            )
        seen: set[str] = set()
        for mention_ordinal, item in enumerate(records):
            raw_drug_name = repository_values.normalize_string(
                item.get("raw_drug_name") or item.get("name")
            )
            if raw_drug_name is None:
                continue
            raw_drug_name_norm = normalize_drug_name(raw_drug_name)
            duplicate_mention = not raw_drug_name_norm or raw_drug_name_norm in seen
            seen.add(raw_drug_name_norm)
            matched_drug_name = repository_values.normalize_string(item.get("matched_drug_name"))
            rxcui = repository_values.normalize_string(
                item.get("rxcui") or item.get("rxnorm_rxcui")
            )
            nbk_id = repository_values.normalize_string(item.get("nbk_id"))
            resolved_drug_id = repository_values.to_int(item.get("drug_id"))
            match_reason = repository_values.normalize_string(item.get("match_reason"))
            match_confidence = repository_values.to_float(item.get("match_confidence"))
            db_session.add(
                ClinicalDrugMention(
                    session_id=session_id,
                    mention_ordinal=mention_ordinal,
                    raw_name=raw_drug_name,
                    normalized_name=raw_drug_name_norm,
                    drug_id=resolved_drug_id,
                    match_status=(
                        "ambiguous"
                        if bool(item.get("ambiguous_match"))
                        else "matched" if resolved_drug_id is not None else "unresolved"
                    ),
                    confidence=match_confidence,
                    match_reason=match_reason,
                    evidence_json={
                        "matched_drug_name": matched_drug_name,
                        "rxcui": rxcui,
                        "nbk_id": nbk_id,
                        "duplicate_mention": duplicate_mention,
                    },
                )
            )

    # -------------------------------------------------------------------------
    def persist_session_result_payload(
        self, db_session: Session, session_id: int, session_data: dict[str, Any]
    ) -> None:
        payload = session_data.get("session_result_payload")
        serialized_payload = session_result_data.serialize_json_payload(payload)
        if serialized_payload is None:
            return
        db_session.add(
            ClinicalSessionResult(session_id=session_id, payload_json=serialized_payload)
        )
        current_version = db_session.execute(
            select(ClinicalSessionVersion)
            .where(ClinicalSessionVersion.session_id == int(session_id))
            .order_by(ClinicalSessionVersion.version_number.desc())
        ).scalars().first()
        if current_version is not None and isinstance(payload, dict):
            current_version.report_text = repository_values.normalize_string(payload.get("report"))
            current_version.hepatic_pattern = repository_values.normalize_string(payload.get("hepatic_pattern"))
            current_version.total_duration = repository_values.to_float(payload.get("total_duration"))
            current_version.metadata_json = session_result_data.serialize_json_payload(payload.get("metadata"))
