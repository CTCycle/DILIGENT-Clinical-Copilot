from __future__ import annotations

import base64
import binascii
import json
import re
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import and_, delete, exists, func, inspect, or_, select
from sqlalchemy.orm import Session

from common.utils.logger import logger
from repositories.schemas.models import (
    ClinicalSession,
    ClinicalSessionDrug,
    ClinicalSessionLab,
    ClinicalSessionResult,
    ClinicalSessionSection,
    Drug,
    DrugAlias,
    DrugRxnormCode,
    KbMatchCache,
    LiverToxMonograph,
    Patient,
)
from repositories.serialization.catalogs import ReferenceCatalogSerializer
from services.text.normalization import normalize_drug_name
from services.text.vocabulary import (
    invalidate_text_normalization_snapshot,
)

# Extracted from the facade module; functions intentionally accept the facade instance.


def save_clinical_session(self, session_data: dict[str, Any]) -> int | None:
    if not session_data:
        logger.warning("Skipping clinical session save; payload is empty")
        return None
    self.ensure_session_result_table()
    db_session = self.session_factory()
    try:
        persisted_patient = self.persist_patient(db_session, session_data)
        persisted_session = ClinicalSession(
            patient_id=int(persisted_patient.id),
            session_timestamp=self.parse_datetime(
                session_data.get("session_timestamp")
            ),
            version=self.to_int(session_data.get("version")) or 1,
            original_session_id=self.to_int(session_data.get("original_session_id")),
            hepatic_pattern=self.normalize_string(session_data.get("hepatic_pattern")),
            text_extraction_model=self.normalize_string(
                session_data.get("text_extraction_model")
            ),
            clinical_model=self.normalize_string(session_data.get("clinical_model")),
            total_duration=self.to_float(session_data.get("total_duration")),
            session_status=self.normalize_session_status(
                session_data.get("session_status")
            ),
            metadata_json=self.serialize_json_payload(session_data.get("metadata")),
        )
        db_session.add(persisted_session)
        db_session.flush()
        session_id = int(persisted_session.id)
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


def ensure_session_result_table(self) -> None:
    inspector = inspect(self.engine)
    required_tables = (
        Patient.__tablename__,
        ClinicalSession.__tablename__,
        ClinicalSessionSection.__tablename__,
        ClinicalSessionLab.__tablename__,
        ClinicalSessionDrug.__tablename__,
        ClinicalSessionResult.__tablename__,
        Drug.__tablename__,
        LiverToxMonograph.__tablename__,
        DrugRxnormCode.__tablename__,
        DrugAlias.__tablename__,
        KbMatchCache.__tablename__,
    )
    missing_tables = [
        table_name
        for table_name in required_tables
        if not inspector.has_table(table_name)
    ]
    if missing_tables:
        joined = ", ".join(missing_tables)
        raise RuntimeError(
            f"Database schema mismatch: missing required table(s): {joined}"
        )

    required_columns = {
        Patient.__tablename__: {"image_blob"},
        ClinicalSession.__tablename__: {
            "patient_id",
            "session_status",
            "version",
            "original_session_id",
            "metadata_json",
        },
        Drug.__tablename__: {"rxnav_last_update"},
    }
    for table_name, columns in required_columns.items():
        existing = {str(item.get("name")) for item in inspector.get_columns(table_name)}
        missing = sorted(columns - existing)
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(
                "Database schema mismatch: "
                f"missing required column(s) in {table_name}: {joined}"
            )


def normalize_session_status(self, value: Any) -> str:
    normalized = self.normalize_string(value)
    if normalized is None:
        return "successful"
    lowered = normalized.casefold()
    if lowered == "failed":
        return "failed"
    return "successful"


def persist_patient(self, db_session: Session, session_data: dict[str, Any]) -> Patient:
    patient = Patient(
        name=self.normalize_string(session_data.get("patient_name")),
        visit_date=self.normalize_date_value(session_data.get("patient_visit_date")),
        anamnesis=self.normalize_string(session_data.get("anamnesis")),
        drugs=self.normalize_string(session_data.get("drugs")),
        laboratory_analysis=self.normalize_string(
            session_data.get("laboratory_analysis")
        ),
        image_blob=self.decode_patient_image(session_data.get("patient_image_base64")),
    )
    db_session.add(patient)
    db_session.flush()
    return patient


def decode_patient_image(self, value: Any) -> bytes | None:
    normalized = self.normalize_string(value)
    if normalized is None:
        return None
    payload = normalized
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", maxsplit=1)[1].strip()
    try:
        return base64.b64decode(payload, validate=True)
    except binascii.Error, ValueError:
        logger.warning("Skipping invalid patient image payload during session save")
        return None


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
    self.ensure_session_result_table()
    safe_offset = max(int(offset), 0)
    safe_limit = max(int(limit), 1)
    conditions: list[Any] = []
    search_pattern = self.build_search_pattern(search)
    if search_pattern is not None:
        section_match = exists(
            select(1).where(
                ClinicalSessionSection.session_id == ClinicalSession.id,
                func.lower(func.coalesce(ClinicalSessionSection.content, "")).like(
                    search_pattern,
                    escape="\\",
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
                func.lower(func.coalesce(Patient.name, "")).like(
                    search_pattern,
                    escape="\\",
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
        elif date_mode == "exact":
            conditions.append(ClinicalSession.session_timestamp >= day_start)
            conditions.append(ClinicalSession.session_timestamp < next_day)

    db_session = self.session_factory()
    try:
        sessions_stmt = select(ClinicalSession, Patient).join(
            Patient,
            ClinicalSession.patient_id == Patient.id,
        )
        count_stmt = (
            select(func.count())
            .select_from(ClinicalSession)
            .join(Patient, ClinicalSession.patient_id == Patient.id)
        )
        if conditions:
            combined = and_(*conditions)
            sessions_stmt = sessions_stmt.where(combined)
            count_stmt = count_stmt.where(combined)
        total_rows = int(db_session.execute(count_stmt).scalar_one())
        rows = db_session.execute(
            sessions_stmt.order_by(
                ClinicalSession.session_timestamp.desc(),
                ClinicalSession.id.desc(),
            )
            .offset(safe_offset)
            .limit(safe_limit)
        ).all()
        session_ids = [int(session_row.id) for session_row, _ in rows]
        report_session_ids: set[int] = set()
        timeline_session_ids: set[int] = set()
        if session_ids:
            result_rows = db_session.execute(
                select(
                    ClinicalSessionResult.session_id,
                    ClinicalSessionResult.payload_json,
                ).where(ClinicalSessionResult.session_id.in_(session_ids))
            ).all()
            for result_session_id, payload_json in result_rows:
                parsed_payload = self.parse_session_result_payload(payload_json)
                if not isinstance(parsed_payload, dict):
                    continue
                parsed_report = self.normalize_string(parsed_payload.get("report"))
                if parsed_report is not None:
                    report_session_ids.add(int(result_session_id))
                parsed_timeline = parsed_payload.get("patient_timeline")
                if isinstance(parsed_timeline, dict):
                    timeline_session_ids.add(int(result_session_id))
            section_report_rows = db_session.execute(
                select(ClinicalSessionSection.session_id).where(
                    ClinicalSessionSection.session_id.in_(session_ids),
                    ClinicalSessionSection.section_kind == "final_report",
                )
            ).all()
            for (section_session_id,) in section_report_rows:
                report_session_ids.add(int(section_session_id))
        items = [
            {
                "session_id": int(session_row.id),
                "patient_name": self.normalize_string(patient_row.name),
                "session_timestamp": session_row.session_timestamp,
                "version": int(session_row.version or 1),
                "original_session_id": self.to_int(session_row.original_session_id),
                "status": self.normalize_session_status(session_row.session_status),
                "total_duration": self.to_float(session_row.total_duration),
                "has_report": int(session_row.id) in report_session_ids,
                "has_timeline": int(session_row.id) in timeline_session_ids,
                "can_generate_timeline": bool(
                    self.normalize_string(patient_row.anamnesis)
                    or self.normalize_string(patient_row.drugs)
                    or self.normalize_string(patient_row.laboratory_analysis)
                ),
            }
            for session_row, patient_row in rows
        ]
        return items, total_rows
    finally:
        db_session.close()


def get_session_detail(self, session_id: int) -> dict[str, Any] | None:
    self.ensure_session_result_table()
    safe_session_id = int(session_id)
    db_session = self.session_factory()
    try:
        row = db_session.execute(
            select(ClinicalSession, Patient)
            .join(Patient, ClinicalSession.patient_id == Patient.id)
            .where(ClinicalSession.id == safe_session_id)
        ).first()
        if row is None:
            return None
        session_row, patient_row = row
        section_rows = db_session.execute(
            select(
                ClinicalSessionSection.section_kind, ClinicalSessionSection.content
            ).where(ClinicalSessionSection.session_id == safe_session_id)
        ).all()
        sections = {
            str(kind): self.normalize_string(content) or ""
            for kind, content in section_rows
        }
        payload = self.get_session_result_payload(safe_session_id) or {}
        metadata = self.parse_session_result_payload(session_row.metadata_json) or {}
        session_text = self.normalize_string(
            payload.get("original_session_text")
        ) or self.build_session_text_from_sections(sections)
        return {
            "session_id": safe_session_id,
            "patient_name": self.normalize_string(patient_row.name),
            "visit_date": patient_row.visit_date,
            "session_timestamp": session_row.session_timestamp,
            "version": int(session_row.version or 1),
            "original_session_id": self.to_int(session_row.original_session_id),
            "status": self.normalize_session_status(session_row.session_status),
            "text_extraction_model": self.normalize_string(
                session_row.text_extraction_model
            ),
            "clinical_model": self.normalize_string(session_row.clinical_model),
            "metadata": metadata,
            "sections": sections,
            "session_text": session_text,
            "result_payload": payload,
            "report": self.normalize_string(payload.get("report"))
            or self.normalize_string(sections.get("final_report")),
        }
    finally:
        db_session.close()


def build_session_text_from_sections(self, sections: dict[str, str]) -> str:
    chunks: list[str] = []
    for key, label in (
        ("anamnesis", "ANAMNESIS"),
        ("drugs", "THERAPY"),
        ("laboratory_analysis", "LABORATORY ANALYSIS"),
    ):
        value = self.normalize_string(sections.get(key))
        if value:
            chunks.append(f"{label}\n{value}")
    return "\n\n".join(chunks)


def update_session_text_and_metadata(
    self,
    session_id: int,
    *,
    session_text: str | None,
    metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    self.ensure_session_result_table()
    safe_session_id = int(session_id)
    db_session = self.session_factory()
    try:
        existing = db_session.get(ClinicalSession, safe_session_id)
        if existing is None:
            return None
        if metadata is not None:
            existing.metadata_json = self.serialize_json_payload(metadata)
        if session_text is not None:
            result = db_session.execute(
                select(ClinicalSessionResult).where(
                    ClinicalSessionResult.session_id == safe_session_id
                )
            ).scalar_one_or_none()
            payload = (
                self.parse_session_result_payload(result.payload_json)
                if result is not None
                else {}
            ) or {}
            payload["original_session_text"] = session_text
            payload["manual_edit_saved_at"] = datetime.now().isoformat()
            serialized_payload = self.serialize_json_payload(payload)
            if serialized_payload is not None:
                if result is None:
                    db_session.add(
                        ClinicalSessionResult(
                            session_id=safe_session_id,
                            payload_json=serialized_payload,
                        )
                    )
                else:
                    result.payload_json = serialized_payload
        db_session.commit()
        return self.get_session_detail(safe_session_id)
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def get_next_session_version(self, original_session_id: int) -> int:
    self.ensure_session_result_table()
    safe_original_id = int(original_session_id)
    db_session = self.session_factory()
    try:
        max_version = db_session.execute(
            select(func.max(ClinicalSession.version)).where(
                or_(
                    ClinicalSession.id == safe_original_id,
                    ClinicalSession.original_session_id == safe_original_id,
                )
            )
        ).scalar_one_or_none()
        return int(max_version or 1) + 1
    finally:
        db_session.close()


def parse_session_result_payload(
    self, payload_json: str | None
) -> dict[str, Any] | None:
    normalized_payload = self.normalize_string(payload_json)
    if normalized_payload is None:
        return None
    try:
        parsed = json.loads(normalized_payload)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def get_session_result_payload(self, session_id: int) -> dict[str, Any] | None:
    self.ensure_session_result_table()
    safe_session_id = int(session_id)
    db_session = self.session_factory()
    try:
        payload_json = db_session.execute(
            select(ClinicalSessionResult.payload_json).where(
                ClinicalSessionResult.session_id == safe_session_id
            )
        ).scalar_one_or_none()
        return self.parse_session_result_payload(payload_json)
    finally:
        db_session.close()


def upsert_session_result_payload(
    self, session_id: int, payload: dict[str, Any]
) -> bool:
    self.ensure_session_result_table()
    safe_session_id = int(session_id)
    serialized_payload = self.serialize_json_payload(payload)
    if serialized_payload is None:
        return False
    db_session = self.session_factory()
    try:
        existing_session = db_session.get(ClinicalSession, safe_session_id)
        if existing_session is None:
            return False
        existing_result = db_session.execute(
            select(ClinicalSessionResult).where(
                ClinicalSessionResult.session_id == safe_session_id
            )
        ).scalar_one_or_none()
        if existing_result is None:
            db_session.add(
                ClinicalSessionResult(
                    session_id=safe_session_id,
                    payload_json=serialized_payload,
                )
            )
        else:
            existing_result.payload_json = serialized_payload
        db_session.commit()
        return True
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def get_session_timeline_source(self, session_id: int) -> dict[str, Any] | None:
    self.ensure_session_result_table()
    safe_session_id = int(session_id)
    db_session = self.session_factory()
    try:
        row = db_session.execute(
            select(ClinicalSession, Patient)
            .join(Patient, ClinicalSession.patient_id == Patient.id)
            .where(ClinicalSession.id == safe_session_id)
        ).first()
        if row is None:
            return None
        session_row, patient_row = row
        payload_json = db_session.execute(
            select(ClinicalSessionResult.payload_json).where(
                ClinicalSessionResult.session_id == safe_session_id
            )
        ).scalar_one_or_none()
        session_payload = self.parse_session_result_payload(payload_json) or {}
        section_rows = db_session.execute(
            select(
                ClinicalSessionSection.section_kind, ClinicalSessionSection.content
            ).where(ClinicalSessionSection.session_id == safe_session_id)
        ).all()
        sections = {
            str(kind): self.normalize_string(content)
            for kind, content in section_rows
            if self.normalize_string(kind) is not None
        }
        return {
            "session_id": safe_session_id,
            "patient_name": self.normalize_string(patient_row.name),
            "visit_date": patient_row.visit_date.isoformat()
            if patient_row.visit_date
            else None,
            "session_timestamp": (
                session_row.session_timestamp.isoformat()
                if session_row.session_timestamp
                else None
            ),
            "anamnesis": self.normalize_string(patient_row.anamnesis),
            "drugs": self.normalize_string(patient_row.drugs),
            "laboratory_analysis": self.normalize_string(
                patient_row.laboratory_analysis
            ),
            "text_extraction_model": self.normalize_string(
                session_row.text_extraction_model
            ),
            "clinical_model": self.normalize_string(session_row.clinical_model),
            "sections": sections,
            "session_result_payload": session_payload,
        }
    finally:
        db_session.close()


def delete_session(self, session_id: int) -> bool:
    self.ensure_session_result_table()
    safe_session_id = int(session_id)
    db_session = self.session_factory()
    try:
        existing = db_session.get(ClinicalSession, safe_session_id)
        if existing is None:
            return False
        patient_id = int(existing.patient_id)
        db_session.execute(
            delete(ClinicalSessionResult).where(
                ClinicalSessionResult.session_id == safe_session_id
            )
        )
        db_session.execute(
            delete(ClinicalSessionSection).where(
                ClinicalSessionSection.session_id == safe_session_id
            )
        )
        db_session.execute(
            delete(ClinicalSessionLab).where(
                ClinicalSessionLab.session_id == safe_session_id
            )
        )
        db_session.execute(
            delete(ClinicalSessionDrug).where(
                ClinicalSessionDrug.session_id == safe_session_id
            )
        )
        db_session.execute(
            delete(ClinicalSession).where(ClinicalSession.id == safe_session_id)
        )
        remaining_patient_sessions = db_session.execute(
            select(func.count())
            .select_from(ClinicalSession)
            .where(ClinicalSession.patient_id == patient_id)
        ).scalar_one()
        if int(remaining_patient_sessions) == 0:
            db_session.execute(delete(Patient).where(Patient.id == patient_id))
        db_session.commit()
        return True
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def normalize_string(self, value: Any) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        if normalized.lower() in {"not available", "nan", "none", "<na>", "nat"}:
            return None
        return normalized
    if pd.isna(value):
        return None
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if normalized.lower() in {"not available", "nan", "none", "<na>", "nat"}:
        return None
    return normalized


def normalize_flag(self, value: Any) -> int | None:
    normalized = self.normalize_string(value)
    if normalized is None:
        return None
    lowered = normalized.lower()
    if lowered in {"1", "y", "yes", "true"}:
        return 1
    if lowered in {"0", "n", "no", "false"}:
        return 0
    if lowered == "2":
        return 0
    try:
        numeric = int(normalized)
    except TypeError, ValueError:
        return None
    return 1 if numeric != 0 else 0


def normalize_date(self, value: Any) -> str | None:
    normalized_date = self.normalize_date_value(value)
    if normalized_date is None:
        normalized = self.normalize_string(value)
        return normalized or None
    return normalized_date.isoformat()


def normalize_date_value(self, value: Any) -> date | None:
    normalized = self.normalize_string(value)
    if not normalized:
        return None
    parsed: Any
    if re.fullmatch(r"[+-]?\d+", normalized):
        digits = normalized[1:] if normalized.startswith(("+", "-")) else normalized
        if len(digits) == 8:
            parsed = pd.to_datetime(
                normalized, errors="coerce", format="%Y%m%d", utc=True
            )
        else:
            inferred_unit = {
                10: "s",
                13: "ms",
                16: "us",
                19: "ns",
            }.get(len(digits))
            if inferred_unit is None:
                return None
            parsed = pd.to_datetime(
                int(normalized),
                errors="coerce",
                utc=True,
                unit=inferred_unit,
            )
    else:
        parsed = pd.to_datetime(normalized, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.date()


def join_values(self, values: set[str]) -> str | None:
    if not values:
        return None
    return "; ".join(sorted(values))


def to_int(self, value: Any) -> int | None:
    normalized = self.normalize_string(value)
    if normalized is None:
        return None
    try:
        return int(float(normalized))
    except TypeError, ValueError:
        return None


def to_float(self, value: Any) -> float | None:
    normalized = self.normalize_string(value)
    if normalized is None:
        return None
    try:
        return float(normalized)
    except TypeError, ValueError:
        return None


def parse_datetime(self, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    if isinstance(parsed, pd.Timestamp):
        return parsed.to_pydatetime()
    return parsed


def persist_session_sections(
    self, db_session: Session, session_id: int, session_data: dict[str, Any]
) -> None:
    issues_content: str | None = None
    issues_raw = session_data.get("issues")
    if isinstance(issues_raw, (list, dict)):
        issues_content = json.dumps(issues_raw, ensure_ascii=False)
    elif isinstance(issues_raw, str):
        issues_content = self.normalize_string(issues_raw)
    payload = {
        "anamnesis": session_data.get("anamnesis"),
        "drugs": session_data.get("drugs"),
        "laboratory_analysis": session_data.get("laboratory_analysis"),
        "final_report": session_data.get("final_report"),
        "issues": issues_content,
    }
    for section_kind, value in payload.items():
        content = self.normalize_string(value)
        if content is None:
            continue
        db_session.add(
            ClinicalSessionSection(
                session_id=session_id,
                section_kind=section_kind,
                content=content,
            )
        )


def persist_session_labs(
    self, db_session: Session, session_id: int, session_data: dict[str, Any]
) -> None:
    result_payload = session_data.get("session_result_payload")
    if not isinstance(result_payload, dict):
        return
    timeline_raw = result_payload.get("lab_timeline")
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
    # DB schema enforces one lab row per (session_id, lab_code), so collapse
    # repeated timeline points of the same marker into a single persisted row.
    rows_by_lab_code: dict[str, tuple[str | None, str | None]] = {}
    for item in timeline_raw:
        if not isinstance(item, dict):
            continue
        marker_name = self.normalize_string(item.get("marker_name"))
        if marker_name is None:
            continue
        lab_code = persisted_codes.get(marker_name.upper())
        if lab_code is None:
            continue
        value_raw = self.normalize_string(item.get("value")) or self.normalize_string(
            item.get("value_text")
        )
        upper_limit_raw = self.normalize_string(
            item.get("upper_limit_normal")
        ) or self.normalize_string(item.get("upper_limit_text"))
        if value_raw is None and upper_limit_raw is None:
            continue
        existing_value_raw, existing_upper_limit_raw = rows_by_lab_code.get(
            lab_code, (None, None)
        )
        merged_value_raw = existing_value_raw or value_raw
        merged_upper_limit_raw = existing_upper_limit_raw or upper_limit_raw
        rows_by_lab_code[lab_code] = (merged_value_raw, merged_upper_limit_raw)
    for lab_code, (value_raw, upper_limit_raw) in rows_by_lab_code.items():
        db_session.add(
            ClinicalSessionLab(
                session_id=session_id,
                lab_code=lab_code,
                value_raw=value_raw,
                upper_limit_raw=upper_limit_raw,
            )
        )


def persist_session_drugs(
    self, db_session: Session, session_id: int, session_data: dict[str, Any]
) -> None:
    payload = session_data.get("matched_drugs")
    records: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                records.append(item)
            elif isinstance(item, str):
                records.append({"raw_drug_name": item})
    if not records:
        detected_drugs = session_data.get("detected_drugs")
        if isinstance(detected_drugs, list):
            for item in detected_drugs:
                if isinstance(item, str):
                    records.append({"raw_drug_name": item})
    seen: set[str] = set()
    vocabulary_changed = False
    for item in records:
        raw_drug_name = self.normalize_string(
            item.get("raw_drug_name") or item.get("name")
        )
        if raw_drug_name is None:
            continue
        raw_drug_name_norm = normalize_drug_name(raw_drug_name)
        if not raw_drug_name_norm or raw_drug_name_norm in seen:
            continue
        seen.add(raw_drug_name_norm)
        matched_drug_name = self.normalize_string(item.get("matched_drug_name"))
        rxcui = self.normalize_string(item.get("rxcui"))
        nbk_id = self.normalize_string(item.get("nbk_id"))
        resolved_drug_id = self.resolve_drug_id(
            db_session,
            matched_drug_name=matched_drug_name,
            rxcui=rxcui,
            nbk_id=nbk_id,
        )
        match_reason = self.normalize_string(item.get("match_reason"))
        match_confidence = self.to_float(item.get("match_confidence"))
        if resolved_drug_id is None:
            resolved_drug_id = self.resolve_drug_id_from_match_cache(
                db_session,
                normalized_drug_key=raw_drug_name_norm,
            )
        promoted_drug_id = resolved_drug_id
        should_promote_observed_alias = (
            promoted_drug_id is not None
            and match_reason == "exact_canonical"
            and match_confidence == 1.0
        )
        if should_promote_observed_alias:
            if promoted_drug_id is None:
                raise RuntimeError("Promoted drug id unexpectedly missing")
            self.upsert_drug_alias(
                db_session,
                drug_id=promoted_drug_id,
                alias=raw_drug_name,
                alias_kind="observed_query",
                source="session",
                term_type=None,
            )
        else:
            observation_category = (
                "observed_unresolved_query"
                if resolved_drug_id is None
                else "observed_unpromoted_query"
            )
            catalogs = ReferenceCatalogSerializer(self.session_factory)
            catalogs.upsert_runtime_observation(
                term=raw_drug_name,
                category=observation_category,
                source="session",
                is_active=True,
                db_session=db_session,
            )
            vocabulary_changed = True
        notes = item.get("match_notes")
        if isinstance(notes, (list, dict)):
            notes_value = json.dumps(notes, ensure_ascii=False)
        else:
            notes_value = self.normalize_string(notes)
        db_session.add(
            ClinicalSessionDrug(
                session_id=session_id,
                raw_drug_name=raw_drug_name,
                raw_drug_name_norm=raw_drug_name_norm,
                drug_id=resolved_drug_id,
                match_confidence=match_confidence,
                match_reason=match_reason,
                match_notes=notes_value,
            )
        )
        self.upsert_high_confidence_kb_match_cache(
            db_session,
            raw_drug_name=raw_drug_name,
            raw_drug_name_norm=raw_drug_name_norm,
            normalized_drug_key=raw_drug_name_norm,
            drug_id=resolved_drug_id,
            rxnorm_rxcui=rxcui,
            livertox_nbk_id=nbk_id,
            source="rxnav" if rxcui else "livertox",
            confidence=match_confidence,
            evidence={
                "match_reason": match_reason,
                "match_notes": notes,
                "matched_drug_name": matched_drug_name,
                "source_session_id": session_id,
            },
            ambiguous=bool(item.get("ambiguous_match")),
        )
    if vocabulary_changed:
        invalidate_text_normalization_snapshot()


def persist_session_result_payload(
    self, db_session: Session, session_id: int, session_data: dict[str, Any]
) -> None:
    payload = session_data.get("session_result_payload")
    serialized_payload = self.serialize_json_payload(payload)
    if serialized_payload is None:
        return
    db_session.add(
        ClinicalSessionResult(
            session_id=session_id,
            payload_json=serialized_payload,
        )
    )


def serialize_json_payload(self, payload: Any) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        return self.normalize_string(payload)
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except TypeError, ValueError:
        return self.normalize_string(payload)
