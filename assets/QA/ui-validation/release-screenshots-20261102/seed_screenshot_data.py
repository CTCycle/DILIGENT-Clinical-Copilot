"""Seed an isolated data root with fictional sessions for release screenshots.

Reports come from real pipeline runs over the synthetic audit cases
(audits/clinical-20261028); patient names and dates are fictional.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "app" / "server"))
sys.path.insert(0, str(REPO / "app" / "tests" / "ci"))

from sqlalchemy import text  # noqa: E402

from configurations.startup import get_server_settings  # noqa: E402
from repositories.clinical_session_repository import ClinicalSessionRepository  # noqa: E402
from repositories.context import RepositoryContext  # noqa: E402
from repositories.database.initializer import initialize_sqlite_database  # noqa: E402
from repositories.session_timeline_repository import SessionTimelineRepository  # noqa: E402
from seed_browser_regression_database import seed_knowledge_base  # noqa: E402

AUDIT = REPO / "assets" / "QA" / "audits" / "clinical-20261028"
# (case, patient, status, age in days, model)
CASES = [
    ("C01_isoniazid_hepatocellular_hys", "Marco Bianchi", "completed", 1),
    ("C03_tmpsmx_mixed_hypersensitivity", "Elena Rossi", "completed", 2),
    ("C05_terbinafine_hys_law_severe", "Luca Ferrari", "completed", 4),
    ("C06_flucloxacillin_hys_mimic_cholestatic", "Giulia Conti", "completed", 9),
    ("C09_masld_abnormal_baseline", "Paolo Greco", "completed", 15),
    ("C02_amoxclav_cholestatic_anamnesis_only", "Chiara Romano", "completed", 26),
]


def section(text: str, heading: str) -> str:
    match = re.search(rf"## {heading}[^\n]*\n(.*?)(?=\n## |\Z)", text, re.S | re.I)
    return match.group(1).strip() if match else ""


def load_result(case: str) -> dict:
    for folder in ("e2e_p1", "e2e", "e2e_p0"):
        path = AUDIT / "outputs" / folder / case / "run1.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["job"]["result"]
    raise FileNotFoundError(case)


def timeline_events(result: dict) -> list[dict]:
    kinds = {"drug_start": "therapy", "drug_stop": "therapy", "symptom_onset": "disease",
             "jaundice_or_bilirubin_rise": "disease"}
    titles = {"drug_start": "Started", "drug_stop": "Stopped", "symptom_onset": "Symptom onset",
              "jaundice_or_bilirubin_rise": "Bilirubin rise"}
    events = []
    jaundice_seen = False
    for raw in result["dili_evidence_bundle"]["timeline"]["events"]:
        etype = raw.get("event_type")
        if etype == "jaundice_or_bilirubin_rise":
            if jaundice_seen:
                continue
            jaundice_seen = True
        if etype in kinds:
            title = titles[etype] + (f" {raw['drug_name']}" if raw.get("drug_name") else "")
        elif etype == "abnormal_liver_test" and raw.get("marker") in {"ALT", "ALP", "BILIRUBIN"}:
            title = f"{raw['marker']} {raw.get('value'):g} U/L"
        else:
            continue
        evidence = raw.get("evidence") or {}
        events.append(
            {
                "event_id": f"ev-{len(events)}",
                "title": title,
                "description": evidence.get("claim") or title,
                "event_type": kinds.get(etype, "lab"),
                "timing_type": "explicit_date",
                "event_date": raw["event_date"],
                "date_precision": "day",
                "date_certainty": "explicit",
                "extracted_timing_text": raw["event_date"],
                "source_evidence": evidence.get("quote") or "",
                "source": "patient_record",
                "confidence": 1.0,
                "confidence_rationale": "Explicit date in the clinician-entered record.",
                "sort_order": len(events),
            }
        )
    return events


def ensure_drugs(context: RepositoryContext, matched: list[dict]) -> None:
    """Insert bare drug rows so mention foreign keys resolve in the isolated database."""
    with context.engine.begin() as connection:
        for drug in matched:
            drug_id, name = drug.get("drug_id"), drug.get("raw_drug_name")
            if drug_id is None or not name:
                continue
            connection.execute(
                text(
                    "INSERT OR IGNORE INTO drugs (id, canonical_name, canonical_name_norm, concept_type) "
                    "VALUES (:id, :name, :norm, 'ingredient')"
                ),
                {"id": drug_id, "name": name, "norm": name.lower()},
            )


def main() -> None:
    settings = get_server_settings().database
    initialize_sqlite_database(settings)
    context = RepositoryContext.create()
    seed_knowledge_base(context)
    sessions = ClinicalSessionRepository(context)
    timelines = SessionTimelineRepository(context)
    now = datetime.now(UTC).replace(microsecond=0)
    for case, patient, status, age in CASES:
        text = (AUDIT / "cases" / f"{case}.txt").read_text(encoding="utf-8")
        result = json.loads(
            re.sub(r"AUDIT-\w+-r\d", patient, json.dumps(load_result(case)))
        )
        visit = re.search(r"Visit date:\*\* (\d+ \w+ \d{4})", result["report"])
        visit_date = datetime.strptime(visit.group(1), "%d %B %Y").date()
        stamp = now - timedelta(days=age, hours=3)
        meta = result.get("extraction_metadata", {})
        runtime = result.get("runtime_settings", {})
        payload = dict(result, patient_name=patient)
        ensure_drugs(context, result.get("matched_drugs") or [])
        session_id = sessions.save_clinical_session(
            {
                "patient_name": patient,
                "patient_visit_date": visit_date,
                "session_timestamp": stamp,
                "session_status": status,
                "session_kind": "standard_assessment",
                "clinical_input": text,
                "anamnesis": section(text, "Anamnesis"),
                "drugs": section(text, "Therapy"),
                "laboratory_analysis": section(text, "Laboratory"),
                "hepatic_pattern": (meta.get("hepatic_pattern") or {}).get("final_value"),
                "text_extraction_model": runtime.get("text_extraction_model"),
                "clinical_model": runtime.get("clinical_model"),
                "total_duration": result.get("progress_elapsed"),
                "final_report": result["final_report"],
                "llm_clinical_summary": result.get("llm_clinical_summary"),
                "detected_drugs": result.get("detected_drugs"),
                "matched_drugs": result.get("matched_drugs"),
                "issues": result.get("issues"),
                "session_result_payload": payload,
            }
        )
        events = timeline_events(result)
        timelines.create_session_timeline_record(
            session_id,
            {
                "session_id": session_id,
                "generated_at": stamp,
                "generation_status": "llm_generated",
                "generation_note": "Fictional case.",
                "source_model": runtime.get("clinical_model"),
                "source_kind": "cloud",
                "model_provider": runtime.get("llm_provider"),
                "events": events,
            },
        )
        print(session_id, patient, len(events), "events")


if __name__ == "__main__":
    main()
