from __future__ import annotations

import re
from datetime import date
from typing import Any

from common.constants import (
    DEFAULT_DILI_CLASSIFICATION,
    R_SCORE_CHOLESTATIC_THRESHOLD,
    R_SCORE_HEPATOCELLULAR_THRESHOLD,
)
from domain.clinical.entities import (
    ClinicalLabEntry,
    DrugClinicalAssessment,
    DrugRucamAssessment,
    DrugSuspensionContext,
    HepatotoxicityPatternAssessment,
    HepatotoxicityPatternScore,
    PatientLabTimeline,
    PipelineIssue,
)
from services.clinical.report_language import (
    phrase,
    report_heading,
    rucam_summary_text,
)

###############################################################################
NOT_AVAILABLE_TEXT = "Not available"
REDUNDANT_REPORT_LINE_RE = re.compile(
    r"generated\s+report.*?(drug[- ]induced\s+liver\s+injury|\bdili\b)",
    re.IGNORECASE,
)
LIVERTOX_TITLE_LINE_RE = re.compile(
    r"^\s*\*{0,2}[^*\n]+?\s*-\s*LiverTox score\b.*\*{0,2}\s*$",
    re.IGNORECASE,
)
REPORT_LABEL_LINE_RE = re.compile(r"^\s*\*{0,2}\s*Report\s*\*{0,2}\s*$", re.IGNORECASE)
BIBLIOGRAPHY_LINE_RE = re.compile(
    r"^\s*\*{0,2}\s*Bibliography source\s*\*{0,2}\s*:\s*LiverTox\s*$",
    re.IGNORECASE,
)
DRIFT_SECTION_LINE_RE = re.compile(
    r"^\s*(medication|assessment|plan)\s*$", re.IGNORECASE
)
STRUCTURED_DILI_SECTION_LINE_RE = re.compile(
    r"^\s*#{0,6}\s*\*{0,2}\s*Structured\s+DILI\s+Assessment\s+Report\s*\*{0,2}\s*$",
    re.IGNORECASE,
)
RATE_LIMIT_WAIT_HINT_RE = re.compile(
    r"please\s+try\s+again\s+in\s+([0-9]+(?:\.[0-9]+)?)s",
    re.IGNORECASE,
)


###############################################################################
class HepatotoxicityPatternCalculator:
    # -------------------------------------------------------------------------
    def calculate(
        self,
        *,
        alt_value: float | None,
        alt_uln: float | None,
        alp_value: float | None,
        alp_uln: float | None,
    ) -> HepatotoxicityPatternScore:
        alt_multiple = self.safe_ratio(alt_value, alt_uln)
        alp_multiple = self.safe_ratio(alp_value, alp_uln)

        r_score = None
        if alt_multiple is not None and alp_multiple not in (None, 0.0):
            r_score = alt_multiple / alp_multiple

        classification = DEFAULT_DILI_CLASSIFICATION
        if r_score is not None:
            if r_score > R_SCORE_HEPATOCELLULAR_THRESHOLD:
                classification = "hepatocellular"
            elif r_score < R_SCORE_CHOLESTATIC_THRESHOLD:
                classification = "cholestatic"
            else:
                classification = "mixed"

        return HepatotoxicityPatternScore(
            alt_multiple=alt_multiple,
            alp_multiple=alp_multiple,
            r_score=r_score,
            classification=classification,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def safe_ratio(value: float | None, reference: float | None) -> float | None:
        if value is None or reference is None:
            return None
        if reference == 0:
            return None
        return value / reference


###############################################################################
class HepatotoxicityPatternAnalyzer:
    def __init__(self) -> None:
        self.r_score: float | None = None
        self.calculator = HepatotoxicityPatternCalculator()

    # -------------------------------------------------------------------------
    def calculate_hepatotoxicity_pattern(
        self, lab_timeline: PatientLabTimeline
    ) -> HepatotoxicityPatternScore:
        anchor = self.select_anchor_pair(lab_timeline)
        if anchor is None:
            score = HepatotoxicityPatternScore(
                alt_multiple=None,
                alp_multiple=None,
                r_score=None,
                classification=DEFAULT_DILI_CLASSIFICATION,
            )
            self.r_score = None
            return score
        score = self.calculator.calculate(
            alt_value=anchor["alt_value"],
            alt_uln=anchor["alt_uln"],
            alp_value=anchor["alp_value"],
            alp_uln=anchor["alp_uln"],
        )
        self.r_score = score.r_score
        return score

    # -------------------------------------------------------------------------
    def assess_payload(
        self,
        lab_timeline: PatientLabTimeline,
    ) -> HepatotoxicityPatternAssessment:
        score = self.calculate_hepatotoxicity_pattern(lab_timeline)
        if score.r_score is None:
            issue = PipelineIssue(
                severity="warning",
                code="missing_hepatotoxicity_inputs",
                message=(
                    "Laboratory data are insufficient for a numeric R ratio "
                    "(ideally ALT/AST, ALP, and bilirubin). Continuing with "
                    "indeterminate pattern and reduced confidence."
                ),
                field="laboratory_analysis",
            )
            self.r_score = None
            return HepatotoxicityPatternAssessment(
                score=score,
                status="undetermined_due_to_missing_labs",
                issues=[issue],
            )
        self.r_score = score.r_score
        return HepatotoxicityPatternAssessment(
            score=score,
            status="ok",
            issues=[],
        )

    # -------------------------------------------------------------------------
    def select_anchor_pair(
        self, lab_timeline: PatientLabTimeline
    ) -> dict[str, float] | None:
        dated_candidates = self.group_entries_by_date(lab_timeline.entries)
        for sample_date in sorted(dated_candidates):
            bucket = dated_candidates[sample_date]
            pair = self.build_anchor_from_bucket(bucket)
            if pair is not None:
                return pair
        undated = self.build_anchor_from_bucket(lab_timeline.entries)
        return undated

    # -------------------------------------------------------------------------
    def group_entries_by_date(
        self,
        entries: list[ClinicalLabEntry],
    ) -> dict[str, list[ClinicalLabEntry]]:
        grouped: dict[str, list[ClinicalLabEntry]] = {}
        for entry in entries:
            if not entry.sample_date:
                continue
            grouped.setdefault(entry.sample_date, []).append(entry)
        return grouped

    # -------------------------------------------------------------------------
    def build_anchor_from_bucket(
        self,
        entries: list[ClinicalLabEntry],
    ) -> dict[str, float] | None:
        alt_like = self.pick_best_entry(entries, {"ALT", "AST"})
        alp = self.pick_best_entry(entries, {"ALP"})
        if alt_like is None or alp is None:
            return None
        alt_value = self.parse_entry_value(alt_like)
        alp_value = self.parse_entry_value(alp)
        if alt_value is None or alp_value is None:
            return None
        alt_uln = self.resolve_uln(alt_like, fallback=40.0)
        alp_uln = self.resolve_uln(alp, fallback=120.0)
        if alt_uln <= 0 or alp_uln <= 0:
            return None
        return {
            "alt_value": alt_value,
            "alt_uln": alt_uln,
            "alp_value": alp_value,
            "alp_uln": alp_uln,
        }

    # -------------------------------------------------------------------------
    def pick_best_entry(
        self,
        entries: list[ClinicalLabEntry],
        marker_names: set[str],
    ) -> ClinicalLabEntry | None:
        selected: ClinicalLabEntry | None = None
        for entry in entries:
            if entry.marker_name.upper() not in marker_names:
                continue
            if selected is None:
                selected = entry
                continue
            selected_value = self.parse_entry_value(selected)
            current_value = self.parse_entry_value(entry)
            if selected_value is None and current_value is not None:
                selected = entry
            elif (
                current_value is not None
                and selected_value is not None
                and current_value > selected_value
            ):
                selected = entry
        return selected

    # -------------------------------------------------------------------------
    def parse_entry_value(self, entry: ClinicalLabEntry) -> float | None:
        if entry.value is not None:
            return float(entry.value)
        return self.parse_marker_value(entry.value_text)

    # -------------------------------------------------------------------------
    def resolve_uln(self, entry: ClinicalLabEntry, *, fallback: float) -> float:
        if entry.upper_limit_normal is not None and entry.upper_limit_normal > 0:
            return float(entry.upper_limit_normal)
        parsed = self.parse_marker_value(entry.upper_limit_text)
        if parsed is not None and parsed > 0:
            return parsed
        return fallback

    # -------------------------------------------------------------------------
    def parse_marker_value(self, raw: str | None) -> float | None:
        if raw is None:
            return None
        normalized = raw.replace(",", ".")
        match = re.search(r"[-+]?\d*\.?\d+", normalized)
        if not match:
            return None
        try:
            return float(match.group())
        except ValueError:
            return None

    # -------------------------------------------------------------------------
    def safe_ratio(self, value: float | None, reference: float | None) -> float | None:
        return self.calculator.safe_ratio(value, reference)

    # -------------------------------------------------------------------------
    def stringify_scores(
        self, pattern_score: HepatotoxicityPatternScore | None
    ) -> dict[str, str]:
        if not pattern_score:
            return {}

        mapping = {
            "alt_multiple": (pattern_score.alt_multiple, "{:.2f}x ULN"),
            "alp_multiple": (pattern_score.alp_multiple, "{:.2f}x ULN"),
            "r_score": (pattern_score.r_score, "{:.2f}"),
        }

        return {
            key: fmt.format(val) if val is not None else NOT_AVAILABLE_TEXT
            for key, (val, fmt) in mapping.items()
        }


###############################################################################

# Extracted from the facade module; functions intentionally accept the facade instance.


def format_similarity_header(
    index: int,
    *,
    distance: Any,
    rerank_score: Any = None,
) -> str:
    segments = [f"Document {index}"]
    if isinstance(rerank_score, (int, float)):
        segments.append(f"Rerank: {float(rerank_score):.4f}")
    if isinstance(distance, (int, float)):
        segments.append(f"Distance: {float(distance):.4f}")
    return f"[{' | '.join(segments)}]"


def format_start_note(
    self,
    *,
    start_reported: bool,
    start_date: date | None,
    start_interval_days: int | None,
    visit_date: date | None,
) -> str:
    if not start_reported:
        return "Therapy start was not documented; assume chronic exposure unless another source clarifies the onset."
    if start_date is None:
        return "Therapy start was reported but no reliable date could be parsed from the notes."
    if visit_date is None or start_interval_days is None:
        return f"Therapy started on {start_date.isoformat()}, but the visit date was unavailable for latency comparisons."
    if start_interval_days < 0:
        humanized = self.humanize_interval(abs(start_interval_days))
        return f"Therapy was documented to start on {start_date.isoformat()}, {humanized} after the visit; verify this discrepancy manually."
    if start_interval_days == 0:
        return f"Therapy started on {start_date.isoformat()}, coinciding with the clinical visit."
    humanized = self.humanize_interval(start_interval_days)
    return f"Therapy started on {start_date.isoformat()}, roughly {humanized} before the visit."


def format_suspension_prompt(self, suspension: DrugSuspensionContext) -> str:
    segments: list[str] = []
    if not suspension.suspended:
        segments.append("Active therapy; no suspension reported.")
    elif suspension.suspension_date is None:
        segments.append(
            "Reported as suspended without a reliable date; evaluate latency with the LiverTox excerpt."
        )
    elif suspension.interval_days is None:
        segments.append(
            f"Suspended on {suspension.suspension_date.isoformat()}, but the interval relative to the visit is unclear; rely on LiverTox latency guidance."
        )
    elif suspension.interval_days < 0:
        days = abs(suspension.interval_days)
        segments.append(
            f"Suspended on {suspension.suspension_date.isoformat()} ({days} days after the visit); treat as ongoing exposure."
        )
    elif suspension.interval_days == 0:
        segments.append(
            f"Suspended on {suspension.suspension_date.isoformat()} (same day as the visit); residual exposure is expected."
        )
    else:
        segments.append(
            f"Suspended on {suspension.suspension_date.isoformat()} ({suspension.interval_days} days before the visit); compare with LiverTox latency guidance."
        )

    return " ".join(segment for segment in segments if segment)


def format_start_prompt(self, suspension: DrugSuspensionContext) -> str:
    if suspension.start_note:
        return suspension.start_note
    if suspension.start_reported:
        return "Therapy start was reported, but no reliable date was available."
    return "No therapy start information was detected; treat the exposure window as chronic unless contradicted."


def format_visit_date_anchor(visit_date: date | None) -> str:
    if visit_date is None:
        return "Not provided."
    return visit_date.isoformat()


def prepare_metadata_prompt(self, metadata: dict[str, Any] | None) -> tuple[str, str]:
    score = self.resolve_livertox_score(metadata)
    details: list[str] = [f"- Likelihood score: {score}"]
    if metadata:
        mapping = [
            ("Agent classification", metadata.get("agent_classification")),
            ("Primary classification", metadata.get("primary_classification")),
            ("Secondary classification", metadata.get("secondary_classification")),
            ("Reference count", metadata.get("reference_count")),
            ("Year approved", metadata.get("year_approved")),
        ]
        seen: set[str] = set()
        for label, raw in mapping:
            if raw is None:
                continue
            value = str(raw).strip()
            if not value or value.lower() == "nan":
                continue
            key = f"{label}:{value}"
            if key in seen:
                continue
            seen.add(key)
            details.append(f"- {label}: {value}")
    if len(details) == 1:
        details.append("- No additional LiverTox metadata was available.")
    return score, "\n".join(details)


def format_drug_heading(self, drug_name: str, score: str) -> str:
    normalized_name = drug_name.strip() if drug_name else ""
    if not normalized_name:
        normalized_name = "Unnamed drug"
    normalized_score = score.strip() if score else ""
    if not normalized_score:
        normalized_score = NOT_AVAILABLE_TEXT
    return f"{normalized_name} - LiverTox score {normalized_score}"


def format_rucam_prompt_block(self, rucam: DrugRucamAssessment | None) -> str:
    if rucam is None:
        return "Estimated RUCAM not available."
    limitations = ", ".join((rucam.limitations or [])[:3]) or "not specified"
    return (
        f"- Score: {rucam.total_score}\n"
        f"- Category: {rucam.causality_category}\n"
        f"- Confidence: {rucam.confidence}\n"
        f"- Estimated due to incomplete clinical data: yes\n"
        f"- Key limitations: {limitations}"
    )


def escape_braces(value: str) -> str:
    return value.replace("{", "{{").replace("}", "}}")


def remove_redundant_report_sentence(text: str) -> str:
    if not text:
        return ""
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        if STRUCTURED_DILI_SECTION_LINE_RE.match(raw_line.strip()):
            break
        compact = re.sub(r"[\s*_`#:\-]+", " ", raw_line).strip()
        if compact and REDUNDANT_REPORT_LINE_RE.search(compact):
            continue
        cleaned_lines.append(raw_line)
    cleaned = "\n".join(cleaned_lines).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def render_matched_drug_section(
    self,
    entry: DrugClinicalAssessment,
    *,
    report_language: str = "en",
) -> str:
    score = self.resolve_livertox_score(entry.matched_livertox_row)
    title = self.format_drug_heading(entry.drug_name, score)
    body = self.sanitize_renderable_body(entry)
    if not body:
        body = self.build_fallback_technical_note(
            entry, report_language=report_language
        )
    rucam = entry.rucam
    localized_rucam = (
        rucam_summary_text(rucam, report_language)
        if rucam is not None
        else phrase("rucam_not_calculated", report_language)
    )
    evidence_lines = self.render_evidence_quality_lines(
        entry,
        report_language=report_language,
    )
    report_label = phrase("report_label", report_language)
    bibliography_label = phrase("bibliography_source", report_language)
    return (
        f"**{title}**\n\n"
        f"{evidence_lines}\n\n"
        f"**RUCAM**: {localized_rucam}\n\n"
        f"**{report_label}**\n\n"
        f"{body}\n\n"
        f"**{bibliography_label}**: {self.bibliography_source_label()}"
    ).strip()


def render_evidence_quality_lines(
    entry: DrugClinicalAssessment,
    *,
    report_language: str = "en",
) -> str:
    quality = entry.evidence_quality or phrase("unknown", report_language)
    matched_name = ""
    if isinstance(entry.matched_livertox_row, dict):
        matched_name = str(entry.matched_livertox_row.get("drug_name") or "").strip()
    target = (
        matched_name or entry.canonical_name or phrase("not_available", report_language)
    )
    warnings = (
        "; ".join(entry.evidence_warnings)
        if entry.evidence_warnings
        else phrase("none", report_language)
    )
    return (
        f"**{phrase('evidence_match', report_language)}**: {quality}. "
        f"{phrase('matched_local_record', report_language)}: {target}. "
        f"{phrase('warnings', report_language)}: {warnings}."
    )


def sanitize_renderable_body(self, entry: DrugClinicalAssessment) -> str:
    text = entry.paragraph.strip() if entry.paragraph else ""
    if not text:
        return ""
    expected_name = (entry.drug_name or "").strip().lower()
    lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if lines and lines[-1]:
                lines.append("")
            continue
        if REDUNDANT_REPORT_LINE_RE.search(
            re.sub(r"[\s*_`#:\-]+", " ", stripped).strip()
        ):
            continue
        if REPORT_LABEL_LINE_RE.match(stripped):
            continue
        if BIBLIOGRAPHY_LINE_RE.match(stripped):
            continue
        if stripped == "---":
            continue
        if stripped.lower().startswith("## global synthesis"):
            break
        if DRIFT_SECTION_LINE_RE.match(stripped):
            break
        if STRUCTURED_DILI_SECTION_LINE_RE.match(stripped):
            break
        title_match = LIVERTOX_TITLE_LINE_RE.match(stripped)
        if title_match:
            if expected_name and expected_name not in stripped.lower():
                continue
            continue
        lines.append(raw_line.rstrip())
    sanitized = "\n".join(lines).strip()
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized).strip()
    normalized = re.sub(r"\s+", " ", sanitized).strip().lower()
    if "local livertox excerpt not available" in normalized:
        return ""
    return sanitized


def build_fallback_technical_note(
    self,
    entry: DrugClinicalAssessment,
    *,
    report_language: str = "en",
) -> str:
    if entry.suspension.excluded:
        return self.build_excluded_paragraph(entry, report_language=report_language)
    if entry.ambiguous_match:
        return self.build_ambiguous_match_paragraph(
            entry,
            report_language=report_language,
        )
    if entry.missing_livertox:
        return self.build_missing_excerpt_paragraph(
            entry,
            report_language=report_language,
        )
    return self.build_error_paragraph(entry, report_language=report_language)


def render_unresolved_mentions_section(
    self,
    entries: list[DrugClinicalAssessment],
    *,
    report_language: str = "en",
) -> str | None:
    if not entries:
        return None
    lines: list[str] = [
        f"## {report_heading('unresolved_mentions', report_language)}",
        "",
    ]
    for entry in entries:
        label = (entry.drug_name or "").strip() or phrase(
            "unnamed_drug", report_language
        )
        reason = self.describe_unresolved_entry(entry, report_language=report_language)
        rucam_summary = (
            rucam_summary_text(entry.rucam, report_language)
            if entry.rucam is not None
            else phrase("rucam_not_calculated", report_language)
        )
        lines.append(f"- **{label}**: {reason} {rucam_summary}.")
    return "\n".join(lines).strip()


def describe_unresolved_entry(
    self,
    entry: DrugClinicalAssessment,
    report_language: str = "en",
) -> str:
    status = (entry.match_status or "").strip().lower()
    if status in {"ambiguous", "ambiguous_match"} or entry.ambiguous_match:
        candidates = (
            ", ".join(entry.match_candidates)
            if entry.match_candidates
            else phrase("rucam_insufficient_data", report_language)
        )
        return (
            f"{phrase('livertox_ambiguous', report_language)} "
            f"{phrase('candidate_matches', report_language, candidates=candidates)} "
            f"{phrase('manual_curation', report_language)}"
        )
    if status in {"missing", "missing_match"}:
        return phrase("no_matching_record", report_language)
    if status == "matched_no_excerpt":
        return phrase("matched_no_excerpt", report_language)
    if entry.missing_livertox:
        return phrase("matched_no_excerpt", report_language)
    return phrase("deterministic_section_unavailable", report_language)


def build_excluded_paragraph(
    self,
    entry: DrugClinicalAssessment,
    report_language: str = "en",
) -> str:
    suspension = entry.suspension
    if report_language.startswith("it"):
        if suspension.suspension_date is not None:
            detail = (
                f"La terapia è stata sospesa il {suspension.suspension_date.isoformat()} "
                "molto prima della visita; questa esposizione è stata quindi esclusa "
                "dalla valutazione attiva di causalità DILI."
            )
        else:
            detail = (
                "La terapia risulta sospesa molto prima della visita ed è stata "
                "esclusa dalla valutazione attiva di causalità DILI."
            )
        recommendation = (
            "È consigliata una verifica manuale della latenza se l'esposizione "
            "torna clinicamente rilevante."
        )
        return f"{detail} {recommendation}"
    if suspension.suspension_date is not None:
        detail = (
            f"The therapy was suspended on {suspension.suspension_date.isoformat()} "
            "well before the visit, so this exposure was excluded from active DILI "
            "causality assessment."
        )
    else:
        detail = (
            "The therapy was reported as suspended well before the visit and was "
            "excluded from active DILI causality assessment."
        )
    recommendation = (
        "Manual latency verification is suggested if the exposure history becomes "
        "clinically relevant again."
    )
    return f"{detail} {recommendation}"


def build_missing_excerpt_paragraph(
    self,
    entry: DrugClinicalAssessment,
    report_language: str = "en",
) -> str:
    _ = entry
    return phrase("livertox_missing", report_language)


def build_ambiguous_match_paragraph(
    self,
    entry: DrugClinicalAssessment,
    report_language: str = "en",
) -> str:
    candidates = (
        ", ".join(entry.match_candidates)
        if entry.match_candidates
        else phrase("rucam_insufficient_data", report_language)
    )
    note = phrase("livertox_ambiguous", report_language)
    details = phrase("candidate_matches", report_language, candidates=candidates)
    guidance = phrase("manual_curation", report_language)
    return f"{note} {details} {guidance}"


def build_error_paragraph(
    self,
    entry: DrugClinicalAssessment,
    report_language: str = "en",
) -> str:
    _ = entry
    message = phrase("rucam_insufficient_data", report_language)
    return message


def format_similarity_fragment(self, index: int, record: dict[str, Any]) -> str | None:
    text = str(record.get("text", "")).strip()
    if not text:
        return None
    header = self.format_similarity_header(
        index,
        distance=record.get("distance"),
        rerank_score=record.get("rerank_score"),
    )
    return f"{header}\n{text}"
