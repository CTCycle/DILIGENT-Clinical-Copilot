from __future__ import annotations

from domain.clinical.dili import ClinicalEvidenceQuote, DiliTimeline, DiliTimelineEvent
from domain.clinical.entities import DrugEntry, PatientLabTimeline


class DiliTimelineEngine:
    def build(self, drugs: list[DrugEntry], labs: PatientLabTimeline) -> DiliTimeline:
        events: list[DiliTimelineEvent] = []
        missing: list[str] = []
        for drug in drugs:
            for event_type, event_date in (
                ("drug_start", drug.therapy_start_date),
                ("drug_stop", drug.suspension_date),
            ):
                events.append(
                    DiliTimelineEvent(
                        event_type=event_type,
                        event_date=event_date,
                        drug_name=drug.name,
                        evidence=ClinicalEvidenceQuote(
                            claim=event_type,
                            quote=drug.evidence,
                            source_section=drug.source,
                            event_date=event_date,
                            source_kind="patient_record",
                        ),
                    )
                )
                if not event_date:
                    missing.append(f"{drug.name}:{event_type}_date")
        dated_labs = []
        for lab in labs.entries:
            events.append(
                DiliTimelineEvent(
                    event_type="laboratory_result",
                    event_date=lab.sample_date,
                    marker=lab.marker_name.upper(),
                    value=lab.value,
                    uln=lab.upper_limit_normal,
                    evidence=ClinicalEvidenceQuote(
                        claim=f"{lab.marker_name} result",
                        quote=lab.evidence,
                        source_section=lab.source,
                        event_date=lab.sample_date,
                        source_kind="patient_record",
                    ),
                )
            )
            if lab.sample_date and lab.value is not None:
                dated_labs.append(lab)
        dated_labs.sort(key=lambda item: item.sample_date or "")
        first_date = dated_labs[0].sample_date if dated_labs else None
        peak_dates: dict[str, str | None] = {}
        for marker in ("ALT", "AST", "ALP", "GGT", "BILIRUBIN", "INR"):
            candidates = [
                item
                for item in dated_labs
                if item.marker_name.upper() == marker and item.value is not None
            ]
            peak_dates[marker] = (
                max(candidates, key=lambda item: float(item.value or 0)).sample_date
                if candidates
                else None
            )
        dechallenge = self._dechallenge(drugs, dated_labs)
        return DiliTimeline(
            events=events,
            first_abnormal_liver_test_date=first_date,
            jaundice_or_bilirubin_rise_date=peak_dates["BILIRUBIN"],
            peak_dates=peak_dates,
            dechallenge_status=dechallenge,
            last_abnormal_date=dated_labs[-1].sample_date if dated_labs else None,
            missing_fields=sorted(set(missing)),
        )

    @staticmethod
    def _dechallenge(drugs: list[DrugEntry], labs: list) -> str:
        stop_dates = [item.suspension_date for item in drugs if item.suspension_date]
        alt_labs = [item for item in labs if item.marker_name.upper() == "ALT"]
        if not stop_dates or len(alt_labs) < 2:
            return "no_follow_up"
        stop = max(stop_dates)
        after = [item for item in alt_labs if item.sample_date and item.sample_date >= stop]
        if len(after) < 2:
            return "insufficient_interval"
        first, last = float(after[0].value), float(after[-1].value)
        if last < first:
            return "improving_after_stop"
        if last > first:
            return "worsening_after_stop"
        return "stable_abnormality"
