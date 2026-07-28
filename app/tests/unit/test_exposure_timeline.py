from __future__ import annotations

from datetime import date

from domain.clinical import DrugEntry
from services.clinical.exposure_timeline import ExposureTimelineService


def test_anamnesis_mention_is_uncertain_current_exposure() -> None:
    suspension = ExposureTimelineService().evaluate_suspension(
        DrugEntry(name="Paracetamol", source="anamnesis", historical_flag=True),
        visit_date=date(2025, 4, 14),
    )

    assert suspension.suspended is False
    assert suspension.note is not None
    assert "Historical mention from anamnesis" in suspension.note
    assert "Active therapy; no suspension reported." not in suspension.note


def test_partial_timeline_date_uses_visit_year() -> None:
    parsed = ExposureTimelineService().parse_timeline_date(
        "14-04", visit_date=date(2025, 4, 20)
    )

    assert parsed == date(2025, 4, 14)


def test_suspension_interval_is_retained_for_latency_comparison() -> None:
    suspension = ExposureTimelineService().evaluate_suspension(
        DrugEntry(name="Drug", suspension_status=True, suspension_date="2025-04-01"),
        visit_date=date(2025, 4, 14),
    )

    assert suspension.interval_days == 13
    assert suspension.suspension_date == date(2025, 4, 1)
    assert "13 days before the visit" in (suspension.note or "")


def test_future_suspension_is_treated_as_ongoing_exposure() -> None:
    suspension = ExposureTimelineService().evaluate_suspension(
        DrugEntry(name="Drug", suspension_status=True, suspension_date="2025-04-20"),
        visit_date=date(2025, 4, 14),
    )

    assert suspension.interval_days == -6
    assert "ongoing exposure" in (suspension.note or "")


def test_start_interval_and_prompt_are_deterministic() -> None:
    service = ExposureTimelineService()
    suspension = service.evaluate_suspension(
        DrugEntry(name="Drug", therapy_start_status=True, therapy_start_date="2025-04-01"),
        visit_date=date(2025, 4, 14),
    )

    assert suspension.start_interval_days == 13
    assert "roughly 13 days before the visit" in service.format_start_prompt(suspension)
    assert service.format_suspension_prompt(suspension) == "Active therapy; no suspension reported."


def test_format_visit_date_anchor_covers_missing_and_present_values() -> None:
    assert ExposureTimelineService.format_visit_date_anchor(None) == "Not provided."
    assert ExposureTimelineService.format_visit_date_anchor(date(2025, 4, 14)) == "2025-04-14"
