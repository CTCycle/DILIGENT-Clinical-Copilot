from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationMessageBundle:
    missing_anamnesis: str
    missing_visit_date: str
    missing_timed_drug: str
    insufficient_labs: str
