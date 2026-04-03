from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(slots=True)
class RucamAnchor:
    onset_date: date | None
    used_alt: float | None
    used_alt_uln: float | None
    used_alp: float | None
    used_alp_uln: float | None
    rationale: str
