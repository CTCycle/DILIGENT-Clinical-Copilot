from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from domain.clinical.entities import RucamComponentAssessment


###############################################################################
@dataclass(slots=True)
class RucamAnchor:
    onset_date: date | None
    used_alt: float | None
    used_alt_uln: float | None
    used_alp: float | None
    used_alp_uln: float | None
    rationale: str
    source: Literal["qualifying_lab", "onset_context", "visit_proxy", "none"] = "none"
    is_score_eligible: bool = False


@dataclass(slots=True)
class RucamSourceReportedScore:
    score: int
    causality_category: str | None
    source_name: str
    evidence: str


@dataclass(slots=True)
class RucamDataSufficiency:
    sufficient: bool
    blocking_reasons: list[str]


@dataclass(slots=True)
class RucamStructuredCalculation:
    total_score: int
    components: list[RucamComponentAssessment]
    limitations: list[str]
