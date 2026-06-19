from __future__ import annotations

from services.extraction_tools.schemas import ExtractionStrategyDecision

UNRESOLVED_DETERMINISTIC_MAX = 0.05
UNRESOLVED_HYBRID_MAX = 0.45
STRUCTURE_DETERMINISTIC_MIN = 0.78
SPAN_COVERAGE_DETERMINISTIC_MIN = 0.7


###############################################################################
def decide_extraction_strategy(
    *,
    section: str,
    meaningful_line_count: int,
    parsed_line_count: int,
    unresolved_line_count: int,
    evidence_span_count: int,
    ambiguity_count: int = 0,
) -> ExtractionStrategyDecision:
    total = max(meaningful_line_count, 1)
    parsed_ratio = min(1.0, max(0.0, parsed_line_count / total))
    unresolved_ratio = min(1.0, max(0.0, unresolved_line_count / total))
    evidence_coverage = min(1.0, max(0.0, evidence_span_count / total))
    structure_score = parsed_ratio
    reasons: list[str] = []
    if (
        unresolved_ratio <= UNRESOLVED_DETERMINISTIC_MAX
        and structure_score >= STRUCTURE_DETERMINISTIC_MIN
        and evidence_coverage >= SPAN_COVERAGE_DETERMINISTIC_MIN
        and ambiguity_count == 0
    ):
        strategy = "deterministic"
        confidence = min(0.98, (structure_score + evidence_coverage) / 2)
        reasons.append("deterministic coverage meets threshold")
    elif unresolved_ratio <= UNRESOLVED_HYBRID_MAX or parsed_line_count > 0:
        strategy = "hybrid"
        confidence = max(0.45, min(0.85, (structure_score + evidence_coverage) / 2))
        reasons.append("deterministic extraction has unresolved or ambiguous fragments")
    else:
        strategy = "llm"
        confidence = 0.35
        reasons.append("deterministic structure is insufficient")
    if ambiguity_count:
        reasons.append(f"{ambiguity_count} ambiguous extraction signals")
    kwargs = {
        "section": section,
        "strategy": strategy,
        "confidence": confidence,
        "unresolved_line_ratio": unresolved_ratio,
        "evidence_span_coverage": evidence_coverage,
        "ambiguity_count": ambiguity_count,
        "reasons": reasons,
        "thresholds": {
            "unresolved_deterministic_max": UNRESOLVED_DETERMINISTIC_MAX,
            "unresolved_hybrid_max": UNRESOLVED_HYBRID_MAX,
            "structure_deterministic_min": STRUCTURE_DETERMINISTIC_MIN,
            "span_coverage_deterministic_min": SPAN_COVERAGE_DETERMINISTIC_MIN,
        },
    }
    if section == "laboratory_history":
        kwargs["laboratory_structure_score"] = structure_score
    else:
        kwargs["therapy_structure_score"] = structure_score
    return ExtractionStrategyDecision(**kwargs)
