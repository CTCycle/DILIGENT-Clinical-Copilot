from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from domain.clinical.entities import HepatotoxicityPatternScore, PipelineIssue

PatternSource = Literal["provided", "calculated", "undetermined"]

###############################################################################
class HepaticPatternResolutionInput(BaseModel):
    explicit_pattern: str | None = None
    calculated_pattern: str | None = None
    r_score: float | None = None

###############################################################################
class HepaticPatternResolutionResult(BaseModel):
    explicit_value: str | None = None
    calculated_value: str | None = None
    final_value: str = "indeterminate"
    source: PatternSource = "undetermined"
    conflict: bool = False
    r_score: float | None = None
    warnings: list[PipelineIssue] = Field(default_factory=list)

###############################################################################
def _normalize_pattern(value: str | None) -> str | None:
    normalized = (value or "").strip().lower().replace(" ", "_")
    if normalized in {"hepatocellular", "cholestatic", "mixed", "indeterminate"}:
        return normalized
    return None

###############################################################################
def resolve_hepatic_pattern(
    resolution_input: HepaticPatternResolutionInput,
) -> HepaticPatternResolutionResult:
    explicit = _normalize_pattern(resolution_input.explicit_pattern)
    calculated = _normalize_pattern(resolution_input.calculated_pattern)
    warnings: list[PipelineIssue] = []
    conflict = bool(explicit and calculated and explicit != calculated)
    if conflict:
        warnings.append(
            PipelineIssue(
                severity="warning",
                code="hepatic_pattern_source_calculation_conflict",
                field="laboratory_analysis",
                message=(
                    "Source-provided hepatic pattern differs from calculated pattern "
                    f"({explicit} vs {calculated}; R ratio={resolution_input.r_score})."
                ),
            )
        )
    if explicit:
        final_value = explicit
        source: PatternSource = "provided"
    elif calculated:
        final_value = calculated
        source = "calculated"
    else:
        final_value = "indeterminate"
        source = "undetermined"
    return HepaticPatternResolutionResult(
        explicit_value=explicit,
        calculated_value=calculated,
        final_value=final_value,
        source=source,
        conflict=conflict,
        r_score=resolution_input.r_score,
        warnings=warnings,
    )

###############################################################################
def copy_pattern_score_with_resolution(
    pattern_score: HepatotoxicityPatternScore,
    resolution: HepaticPatternResolutionResult,
) -> HepatotoxicityPatternScore:
    return pattern_score.model_copy(update={"classification": resolution.final_value})
