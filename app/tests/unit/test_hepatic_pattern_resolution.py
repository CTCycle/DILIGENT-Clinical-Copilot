from services.clinical.pattern_resolution import (
    HepaticPatternResolutionInput,
    resolve_hepatic_pattern,
)

###############################################################################
def test_explicit_pattern_is_preserved_without_overwriting_calculated_value() -> None:
    result = resolve_hepatic_pattern(
        HepaticPatternResolutionInput(
            explicit_pattern="mixed",
            calculated_pattern="hepatocellular",
            r_score=6.2,
        )
    )

    assert result.final_value == "mixed"
    assert result.calculated_value == "hepatocellular"
    assert result.source == "provided"
    assert result.conflict is True
    assert result.warnings[0].code == "hepatic_pattern_source_calculation_conflict"


###############################################################################
def test_explicit_pattern_conflict_keeps_calculation_visible() -> None:
    result = resolve_hepatic_pattern(
        HepaticPatternResolutionInput(
            explicit_pattern="cholestatic",
            calculated_pattern="hepatocellular",
            r_score=7.5,
        )
    )

    assert result.final_value == "cholestatic"
    assert result.calculated_value == "hepatocellular"
    assert result.source == "provided"
    assert result.conflict is True
    assert "7.5" in result.warnings[0].message

###############################################################################
def test_calculated_and_indeterminate_pattern_resolution() -> None:
    calculated = resolve_hepatic_pattern(
        HepaticPatternResolutionInput(calculated_pattern="cholestatic")
    )
    missing = resolve_hepatic_pattern(HepaticPatternResolutionInput())

    assert calculated.final_value == "cholestatic"
    assert calculated.source == "calculated"
    assert missing.final_value == "indeterminate"
    assert missing.source == "undetermined"
