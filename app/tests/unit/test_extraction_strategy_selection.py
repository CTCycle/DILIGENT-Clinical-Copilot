from services.extraction_tools.strategy import decide_extraction_strategy


###############################################################################
def test_strategy_selects_deterministic_hybrid_and_llm() -> None:
    deterministic = decide_extraction_strategy(
        section="therapy",
        meaningful_line_count=3,
        parsed_line_count=3,
        unresolved_line_count=0,
        evidence_span_count=3,
    )
    hybrid = decide_extraction_strategy(
        section="therapy",
        meaningful_line_count=4,
        parsed_line_count=2,
        unresolved_line_count=2,
        evidence_span_count=2,
    )
    llm = decide_extraction_strategy(
        section="laboratory_history",
        meaningful_line_count=4,
        parsed_line_count=0,
        unresolved_line_count=4,
        evidence_span_count=0,
    )

    assert deterministic.strategy == "deterministic"
    assert hybrid.strategy == "hybrid"
    assert llm.strategy == "llm"
    assert deterministic.reasons
