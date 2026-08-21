from __future__ import annotations

from services.llm.context_budget import (
    ContextSegment,
    build_context_plan,
    calculate_input_budget,
    estimate_tokens,
)

###############################################################################
def test_estimate_tokens_is_deterministic_and_nonzero_for_text() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("alpha beta") == estimate_tokens("alpha beta")
    assert estimate_tokens("alpha beta") > 0

###############################################################################
def test_exact_fit_selects_required_and_optional_segments() -> None:
    segments = [
        ContextSegment(key="required", text="a" * 40, priority=100, required=True),
        ContextSegment(key="optional", text="b" * 40, priority=10),
    ]
    budget = sum(segment.estimated_tokens for segment in segments)

    plan = build_context_plan(segments, input_budget=budget)

    assert plan.status == "complete"
    assert [segment.key for segment in plan.selected] == ["required", "optional"]
    assert plan.omitted == ()

###############################################################################
def test_priority_selection_omits_lower_priority_optional_context() -> None:
    segments = [
        ContextSegment(key="low", text="low " * 20, priority=10),
        ContextSegment(key="high", text="high " * 20, priority=20),
    ]

    plan = build_context_plan(segments, input_budget=estimate_tokens(segments[1].text))

    assert [segment.key for segment in plan.selected] == ["high"]
    assert [segment.key for segment in plan.omitted] == ["low"]

###############################################################################
def test_duplicate_segments_keep_the_required_or_higher_priority_variant() -> None:
    plan = build_context_plan(
        [
            ContextSegment(
                key="raw",
                text="same clinical evidence",
                priority=10,
                deduplication_key="evidence-1",
            ),
            ContextSegment(
                key="structured",
                text="same clinical evidence",
                priority=20,
                required=True,
                deduplication_key="evidence-1",
            ),
        ],
        input_budget=100,
    )

    assert plan.deduplicated_count == 1
    assert [segment.key for segment in plan.selected] == ["structured"]

###############################################################################
def test_required_overflow_is_explicit() -> None:
    plan = build_context_plan(
        [ContextSegment(key="required", text="x" * 100, priority=100, required=True)],
        input_budget=1,
    )

    assert plan.status == "required_overflow"
    assert plan.required_overflow is True
    assert plan.omitted == ()

###############################################################################
def test_unknown_capacity_does_not_invent_a_context_ceiling() -> None:
    plan = build_context_plan(
        [ContextSegment(key="one", text="one", priority=1)],
        input_budget=None,
    )

    assert plan.status == "unknown_capacity"
    assert [segment.key for segment in plan.selected] == ["one"]

###############################################################################
def test_reserve_exhaustion_never_makes_input_budget_negative() -> None:
    assert calculate_input_budget(
        context_limit=100,
        visible_output_reserve=80,
        reasoning_reserve=40,
        safety_reserve=20,
    ) == 0
