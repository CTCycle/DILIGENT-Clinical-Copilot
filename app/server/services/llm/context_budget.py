from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import re
from typing import Literal

ContextPlanStatus = Literal["complete", "unknown_capacity", "required_overflow"]

###############################################################################
def estimate_tokens(text: str) -> int:
    """Estimate tokens without a provider tokenizer, conservatively and deterministically."""
    normalized = str(text or "")
    if not normalized:
        return 0
    word_like = len(re.findall(r"\w+|[^\w\s]", normalized, flags=re.UNICODE))
    character_estimate = math.ceil(len(normalized) / 4)
    return max(1, word_like, character_estimate)

###############################################################################
@dataclass(frozen=True)
class ContextSegment:
    key: str
    text: str
    priority: int
    required: bool = False
    source_kind: str = "unknown"
    deduplication_key: str | None = None
    token_estimate: int | None = None

    # -------------------------------------------------------------------------
    @property
    def estimated_tokens(self) -> int:
        return max(0, int(self.token_estimate or estimate_tokens(self.text)))

    # -------------------------------------------------------------------------
    @property
    def dedupe_key(self) -> str:
        if self.deduplication_key:
            return self.deduplication_key
        normalized = " ".join(self.text.split()).casefold()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

###############################################################################
@dataclass(frozen=True)
class ContextPlan:
    input_budget: int | None
    selected: tuple[ContextSegment, ...]
    omitted: tuple[ContextSegment, ...]
    total_selected_tokens: int
    deduplicated_count: int
    status: ContextPlanStatus
    required_overflow: bool
    selection_report: dict[str, object] = field(default_factory=dict)

###############################################################################
def calculate_input_budget(
    *,
    context_limit: int | None,
    visible_output_reserve: int,
    reasoning_reserve: int,
    safety_reserve: int,
) -> int | None:
    if context_limit is None or context_limit <= 0:
        return None
    reserved = max(0, visible_output_reserve) + max(0, reasoning_reserve) + max(0, safety_reserve)
    return max(0, context_limit - reserved)

###############################################################################
def _deduplicate_segments(segments: list[ContextSegment]) -> tuple[list[ContextSegment], int]:
    selected_by_key: dict[str, tuple[int, ContextSegment]] = {}
    duplicate_count = 0
    for index, segment in enumerate(segments):
        key = segment.dedupe_key
        previous = selected_by_key.get(key)
        if previous is None:
            selected_by_key[key] = (index, segment)
            continue
        duplicate_count += 1
        previous_index, previous_segment = previous
        previous_score = (int(previous_segment.required), previous_segment.priority)
        current_score = (int(segment.required), segment.priority)
        if current_score > previous_score:
            selected_by_key[key] = (index, segment)
        elif current_score == previous_score and index < previous_index:
            selected_by_key[key] = (index, segment)
    deduplicated = [item[1] for item in sorted(selected_by_key.values(), key=lambda item: item[0])]
    return deduplicated, duplicate_count

###############################################################################
def build_context_plan(
    segments: list[ContextSegment],
    *,
    input_budget: int | None,
) -> ContextPlan:
    deduplicated, duplicate_count = _deduplicate_segments(segments)
    required = [segment for segment in deduplicated if segment.required]
    optional = [segment for segment in deduplicated if not segment.required]
    ordered = sorted(
        required,
        key=lambda segment: (-segment.priority, deduplicated.index(segment)),
    ) + sorted(
        optional,
        key=lambda segment: (-segment.priority, deduplicated.index(segment)),
    )

    if input_budget is None:
        selected = tuple(ordered)
        omitted: tuple[ContextSegment, ...] = ()
        status: ContextPlanStatus = "unknown_capacity"
        required_overflow = False
    else:
        selected_list: list[ContextSegment] = []
        omitted_list: list[ContextSegment] = []
        remaining = max(0, input_budget)
        required_overflow = False
        for segment in ordered:
            if segment.required:
                selected_list.append(segment)
                remaining -= segment.estimated_tokens
                if remaining < 0:
                    required_overflow = True
                continue
            if segment.estimated_tokens <= remaining:
                selected_list.append(segment)
                remaining -= segment.estimated_tokens
            else:
                omitted_list.append(segment)
        selected = tuple(selected_list)
        omitted = tuple(omitted_list)
        status = "required_overflow" if required_overflow else "complete"

    total_selected_tokens = sum(segment.estimated_tokens for segment in selected)
    return ContextPlan(
        input_budget=input_budget,
        selected=selected,
        omitted=omitted,
        total_selected_tokens=total_selected_tokens,
        deduplicated_count=len(deduplicated),
        status=status,
        required_overflow=required_overflow,
        selection_report={
            "input_budget": input_budget,
            "selected_segment_count": len(selected),
            "omitted_segment_count": len(omitted),
            "deduplicated_count": duplicate_count,
            "selected_input_tokens": total_selected_tokens,
            "required_overflow": required_overflow,
        },
    )
