from __future__ import annotations

import hashlib
from typing import Any

from domain.clinical.entities import PipelineIssue

###############################################################################
def build_source_hash(text: str) -> str:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

###############################################################################
def validate_span(text: str, start: int, end: int, expected: str) -> bool:
    if start < 0 or end < start or end > len(text):
        return False
    return text[start:end] == expected

###############################################################################
def build_section_audit_record(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)

###############################################################################
def build_extraction_decision_record(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)

###############################################################################
def build_tool_call_record(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)

###############################################################################
def append_pipeline_issue(
    issues: list[PipelineIssue],
    *,
    severity: str = "warning",
    code: str,
    message: str,
    field: str | None = None,
    raw_line: str | None = None,
) -> PipelineIssue:
    issue = PipelineIssue(
        severity="error" if severity == "error" else "warning",
        code=code,
        message=message,
        field=field,
        raw_line=raw_line,
    )
    issues.append(issue)
    return issue
