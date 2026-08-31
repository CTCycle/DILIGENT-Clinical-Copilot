from __future__ import annotations

import re

RECHALLENGE_RECOMMENDATION_CODE = "clinical_narrative_recommends_rechallenge"
RECHALLENGE_RECOMMENDATION_MESSAGE = (
    "Generated clinical text recommends or permits drug rechallenge or re-exposure; "
    "rechallenge is never recommended."
)

_RECHALLENGE_TERM = r"(?:rechallenge|re[- ]?exposure|reintroduc\w*|restart\w*|resum\w*)"
_PERMISSIVE_ACTION = (
    r"(?:consider(?:ed|ing)?|may|might|could|can|should|"
    r"recommend(?:ed|ation)?|reasonable|appropriate|trial|attempt|"
    r"permiss\w*|allow\w*|advis\w*|warrant\w*|justif\w*|"
    r"under observation|under close monitoring)"
)
_PERMISSIVE_RE = re.compile(
    rf"(?:\b{_PERMISSIVE_ACTION}\b[^.!?;]{{0,140}}\b{_RECHALLENGE_TERM}\b|"
    rf"\b{_RECHALLENGE_TERM}\b[^.!?;]{{0,140}}\b{_PERMISSIVE_ACTION}\b)",
    re.IGNORECASE,
)
_NEGATIVE_RE = re.compile(
    r"\b(?:no|not|never|without|avoid\w*|contraindicat\w*|"
    r"prohibit\w*|must\s+not|should\s+not|do\s+not|don't)\b",
    re.IGNORECASE,
)
_NEGATIVE_RECHALLENGE_RE = re.compile(
    rf"(?:\b(?:no|not|never|without|avoid\w*|contraindicat\w*|"
    rf"prohibit\w*|must\s+not|should\s+not|do\s+not|don't)\b"
    rf"[^.!?;]{{0,80}}\b{_RECHALLENGE_TERM}\b|"
    rf"\b{_RECHALLENGE_TERM}\b[^.!?;]{{0,80}}\b(?:not\b|never\b|"
    r"did\s+not\s+(?:occur|happen|take\s+place)|was\s+not\b|"
    r"is\s+absent\b|is\s+unavailable\b|is\s+undocumented\b))",
    re.IGNORECASE,
)


###############################################################################
def _sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?;])\s+|\r?\n+", normalized)
        if sentence.strip()
    ]


###############################################################################
def contains_rechallenge_recommendation(text: str | None) -> bool:
    """Detect permissive recommendations to rechallenge or re-expose a drug."""

    for sentence in _sentences(text or ""):
        if not re.search(rf"\b{_RECHALLENGE_TERM}\b", sentence, re.IGNORECASE):
            continue
        for match in _PERMISSIVE_RE.finditer(sentence):
            clause = match.group(0)
            context = sentence[max(0, match.start() - 18) : match.end() + 40]
            if _NEGATIVE_RE.search(context):
                connector = re.search(r"\b(?:but|however|yet|although)\b", clause)
                if connector and re.search(
                    rf"\b{_PERMISSIVE_ACTION}\b",
                    clause[connector.end() :],
                    re.IGNORECASE,
                ):
                    return True
                continue
            return True
    return False


###############################################################################
def contains_explicitly_negative_rechallenge(text: str | None) -> bool:
    """Identify source wording that explicitly says rechallenge did not occur."""

    return any(
        _NEGATIVE_RECHALLENGE_RE.search(sentence) for sentence in _sentences(text or "")
    )
