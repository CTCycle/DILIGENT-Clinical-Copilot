from __future__ import annotations

import re


###############################################################################
class LiverToxExcerptSanitizer:
    WHITESPACE_RE = re.compile(r"\s+")
    OVERVIEW_RE = re.compile(r"\bOVERVIEW\b", re.IGNORECASE)
    OTHER_LINKS_RE = re.compile(r"\bOTHER REFERENCE LINKS\b", re.IGNORECASE)
    BOILERPLATE_PATTERNS = (
        re.compile(
            r"\blivertox\s+clinical\s+and\s+research\s+information\s+on\s+drug-induced\s+liver\s+injury\b",
            re.IGNORECASE,
        ),
        re.compile(r"\bbooks-source-type\s+database\b", re.IGNORECASE),
        re.compile(
            r"\bnational\s+institute\s+of\s+diabetes\s+and\s+digestive\s+and\s+kidney\s+diseases\b",
            re.IGNORECASE,
        ),
    )

    # -------------------------------------------------------------------------
    def sanitize(self, text: str) -> str:
        normalized = self.normalize_whitespace(text)
        if not normalized:
            return ""

        overview_match = self.OVERVIEW_RE.search(normalized)
        if overview_match:
            normalized = normalized[overview_match.start() :].lstrip()

        links_match = self.OTHER_LINKS_RE.search(normalized)
        if links_match:
            normalized = normalized[: links_match.start()].rstrip()

        for pattern in self.BOILERPLATE_PATTERNS:
            normalized = pattern.sub(" ", normalized)

        normalized = self.normalize_whitespace(normalized)
        return normalized

    # -------------------------------------------------------------------------
    def normalize_whitespace(self, value: str) -> str:
        return self.WHITESPACE_RE.sub(" ", str(value)).strip()

