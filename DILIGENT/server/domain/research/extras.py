from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from DILIGENT.server.domain.research.entities import ResearchSource


###############################################################################
@dataclass(slots=True)
class BraveSearchOutcome:
    normalized_query: str
    sources: list[ResearchSource]
    message: str | None = None
    usage: dict[str, Any] | None = None
