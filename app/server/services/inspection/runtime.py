from __future__ import annotations

from typing import Any


###############################################################################
def coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
