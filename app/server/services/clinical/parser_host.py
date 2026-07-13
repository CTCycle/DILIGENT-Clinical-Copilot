from __future__ import annotations

import asyncio
from typing import Any


###############################################################################
class ParserHost:
    client: Any | None
    model: str
    client_lock: asyncio.Lock
    client_loop_id: int | None
    client_provider: str | None
    runtime_revision: int
    forced_provider: str | None
    forced_model: str | None
    timeout_s: float

    # -------------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)
