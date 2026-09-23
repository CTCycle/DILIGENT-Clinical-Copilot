from __future__ import annotations

import asyncio
import logging
from typing import Any

from services.clinical.timeline import PatientTimelineExtractor

logger = logging.getLogger("diligent.qa.timeline_cancellation")


async def wait_until_cancelled(
    self: PatientTimelineExtractor,
    *,
    session_id: int,
    source_payload: dict[str, Any],
    runtime_settings: dict[str, Any] | None = None,
) -> Any:
    _ = self, source_payload, runtime_settings
    logger.warning("CONTROLLED_TIMELINE_EXTRACTOR_STARTED session_id=%s", session_id)
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        logger.warning("CONTROLLED_TIMELINE_EXTRACTOR_CANCELLED session_id=%s", session_id)
        raise
    raise AssertionError("The controlled timeline extractor must be cancelled.")


def install() -> None:
    PatientTimelineExtractor.extract_timeline = wait_until_cancelled  # type: ignore[method-assign]
    logger.warning("CONTROLLED_TIMELINE_EXTRACTOR_INSTALLED")
