from __future__ import annotations

from datetime import UTC, datetime
import logging
import os
from pathlib import Path
import re
import sys
from threading import Lock
from typing import Any

logger = logging.getLogger("diligent.qa.timeline_recovery")
_guard = Lock()
_extractor_calls = 0
_persistence_calls = 0


def _record_event(message: str) -> None:
    log_path = os.environ.get("DILIGENT_QA_TIMELINE_RECOVERY_LOG")
    if not log_path:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{message}\n")


def _is_backend_startup() -> bool:
    if os.environ.get("DILIGENT_QA_TIMELINE_RECOVERY") != "1":
        return False
    if os.name != "nt":
        command_line = " ".join(sys.argv)
    else:
        from ctypes import c_wchar_p, windll

        windll.kernel32.GetCommandLineW.restype = c_wchar_p
        command_line = windll.kernel32.GetCommandLineW()
    return bool(re.search(r"(?i)(?:^|\s)-m\s+uvicorn(?:\s|$)", command_line))


def _install_controlled_faults() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    server_root = repository_root / "app" / "server"
    sys.path.insert(0, str(server_root))

    from domain.patient_timeline import PatientTimeline, PatientTimelineEvent
    from repositories.session_timeline_repository import SessionTimelineRepository
    from services.clinical.timeline import PatientTimelineExtractor
    from services.llm.cloud import LLMError

    original_persist = SessionTimelineRepository.create_session_timeline_record

    async def controlled_extract(
        self: PatientTimelineExtractor,
        *,
        session_id: int,
        source_payload: dict[str, Any],
        runtime_settings: dict[str, Any] | None = None,
    ) -> PatientTimeline:
        global _extractor_calls
        _ = self, source_payload, runtime_settings
        with _guard:
            _extractor_calls += 1
            call_number = _extractor_calls
        if call_number == 1:
            _record_event(
                f"CONTROLLED_TIMELINE_PROVIDER_FAILURE call={call_number} session_id={session_id}"
            )
            logger.warning(
                "CONTROLLED_TIMELINE_PROVIDER_FAILURE call=%s session_id=%s",
                call_number,
                session_id,
            )
            raise LLMError(
                "Controlled provider network failure for timeline validation",
                error_code="network_unavailable",
                retryable=False,
            )
        _record_event(
            f"CONTROLLED_TIMELINE_EXTRACTOR_RESULT call={call_number} session_id={session_id}"
        )
        logger.warning(
            "CONTROLLED_TIMELINE_EXTRACTOR_RESULT call=%s session_id=%s",
            call_number,
            session_id,
        )
        return PatientTimeline(
            session_id=session_id,
            generated_at=datetime.now(UTC),
            events=[
                PatientTimelineEvent(
                    event_id="controlled-retry-event",
                    title="Controlled extractor retry",
                    description="Synthetic retry result for QA only.",
                    event_type="other",
                    timing_type="explicit_date",
                    event_date="2025-01-17",
                    date_precision="day",
                    date_certainty="explicit",
                    extracted_timing_text="2025-01-17",
                    source="qa_controlled_extractor",
                    source_evidence="Synthetic QA source event on 2025-01-17.",
                )
            ],
        )

    def controlled_persist_once(
        self: SessionTimelineRepository,
        session_id: int,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        global _persistence_calls
        with _guard:
            _persistence_calls += 1
            call_number = _persistence_calls
        if call_number == 2:
            _record_event(
                f"CONTROLLED_TIMELINE_PERSISTENCE_FAILURE call={call_number} session_id={session_id}"
            )
            logger.warning(
                "CONTROLLED_TIMELINE_PERSISTENCE_FAILURE call=%s session_id=%s",
                call_number,
                session_id,
            )
            raise RuntimeError("Controlled QA timeline persistence failure.")
        result = original_persist(self, session_id, payload)
        _record_event(
            f"CONTROLLED_TIMELINE_PERSISTENCE_SAVED call={call_number} session_id={session_id}"
        )
        return result

    PatientTimelineExtractor.extract_timeline = controlled_extract  # type: ignore[method-assign]
    SessionTimelineRepository.create_session_timeline_record = controlled_persist_once  # type: ignore[method-assign]
    _record_event(f"CONTROLLED_TIMELINE_RECOVERY_HOOK_INSTALLED pid={os.getpid()}")
    logger.warning("CONTROLLED_TIMELINE_RECOVERY_HOOK_INSTALLED")


if _is_backend_startup():
    _install_controlled_faults()
