from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from DILIGENT.server.services import inspection as inspection_module
from DILIGENT.server.services.inspection import DataInspectionService
from DILIGENT.server.services.updater.dili_priors import DiliPriorUpdater


###############################################################################
class SerializerStub:
    def save_dili_annotations(self, frame: pd.DataFrame) -> None:
        _ = frame


###############################################################################
class JobsStub:
    def should_stop(self, job_id: str) -> bool:
        _ = job_id
        return False

    def update_progress(self, job_id: str, progress: float) -> None:
        _ = job_id, progress

    def update_result(self, job_id: str, patch: dict[str, Any]) -> None:
        _ = job_id, patch


# -----------------------------------------------------------------------------
def test_drug_labels_update_job_initializes_progress_and_stop_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeDailyMedLabelUpdater:
        def __init__(self, **kwargs: Any) -> None:
            _ = kwargs

        def update_labels(
            self,
            *,
            redownload: bool = False,
            progress_callback=None,
            should_stop=None,
        ) -> dict[str, int]:
            _ = redownload
            assert callable(progress_callback)
            assert callable(should_stop)
            progress_callback(35.0, "processing")
            captured["stop_requested"] = should_stop()
            return {"documents_persisted": 2}

    monkeypatch.setattr(inspection_module, "DailyMedLabelUpdater", FakeDailyMedLabelUpdater)

    service = DataInspectionService(serializer=SerializerStub(), jobs=JobsStub())
    result = service.run_drug_labels_update_job("job-1", overrides={"redownload": False})

    assert result == {"summary": {"documents_persisted": 2}}
    assert captured["stop_requested"] is False


# -----------------------------------------------------------------------------
def test_dili_priors_updater_reports_progress_and_preserves_result_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updater = DiliPriorUpdater(serializer=SerializerStub())

    progress_updates: list[tuple[float, str]] = []
    persisted: dict[str, int] = {}

    monkeypatch.setattr(
        updater,
        "download_dilirank",
        lambda: pd.DataFrame([{"drug_name": "Acetaminophen"}]),
    )
    monkeypatch.setattr(
        updater,
        "download_dilist",
        lambda: pd.DataFrame([{"drug": "Ibuprofen"}]),
    )
    monkeypatch.setattr(
        updater,
        "parse_dilirank",
        lambda frame: frame,
    )
    monkeypatch.setattr(
        updater,
        "parse_dilist",
        lambda frame: frame,
    )
    monkeypatch.setattr(
        updater,
        "match_rows_to_drugs",
        lambda frame, *, source_dataset: (
            frame,
            {
                "downloaded_rows": 1,
                "linked_rows": 1,
                "unmatched_rows": 0,
                "ambiguous_rows": 0,
            },
        ),
    )

    def fake_persist(frame: pd.DataFrame) -> None:
        persisted["rows"] = len(frame.index)

    monkeypatch.setattr(updater, "persist_annotations", fake_persist)

    result = updater.update_from_sources(
        progress_callback=lambda progress, message: progress_updates.append((progress, message)),
        should_stop=lambda: False,
    )

    assert "dilirank" in result
    assert "dilist" in result
    assert persisted["rows"] == 2
    assert any("Downloading DILI prior sources" in message for _, message in progress_updates)
    assert any("Persisting DILI prior annotations" in message for _, message in progress_updates)


# -----------------------------------------------------------------------------
def test_dili_priors_updater_respects_cooperative_cancellation() -> None:
    updater = DiliPriorUpdater(serializer=SerializerStub())

    with pytest.raises(RuntimeError, match="cancelled"):
        updater.update_from_sources(
            should_stop=lambda: True,
        )
