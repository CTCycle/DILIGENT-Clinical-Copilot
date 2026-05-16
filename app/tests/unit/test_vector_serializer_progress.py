from __future__ import annotations

from repositories.serialization.vectors import VectorSerializer


def test_batch_progress_scales_through_embedding_window() -> None:
    events: list[tuple[float, str]] = []
    serializer = object.__new__(VectorSerializer)
    serializer.progress_callback = lambda progress, message: events.append(
        (progress, message)
    )

    serializer.report_batch_progress(completed_batches=1, total_batches=4)
    serializer.report_batch_progress(completed_batches=4, total_batches=4)

    assert events == [
        (44.5, "Embedded and persisted batch 1/4"),
        (88.0, "Embedded and persisted batch 4/4"),
    ]
