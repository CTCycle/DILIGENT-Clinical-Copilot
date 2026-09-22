from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from services.inspection.update_jobs import DataInspectionUpdateJobRunner
from services.runtime.jobs import JobManager
from services.runtime.state import JobState


###############################################################################
def build_runner(jobs: JobManager) -> DataInspectionUpdateJobRunner:
    return DataInspectionUpdateJobRunner(
        drug_catalog_repository=object(),  # type: ignore[arg-type]
        knowledge_repository=object(),  # type: ignore[arg-type]
        dilirank_repository=object(),  # type: ignore[arg-type]
        jobs=jobs,
        report_phase_by_target=lambda *_args: None,
        report_job_progress=lambda *_args, **_kwargs: None,
        write_rag_manifest=lambda *_args: Path("rag_index_manifest.json"),
    )

###############################################################################
def make_job(jobs: JobManager, job_id: str = "structured-job") -> str:
    jobs.jobs[job_id] = JobState(
        job_id=job_id,
        job_type="structured_sources_update",
        status="running",
        scope_key="catalog:structured_sources",
    )
    return job_id

###############################################################################
def test_structured_sources_run_in_dependency_order() -> None:
    jobs = JobManager()
    runner = build_runner(jobs)
    job_id = make_job(jobs)
    calls: list[tuple[str, bool]] = []

    def child(
        target: str,
        summary: dict[str, Any],
    ):
        def run(
            _job_id: str,
            _overrides: dict[str, Any],
            *,
            progress_callback,
            report_phase: bool,
        ) -> dict[str, Any]:
            calls.append((target, report_phase))
            progress_callback(100, f"{target} complete")
            return {"summary": summary}

        return run

    runner.run_rxnav_update_job = child("rxnav", {"records": 1})  # type: ignore[method-assign]
    runner.run_livertox_update_job = child("livertox", {"records": 2})  # type: ignore[method-assign]
    runner.run_dilirank_update_job = child("dilirank", {"records": 3})  # type: ignore[method-assign]

    result = runner.run_structured_sources_update_job(
        job_id,
        {
            "rxnav": {"rxnav_max_concurrency": 2},
            "livertox": {"redownload": True},
            "dilirank": {"redownload": False},
        },
    )

    assert calls == [("rxnav", False), ("livertox", False), ("dilirank", False)]
    assert [
        result["sources"][target]["status"]
        for target in ("rxnav", "livertox", "dilirank")
    ] == ["completed", "completed", "completed"]
    assert result["summaries"]["dilirank"] == {"records": 3}
    assert jobs.get_job_status(job_id)["progress"] == 100

###############################################################################
def test_structured_sources_cancellation_does_not_start_following_sources() -> None:
    jobs = JobManager()
    runner = build_runner(jobs)
    job_id = make_job(jobs)
    calls: list[str] = []

    def rxnav_child(
        _job_id: str,
        _overrides: dict[str, Any],
        *,
        progress_callback,
        report_phase: bool,
    ) -> dict[str, Any]:
        _ = report_phase
        calls.append("rxnav")
        progress_callback(40, "RxNav interrupted")
        jobs.jobs[job_id].stop_requested = True
        return {"summary": {"records": 1}}

    runner.run_rxnav_update_job = rxnav_child  # type: ignore[method-assign]
    runner.run_livertox_update_job = lambda *_args, **_kwargs: calls.append("livertox")  # type: ignore[method-assign]
    runner.run_dilirank_update_job = lambda *_args, **_kwargs: calls.append("dilirank")  # type: ignore[method-assign]

    result = runner.run_structured_sources_update_job(job_id)

    assert calls == ["rxnav"]
    assert result["sources"]["rxnav"]["status"] == "cancelled"
    assert result["sources"]["livertox"]["status"] == "cancelled"
    assert result["sources"]["dilirank"]["status"] == "cancelled"

###############################################################################
def test_structured_source_cancellation_exception_is_not_reported_as_failure() -> None:
    jobs = JobManager()
    runner = build_runner(jobs)
    job_id = make_job(jobs)
    calls: list[str] = []

    def cancel_rxnav(*_args, **_kwargs):
        calls.append("rxnav")
        jobs.jobs[job_id].stop_requested = True
        raise RuntimeError("RxNav update cancelled by user request")

    runner.run_rxnav_update_job = cancel_rxnav  # type: ignore[method-assign]
    runner.run_livertox_update_job = lambda *_args, **_kwargs: calls.append("livertox")  # type: ignore[method-assign]
    runner.run_dilirank_update_job = lambda *_args, **_kwargs: calls.append("dilirank")  # type: ignore[method-assign]

    result = runner.run_structured_sources_update_job(job_id)

    assert calls == ["rxnav"]
    assert [
        result["sources"][target]["status"]
        for target in ("rxnav", "livertox", "dilirank")
    ] == ["cancelled", "cancelled", "cancelled"]
    assert all(
        result["sources"][target]["error"] is None for target in result["sources"]
    )
    assert jobs.get_job_status(job_id)["result"]["progress_message"] == (
        "Cancellation requested for the structured source updates."
    )

###############################################################################
def test_structured_source_failure_marks_remaining_sources_skipped() -> None:
    jobs = JobManager()
    runner = build_runner(jobs)
    job_id = make_job(jobs)

    def fail_livertox(*_args, **_kwargs):
        raise RuntimeError("archive unavailable")

    runner.run_rxnav_update_job = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "summary": {"records": 1}
    }
    runner.run_livertox_update_job = fail_livertox  # type: ignore[method-assign]
    runner.run_dilirank_update_job = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "summary": {"records": 3}
    }

    with pytest.raises(RuntimeError, match="archive unavailable"):
        runner.run_structured_sources_update_job(job_id)

    result = jobs.get_job_status(job_id)["result"]
    assert result["sources"]["rxnav"]["status"] == "completed"
    assert result["sources"]["livertox"]["status"] == "failed"
    assert result["sources"]["dilirank"]["status"] == "failed"

###############################################################################
@pytest.mark.parametrize("failing_target", ["rxnav", "livertox", "dilirank"])
def test_structured_source_failure_stops_at_each_ordered_boundary(
    failing_target: str,
) -> None:
    jobs = JobManager()
    runner = build_runner(jobs)
    job_id = make_job(jobs)
    targets = ("rxnav", "livertox", "dilirank")
    calls: list[str] = []

    def child(target: str):
        def run(
            _job_id: str,
            _overrides: dict[str, Any],
            *,
            progress_callback,
            report_phase: bool,
        ) -> dict[str, Any]:
            assert report_phase is False
            calls.append(target)
            if target == failing_target:
                raise RuntimeError(f"{target} refresh failed")
            progress_callback(100, f"{target} complete")
            return {"summary": {"records": targets.index(target) + 1}}

        return run

    runner.run_rxnav_update_job = child("rxnav")  # type: ignore[method-assign]
    runner.run_livertox_update_job = child("livertox")  # type: ignore[method-assign]
    runner.run_dilirank_update_job = child("dilirank")  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match=f"{failing_target} refresh failed"):
        runner.run_structured_sources_update_job(job_id)

    result = jobs.get_job_status(job_id)["result"]
    failed_index = targets.index(failing_target)
    assert calls == list(targets[: failed_index + 1])
    assert [result["sources"][target]["status"] for target in targets] == [
        "completed" if index < failed_index else "failed"
        for index in range(len(targets))
    ]
    assert set(result["summaries"]) == set(targets[:failed_index])
    assert all(
        result["sources"][target]["message"]
        == "Skipped because an earlier source update failed."
        for target in targets[failed_index + 1 :]
    )

###############################################################################
def test_structured_sources_cancelled_while_pending_do_not_start_any_source() -> None:
    jobs = JobManager()
    runner = build_runner(jobs)
    job_id = make_job(jobs)
    jobs.jobs[job_id].stop_requested = True
    calls: list[str] = []
    runner.run_rxnav_update_job = lambda *_args, **_kwargs: calls.append("rxnav")  # type: ignore[method-assign]
    runner.run_livertox_update_job = lambda *_args, **_kwargs: calls.append("livertox")  # type: ignore[method-assign]
    runner.run_dilirank_update_job = lambda *_args, **_kwargs: calls.append("dilirank")  # type: ignore[method-assign]

    result = runner.run_structured_sources_update_job(job_id)

    assert calls == []
    assert [
        result["sources"][target]["status"]
        for target in ("rxnav", "livertox", "dilirank")
    ] == ["cancelled", "cancelled", "cancelled"]
