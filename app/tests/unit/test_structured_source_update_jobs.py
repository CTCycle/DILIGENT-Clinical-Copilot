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
