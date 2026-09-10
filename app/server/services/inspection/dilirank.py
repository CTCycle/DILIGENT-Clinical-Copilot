from __future__ import annotations

from functools import partial
from typing import Any

from configurations.startup import get_server_settings
from repositories.context import RepositoryContext
from repositories.dilirank_repository import DiliRankRepository
from services.runtime.jobs import JobManager
from services.updater.dilirank import DiliRankUpdater

###############################################################################
class DiliRankProgressReporter:

    # -------------------------------------------------------------------------
    def __init__(self, jobs: JobManager, job_id: str) -> None:
        self.jobs = jobs
        self.job_id = job_id

    # -------------------------------------------------------------------------
    def __call__(self, progress: float, message: str) -> None:
        self.jobs.update_progress(self.job_id, min(100.0, max(0.0, float(progress))))
        payload = self.jobs.get_job_status(self.job_id) or {}
        result = dict(payload.get("result") or {})
        result["progress_message"] = message
        self.jobs.update_result(self.job_id, result)

###############################################################################
class DiliRankInspectionService:
    JOB_TYPE = "dilirank_update"

    # -------------------------------------------------------------------------
    def __init__(self, *, context: RepositoryContext, jobs: JobManager) -> None:
        self.repository = DiliRankRepository(context)
        self.jobs = jobs

    # -------------------------------------------------------------------------
    @staticmethod
    def build_update_config_response() -> dict[str, Any]:
        return {
            "target": "dilirank",
            "defaults": {"redownload": False},
            "allowed_fields": ["redownload"],
            "summary": {},
            "read_only": False,
        }

    # -------------------------------------------------------------------------
    def list_catalog(
        self,
        *,
        search: str | None,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        items, total = self.repository.list_catalog(
            search=search,
            offset=offset,
            limit=limit,
        )
        return {
            "items": items,
            "total": total,
            "offset": max(int(offset), 0),
            "limit": max(int(limit), 1),
        }

    # -------------------------------------------------------------------------
    def get_drug_records(self, drug_id: int) -> dict[str, Any] | None:
        records = self.repository.get_records_for_drug(drug_id)
        if not records:
            return None
        return {
            "drug_id": int(drug_id),
            "drug_name": records[0]["drug_name"],
            "records": records,
        }

    # -------------------------------------------------------------------------
    def run_update_job(
        self,
        job_id: str,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        override_values = dict(overrides or {})
        reporter = DiliRankProgressReporter(self.jobs, job_id)
        updater = DiliRankUpdater(
            repository=self.repository,
            redownload=bool(override_values.get("redownload", False)),
        )
        result = updater.update_from_fda(
            progress_callback=reporter,
            should_stop=partial(self.jobs.should_stop, job_id),
        )
        return {"summary": result, "progress_message": "DILIrank 2.0 update completed"}

    # -------------------------------------------------------------------------
    def start_update_job(
        self,
        job_type: str,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if job_type != self.JOB_TYPE:
            raise ValueError(f"Unsupported job type: {job_type}")
        scope_key = f"catalog:{job_type}"
        if self.jobs.is_job_running(job_type, scope_key=scope_key):
            raise ValueError(f"Job type '{job_type}' is already running")
        runner = partial(self.run_update_job, overrides=dict(overrides or {}))
        job_id = self.jobs.start_job(
            job_type=job_type,
            runner=runner,
            scope_key=scope_key,
        )
        payload = self.jobs.get_job_status(job_id)
        if payload is None:
            raise RuntimeError("Failed to initialize DILIrank 2.0 update job")
        payload["poll_interval"] = get_server_settings().jobs.polling_interval
        return payload

    # -------------------------------------------------------------------------
    def get_job_status(
        self,
        job_id: str,
        *,
        expected_type: str,
    ) -> dict[str, Any] | None:
        payload = self.jobs.get_job_status(job_id)
        if payload is None or str(payload.get("job_type") or "") != expected_type:
            return None
        return payload

    # -------------------------------------------------------------------------
    def cancel_job(self, job_id: str, *, expected_type: str) -> bool:
        if self.get_job_status(job_id, expected_type=expected_type) is None:
            return False
        return self.jobs.cancel_job(job_id) is not None
