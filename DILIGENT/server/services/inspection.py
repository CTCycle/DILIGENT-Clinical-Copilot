from __future__ import annotations

from datetime import date
from typing import Any

from DILIGENT.server.common.constants import ARCHIVES_PATH
from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.repositories.serialization.data import DataSerializer
from DILIGENT.server.services.jobs import JobManager, job_manager
from DILIGENT.server.services.updater.livertox import LiverToxUpdater
from DILIGENT.server.services.updater.rxnav import RxNavDrugCatalogBuilder


###############################################################################
class DataInspectionService:
    RXNAV_JOB_TYPE = "rxnav_update"
    LIVERTOX_JOB_TYPE = "livertox_update"

    def __init__(
        self,
        *,
        serializer: DataSerializer | None = None,
        jobs: JobManager = job_manager,
    ) -> None:
        self.serializer = serializer or DataSerializer()
        self.jobs = jobs

    # -------------------------------------------------------------------------
    def list_sessions(
        self,
        *,
        search: str | None,
        status_filter: str | None,
        date_mode: str | None,
        filter_date: date | None,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        items, total = self.serializer.list_sessions(
            search=search,
            status_filter=status_filter,
            date_mode=date_mode,
            filter_date=filter_date,
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
    def get_session_report(self, session_id: int) -> str | None:
        return self.serializer.get_session_report(session_id)

    # -------------------------------------------------------------------------
    def delete_session(self, session_id: int) -> bool:
        return self.serializer.delete_session(session_id)

    # -------------------------------------------------------------------------
    def list_rxnav_catalog(
        self,
        *,
        search: str | None,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        items, total = self.serializer.list_rxnav_catalog(
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
    def get_rxnav_alias_groups(self, drug_id: int) -> dict[str, Any] | None:
        return self.serializer.get_rxnav_alias_groups(drug_id)

    # -------------------------------------------------------------------------
    def list_livertox_catalog(
        self,
        *,
        search: str | None,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        items, total = self.serializer.list_livertox_catalog(
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
    def get_livertox_excerpt(self, drug_id: int) -> dict[str, Any] | None:
        return self.serializer.get_livertox_excerpt(drug_id)

    # -------------------------------------------------------------------------
    def delete_drug(self, drug_id: int) -> bool:
        return self.serializer.delete_drug_with_cleanup(drug_id)

    # -------------------------------------------------------------------------
    def report_job_progress(self, *, job_id: str, progress: float, message: str) -> None:
        bounded_progress = min(100.0, max(0.0, float(progress)))
        self.jobs.update_progress(job_id, bounded_progress)
        self.jobs.update_result(job_id, {"progress_message": message})

    # -------------------------------------------------------------------------
    def run_rxnav_update_job(self, job_id: str) -> dict[str, Any]:
        stop_check = lambda: self.jobs.should_stop(job_id)

        def progress_callback(progress: float, message: str) -> None:
            self.report_job_progress(job_id=job_id, progress=progress, message=message)

        if stop_check():
            return {}
        self.report_job_progress(
            job_id=job_id,
            progress=1.0,
            message="Initializing RxNav catalog refresh",
        )
        builder = RxNavDrugCatalogBuilder(serializer=self.serializer)
        result = builder.update_drug_catalog(
            progress_callback=progress_callback,
            should_stop=stop_check,
        )
        self.report_job_progress(
            job_id=job_id,
            progress=100.0,
            message="RxNav catalog update completed",
        )
        return {"summary": result}

    # -------------------------------------------------------------------------
    def run_livertox_update_job(self, job_id: str) -> dict[str, Any]:
        stop_check = lambda: self.jobs.should_stop(job_id)

        def progress_callback(progress: float, message: str) -> None:
            self.report_job_progress(job_id=job_id, progress=progress, message=message)

        if stop_check():
            return {}
        self.report_job_progress(
            job_id=job_id,
            progress=1.0,
            message="Initializing LiverTox refresh",
        )
        updater = LiverToxUpdater(
            ARCHIVES_PATH,
            redownload=False,
            serializer=self.serializer,
        )
        result = updater.update_from_livertox(
            progress_callback=progress_callback,
            should_stop=stop_check,
        )
        self.report_job_progress(
            job_id=job_id,
            progress=100.0,
            message="LiverTox update completed",
        )
        return {"summary": result}

    # -------------------------------------------------------------------------
    def start_update_job(self, job_type: str) -> dict[str, Any]:
        if self.jobs.is_job_running(job_type):
            raise ValueError(f"Job type '{job_type}' is already running")
        if job_type == self.RXNAV_JOB_TYPE:
            runner = self.run_rxnav_update_job
        elif job_type == self.LIVERTOX_JOB_TYPE:
            runner = self.run_livertox_update_job
        else:
            raise ValueError(f"Unsupported job type: {job_type}")
        job_id = self.jobs.start_job(job_type=job_type, runner=runner)
        status_payload = self.jobs.get_job_status(job_id)
        if status_payload is None:
            raise RuntimeError(f"Failed to initialize {job_type} job")
        return status_payload

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str, *, expected_type: str) -> dict[str, Any] | None:
        payload = self.jobs.get_job_status(job_id)
        if payload is None:
            return None
        job_type = str(payload.get("job_type") or "")
        if job_type != expected_type:
            logger.warning(
                "Job type mismatch for %s: expected %s, got %s",
                job_id,
                expected_type,
                job_type,
            )
            return None
        return payload

    # -------------------------------------------------------------------------
    def cancel_job(self, job_id: str, *, expected_type: str) -> bool:
        payload = self.get_job_status(job_id, expected_type=expected_type)
        if payload is None:
            return False
        return self.jobs.cancel_job(job_id)

