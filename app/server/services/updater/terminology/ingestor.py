# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Orchestrates terminology source refreshes with provenance and reports."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from common.utils.logger import logger
from repositories.terminology_repository import TerminologyRepository
from services.updater.terminology.snapshot import (
    ProgressFn,
    StopFn,
    TerminologySource,
    TerminologySourceUnavailable,
)

###############################################################################
class TerminologyIngestor:
    """Refresh sources one at a time; each source commits atomically.

    A failing or unchanged source never blocks the others, and a source is
    only replaced after its complete snapshot has been fetched and staged.
    """

    # -------------------------------------------------------------------------
    def __init__(
        self,
        repository: TerminologyRepository,
        sources: Sequence[TerminologySource],
        *,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.repository = repository
        self.sources = list(sources)
        self.on_change = on_change

    # -------------------------------------------------------------------------
    def run(
        self,
        *,
        force_full: bool,
        progress: ProgressFn,
        should_stop: StopFn,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        total = max(1, len(self.sources))
        for index, source in enumerate(self.sources):
            if should_stop():
                results[source.key] = {"status": "cancelled"}
                continue
            base = 100.0 * index / total

            def report(value: float, message: str, *, _base: float = base) -> None:
                progress(_base + max(0.0, min(100.0, value)) / total, message)

            report(0.0, f"Refreshing {source.key}")
            results[source.key] = self._run_source(source, force_full, report, should_stop)
        failed = sorted(key for key, value in results.items() if value.get("status") == "failed")
        progress(100.0, "Terminology refresh finished" + (f" ({', '.join(failed)} failed)" if failed else ""))
        return {"sources": results, "failed_sources": failed}

    # -------------------------------------------------------------------------
    def _run_source(
        self,
        source: TerminologySource,
        force_full: bool,
        progress: ProgressFn,
        should_stop: StopFn,
    ) -> dict[str, Any]:
        mode = "full" if force_full else "incremental"
        run_id = self.repository.start_run(source=source.key, mode=mode, license_text=source.license)
        started = time.perf_counter()
        version: str | None = None
        try:
            previous = self.repository.last_completed_version(source.key)
            version = source.probe_version()
            if not force_full and version and version == previous:
                return self._skip(run_id, version, started)
            snapshot = source.fetch(progress, should_stop)
            version = snapshot.version
            if not force_full and version and version == previous:
                return self._skip(run_id, version, started)
            if should_stop():
                raise TerminologySourceUnavailable("Refresh cancelled before persistence")
            progress(70.0, f"Persisting {source.key}")
            stats = self.repository.persist_snapshot(snapshot, run_id)
            report: dict[str, Any] = {
                **snapshot.stats,
                **stats.to_dict(),
                "seconds": round(time.perf_counter() - started, 2),
                "attribution": source.attribution,
            }
            self.repository.finish_run(run_id, status="completed", version=version, stats=report)
            logger.info("Terminology source %s refreshed: %s", source.key, report)
            if self.on_change is not None:
                self.on_change()
            progress(100.0, f"{source.key} refreshed")
            return {"status": "completed", "version": version, "run_id": run_id, "stats": report}
        except TerminologySourceUnavailable as exc:
            status = "cancelled" if should_stop() else "failed"
            self.repository.finish_run(run_id, status=status, version=version, stats=None, error=str(exc))
            logger.warning("Terminology source %s %s: %s", source.key, status, exc)
            return {"status": status, "version": version, "run_id": run_id, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - recorded, then surfaced in the report
            self.repository.finish_run(
                run_id, status="failed", version=version, stats=None, error=f"{type(exc).__name__}: {exc}"
            )
            logger.exception("Terminology source %s failed", source.key)
            return {"status": "failed", "version": version, "run_id": run_id, "error": str(exc)}

    # -------------------------------------------------------------------------
    def _skip(self, run_id: int, version: str, started: float) -> dict[str, Any]:
        stats = {"reason": "unchanged", "seconds": round(time.perf_counter() - started, 2)}
        self.repository.finish_run(run_id, status="skipped", version=version, stats=stats)
        return {"status": "skipped", "version": version, "run_id": run_id, "stats": stats}
