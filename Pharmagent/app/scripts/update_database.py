from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from typing import Any

from Pharmagent.app.constants import LIVERTOX_ARCHIVE, SOURCES_PATH
from Pharmagent.app.logger import logger
from Pharmagent.app.utils.database.sqlite import database
from Pharmagent.app.utils.services.retrieval import RxNavClient
from Pharmagent.app.utils.services.scraper import LiverToxClient
from Pharmagent.app.utils.serializer import DataSerializer

REDOWNLOAD = True
CONVERT_TO_DATAFRAME = False


###############################################################################
class LiverToxUpdater:
    ###########################################################################
    def __init__(
        self,
        sources_path: str,
        *,
        redownload: bool,
        convert_to_dataframe: bool,
        livertox_client: LiverToxClient | None = None,
        rx_client: RxNavClient | None = None,
        serializer: DataSerializer | None = None,
        database_client=database,
    ) -> None:
        self.sources_path = os.path.abspath(sources_path)
        self.redownload = redownload
        self.convert_to_dataframe = convert_to_dataframe
        self.livertox_client = livertox_client or LiverToxClient()
        self.rx_client = rx_client or RxNavClient()
        self.serializer = serializer or DataSerializer()
        self.database = database_client

    # -----------------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        logger.info("Starting LiverTox update")
        self._ensure_sources_dir()
        archive_path = self._normalize_archive_path()
        if self.redownload:
            logger.info("Redownload flag enabled; fetching latest LiverTox archive")
            download_info = self._download_archive()
            archive_path = os.path.abspath(str(download_info.get("file_path", archive_path)))
        else:
            logger.info("Using existing LiverTox archive")
            download_info = self._collect_local_archive_info(archive_path)
        logger.info("Extracting LiverTox monographs from %s", archive_path)
        extracted = self._extract_monographs(archive_path)
        logger.info("Sanitizing %d extracted entries", len(extracted))
        records = self._sanitize_records(extracted)
        logger.info("Enriching %d sanitized entries with RxNav terms", len(records))
        enriched = self._enrich_records(records)
        logger.info("Persisting enriched records to database")
        self._persist_records(enriched)
        logger.info("Verifying persisted records")
        stored_count = self._verify_persistence(enriched)
        if self.convert_to_dataframe:
            logger.info("Converting archive to DataFrame representation")
            self._maybe_convert_dataframe()
        payload = dict(download_info)
        payload["file_path"] = archive_path
        payload["processed_entries"] = len(enriched)
        payload["records"] = stored_count
        logger.info("LiverTox update completed successfully")
        return payload

    # -----------------------------------------------------------------------------
    def _ensure_sources_dir(self) -> None:
        os.makedirs(self.sources_path, exist_ok=True)

    # -----------------------------------------------------------------------------
    def _normalize_archive_path(self) -> str:
        return os.path.join(self.sources_path, LIVERTOX_ARCHIVE)

    # -----------------------------------------------------------------------------
    def _collect_local_archive_info(self, archive_path: str) -> dict[str, Any]:
        if not os.path.isfile(archive_path):
            raise RuntimeError(
                "LiverTox archive not found; enable REDOWNLOAD to fetch a fresh copy."
            )
        size = os.path.getsize(archive_path)
        modified = datetime.fromtimestamp(os.path.getmtime(archive_path), UTC).isoformat()
        return {"file_path": archive_path, "size": size, "last_modified": modified}

    # -----------------------------------------------------------------------------
    def _download_archive(self) -> dict[str, Any]:
        try:
            return asyncio.run(self.livertox_client.download_bulk_data(self.sources_path))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to download LiverTox archive: {exc}") from exc

    # -----------------------------------------------------------------------------
    def _extract_monographs(self, archive_path: str) -> list[dict[str, Any]]:
        try:
            entries = self.livertox_client.collect_monographs(archive_path)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to extract LiverTox monographs: {exc}") from exc
        if not entries:
            raise RuntimeError("No LiverTox monographs were extracted from the archive.")
        return entries

    # -----------------------------------------------------------------------------
    def _sanitize_records(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sanitized = self.serializer.sanitize_livertox_records(entries)
        if sanitized.empty:
            raise RuntimeError("No valid LiverTox monographs were available after sanitization.")
        return sanitized.to_dict(orient="records")

    # -----------------------------------------------------------------------------
    def _enrich_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for entry in records:
            drug_name = entry.get("drug_name")
            if not isinstance(drug_name, str) or not drug_name.strip():
                entry["additional_names"] = None
                entry["synonyms"] = None
                continue
            try:
                names, synonyms = self.rx_client.fetch_drug_terms(drug_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to enrich '%s': %s", drug_name, exc)
                entry["additional_names"] = None
                entry["synonyms"] = None
                continue
            entry["additional_names"] = ", ".join(names) if names else None
            entry["synonyms"] = ", ".join(synonyms) if synonyms else None
        return records

    # -----------------------------------------------------------------------------
    def _persist_records(self, records: list[dict[str, Any]]) -> None:
        try:
            self.serializer.save_livertox_records(records)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to persist LiverTox records: {exc}") from exc

    # -----------------------------------------------------------------------------
    def _verify_persistence(self, records: list[dict[str, Any]]) -> int:
        try:
            stored_count = self.database.count_rows("LIVERTOX_MONOGRAPHS")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to verify stored LiverTox records: {exc}") from exc
        if stored_count != len(records):
            raise RuntimeError(
                "Mismatch between processed entries and stored rows; import verification failed."
            )
        return stored_count

    # -----------------------------------------------------------------------------
    def _maybe_convert_dataframe(self) -> None:
        try:
            self.livertox_client.convert_file_to_dataframe()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to convert archive to DataFrame: {exc}") from exc


# -----------------------------------------------------------------------------
def main() -> None:
    updater = LiverToxUpdater(
        SOURCES_PATH,
        redownload=REDOWNLOAD,
        convert_to_dataframe=CONVERT_TO_DATAFRAME,
    )
    try:
        result = updater.run()
    except Exception as exc:  # noqa: BLE001
        logger.error("LiverTox update failed: %s", exc)
        raise SystemExit(1) from exc
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
