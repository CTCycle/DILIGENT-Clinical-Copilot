from __future__ import annotations

import io
import json
import os
import zipfile
from typing import Any

import httpx
from tqdm import tqdm

from DILIGENT.app.constants import (
    OPENFDA_DOWNLOAD_BASE_URL,
    OPENFDA_DRUGS_FDA_DATASET,
    OPENFDA_DRUGS_FDA_INDEX,
)
from DILIGENT.app.logger import logger
from DILIGENT.app.utils.repository.database import database
from DILIGENT.app.utils.repository.serializer import DataSerializer

__all__ = ["FdaUpdater"]

DEFAULT_HTTP_HEADERS = {
    "User-Agent": "DILIGENTClinicalCopilot/1.0 (contact=clinical-copilot@pharmagent.local)",
}

DOWNLOAD_CHUNK_SIZE = 262_144
METADATA_FILENAME = "drugsfda.metadata.json"


###############################################################################
class FdaUpdater:

    def __init__(
        self,
        sources_path: str,
        *,
        redownload: bool,
        serializer: DataSerializer | None = None,
        database_client=database,
        chunk_size: int = DOWNLOAD_CHUNK_SIZE,
    ) -> None:
        self.sources_path = os.path.abspath(sources_path)
        self.dataset_url = os.path.join(
            OPENFDA_DOWNLOAD_BASE_URL,
            OPENFDA_DRUGS_FDA_DATASET,
        )
        self.index_url = os.path.join(self.dataset_url, OPENFDA_DRUGS_FDA_INDEX)
        self.download_directory = os.path.join(self.sources_path, "fda")
        self.metadata_path = os.path.join(self.download_directory, METADATA_FILENAME)
        self.redownload = redownload
        self.serializer = serializer or DataSerializer()
        self.database = database_client
        self.chunk_size = chunk_size
        self.http_headers = dict(DEFAULT_HTTP_HEADERS)
        self.timeout = httpx.Timeout(120.0, connect=30.0)

    # -------------------------------------------------------------------------
    def update_from_drugsfda(self) -> dict[str, Any]:
        os.makedirs(self.download_directory, exist_ok=True)
        metadata = {} if self.redownload else self.load_metadata()
        partitions_metadata = metadata.get("partitions", {})
        export_date = metadata.get("export_date")
        downloaded = 0
        aggregated: list[dict[str, Any]] = []

        with httpx.Client(headers=self.http_headers, timeout=self.timeout) as client:
            index_payload = self.fetch_index(client)
            index_results = index_payload.get("results") if isinstance(index_payload, dict) else None
            latest_index = index_results[0] if index_results else {}
            current_export_date = latest_index.get("export_date")
            if current_export_date and current_export_date != export_date:
                partitions_metadata = {}
            partitions = latest_index.get("partitions") or []
            cleaned_partitions = [
                partition
                for partition in partitions
                if isinstance(partition, dict) and partition.get("file")
            ]
            for partition in cleaned_partitions:
                file_name = partition.get("file")
                destination = os.path.join(self.download_directory, file_name)
                should_download = (
                    self.redownload
                    or not self.metadata_matches(partitions_metadata.get(file_name, {}), partition)
                    or not os.path.isfile(destination)
                )
                if should_download:
                    logger.info("Downloading FDA partition %s", file_name)
                    self.download_partition(client, partition, destination)
                    downloaded += 1
                records = self.read_partition_records(destination)
                aggregated.extend(records)
                partitions_metadata[file_name] = {
                    "last_modified": partition.get("last_modified"),
                    "size": partition.get("size"),
                }

        if not aggregated:
            logger.warning("No FDA records retrieved from the bulk dataset")
            sanitized = self.serializer.sanitize_fda_records([])
            self.serializer.save_fda_records(sanitized)
            updated_export_date = current_export_date or export_date
        else:
            sanitized = self.serializer.sanitize_fda_records(aggregated)
            self.serializer.save_fda_records(sanitized)
            updated_export_date = current_export_date

        metadata_payload = {
            "export_date": updated_export_date,
            "partitions": partitions_metadata,
        }
        self.save_metadata(metadata_payload)

        return {
            "records_processed": int(len(sanitized.index)),
            "partitions_processed": len(partitions_metadata),
            "partitions_downloaded": downloaded,
            "export_date": updated_export_date,
        }

    # -------------------------------------------------------------------------
    def fetch_index(self, client: httpx.Client) -> dict[str, Any]:
        try:
            response = client.get(self.index_url, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Failed to retrieve FDA index %s: %s", self.index_url, exc)
            return {}
        try:
            payload = response.json()
        except ValueError as exc:
            logger.error("Failed to decode FDA index response %s: %s", self.index_url, exc)
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload

    # -------------------------------------------------------------------------
    def download_partition(
        self,
        client: httpx.Client,
        partition: dict[str, Any],
        destination: str,
    ) -> None:
        url = os.path.join(self.dataset_url, partition.get("file"))
        total_size = int(partition.get("size") or 0)
        with client.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            with open(destination, "wb") as output:
                if total_size > 0:
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"FDA {partition.get('file')}",
                        ncols=80,
                    ) as progress:
                        for chunk in response.iter_bytes(chunk_size=self.chunk_size):
                            if chunk:
                                output.write(chunk)
                                progress.update(len(chunk))
                else:
                    for chunk in response.iter_bytes(chunk_size=self.chunk_size):
                        if chunk:
                            output.write(chunk)

    # -------------------------------------------------------------------------
    def read_partition_records(self, path: str) -> list[dict[str, Any]]:
        if not os.path.isfile(path):
            return []
        try:
            with zipfile.ZipFile(path) as archive:
                records: list[dict[str, Any]] = []
                for member in archive.namelist():
                    with archive.open(member) as handle:
                        text_stream = io.TextIOWrapper(handle, encoding="utf-8")
                        payload = json.load(text_stream)
                        if isinstance(payload, dict):
                            items = payload.get("results")
                            if isinstance(items, list):
                                records.extend(
                                    [item for item in items if isinstance(item, dict)]
                                )
                        elif isinstance(payload, list):
                            records.extend(
                                [item for item in payload if isinstance(item, dict)]
                            )
                return records
        except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
            logger.error("Failed to read FDA partition %s: %s", path, exc)
            return []

    # -------------------------------------------------------------------------
    def load_metadata(self) -> dict[str, Any]:
        if not os.path.isfile(self.metadata_path):
            return {}
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
                if isinstance(payload, dict):
                    return payload
        except (OSError, json.JSONDecodeError):
            return {}
        return {}

    # -------------------------------------------------------------------------
    def save_metadata(self, payload: dict[str, Any]) -> None:
        with open(self.metadata_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    # -------------------------------------------------------------------------
    def metadata_matches(
        self,
        stored: dict[str, Any] | None,
        remote: dict[str, Any],
    ) -> bool:
        if not stored:
            return False
        stored_last_modified = stored.get("last_modified")
        remote_last_modified = remote.get("last_modified")
        stored_size = int(stored.get("size") or 0)
        remote_size = int(remote.get("size") or 0)
        return stored_last_modified == remote_last_modified and stored_size == remote_size
