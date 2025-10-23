from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from collections.abc import Iterator
from typing import Any
from urllib.parse import urljoin

import httpx
from tqdm import tqdm

from DILIGENT.app.constants import (
    OPENFDA_DOWNLOAD_BASE_URL,
    OPENFDA_DOWNLOAD_CATALOG_URL,
    OPENFDA_DRUG_EVENT_DATASET,
    OPENFDA_DRUG_EVENT_INDEX,
)
from DILIGENT.app.logger import logger
from DILIGENT.app.utils.repository.database import database
from DILIGENT.app.utils.repository.serializer import DataSerializer

__all__ = ["FdaUpdater"]

DEFAULT_HTTP_HEADERS = {
    "User-Agent": "DILIGENTClinicalCopilot/1.0 (contact=clinical-copilot@pharmagent.local)",
}

DOWNLOAD_CHUNK_SIZE = 262_144
METADATA_FILENAME = "fda-adverse-events.metadata.json"


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
        self.download_directory = os.path.join(sources_path, "fda")
        self.download_base_url = OPENFDA_DOWNLOAD_BASE_URL
        self.catalog_url = OPENFDA_DOWNLOAD_CATALOG_URL
        self.dataset_key = "event"
        self.dataset_category = "drug"
        self.dataset_url = os.path.join(
            OPENFDA_DOWNLOAD_BASE_URL,
            OPENFDA_DRUG_EVENT_DATASET,
        )
        self.dataset_base_url = f"{self.dataset_url.rstrip('/')}/"
        self.index_url = os.path.join(self.dataset_url, OPENFDA_DRUG_EVENT_INDEX)
        self.metadata_path = os.path.join(self.download_directory, METADATA_FILENAME)
        self.redownload = redownload
        self.serializer = serializer or DataSerializer()
        self.database = database_client
        self.chunk_size = chunk_size
        self.http_headers = dict(DEFAULT_HTTP_HEADERS)
        self.timeout = httpx.Timeout(120.0, connect=30.0)

    # -------------------------------------------------------------------------
    def update_from_fda(self) -> dict[str, Any]:
        os.makedirs(self.download_directory, exist_ok=True)
        metadata = {} if self.redownload else self.load_metadata()
        partitions_metadata = metadata.get("partitions", {})
        export_date = metadata.get("export_date")
        downloaded = 0
        records_processed = 0
        partitions_processed = 0
        current_export_date = export_date

        with httpx.Client(headers=self.http_headers, timeout=self.timeout) as client:
            catalog_payload = self.fetch_download_catalog(client)
            dataset_payload = self.resolve_dataset_entry(catalog_payload)
            if not dataset_payload:
                fallback_payload = self.fetch_index(client)
                dataset_payload = self.parse_index_payload(fallback_payload)
            partitions = self.get_partition_entries(dataset_payload)
            dataset_export_date = None
            if isinstance(dataset_payload, dict):
                dataset_export_date = dataset_payload.get(
                    "export_date"
                ) or dataset_payload.get("exportDate")
            if dataset_export_date and dataset_export_date != export_date:
                partitions_metadata = {}
            if dataset_export_date:
                current_export_date = dataset_export_date

            if not partitions:
                logger.warning(
                    "FDA download index did not provide any partitions for dataset %s",
                    self.dataset_key,
                )

            for partition in partitions:
                file_reference = self.get_partition_reference(partition)
                if not file_reference:
                    continue
                file_name = os.path.basename(file_reference)
                destination = os.path.join(self.download_directory, file_name)
                partition_metadata = self.build_partition_metadata(partition)
                should_download = self.should_download_partition(
                    destination,
                    partitions_metadata.get(file_name),
                    partition_metadata,
                )
                if should_download:
                    url = self.build_partition_url(partition)
                    if not url:
                        logger.warning(
                            "Skipping FDA partition %s because the download URL is missing",
                            file_name,
                        )
                        continue
                    logger.info("Downloading FDA partition %s", file_name)
                    success = self.download_partition(
                        client,
                        url,
                        destination,
                        partition_metadata.get("size"),
                        partition_metadata.get("sha256"),
                    )
                    if not success:
                        logger.error("Failed to download FDA partition %s", file_name)
                        continue
                    downloaded += 1
                partition_count = self.process_partition(destination)
                if partition_count is None:
                    continue
                records_processed += partition_count
                partitions_processed += 1
                partitions_metadata[file_name] = partition_metadata

        if records_processed == 0:
            logger.warning(
                "No FDA adverse event records were processed during the update"
            )

        updated_export_date = current_export_date

        metadata_payload = {
            "export_date": updated_export_date,
            "partitions": partitions_metadata,
        }
        self.save_metadata(metadata_payload)

        return {
            "records_processed": records_processed,
            "partitions_processed": partitions_processed,
            "partitions_downloaded": downloaded,
            "export_date": updated_export_date,
        }

    # -------------------------------------------------------------------------
    def process_partition(self, path: str) -> int | None:
        if not os.path.isfile(path):
            logger.warning("FDA partition %s is missing; skipping", path)
            return None
        records_processed = 0
        batch: list[dict[str, Any]] = []
        batch_limit = max(1, getattr(self.database, "insert_batch_size", 1000))
        try:
            for record in self.stream_partition_records(path):
                batch.append(record)
                if len(batch) >= batch_limit:
                    records_processed += self.persist_records(batch)
                    batch = []
            if batch:
                records_processed += self.persist_records(batch)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to process FDA partition %s: %s", path, exc)
            return None
        return records_processed

    # -------------------------------------------------------------------------
    def persist_records(self, records: list[dict[str, Any]]) -> int:
        sanitized = self.serializer.sanitize_fda_records(records)
        if sanitized.empty:
            return 0
        self.serializer.upsert_fda_records(sanitized)
        return int(len(sanitized.index))

    # -------------------------------------------------------------------------
    def fetch_download_catalog(self, client: httpx.Client) -> dict[str, Any]:
        try:
            response = client.get(self.catalog_url, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error(
                "Failed to retrieve FDA download catalog %s: %s",
                self.catalog_url,
                exc,
            )
            return {}
        try:
            payload = response.json()
        except ValueError as exc:
            logger.error(
                "Failed to decode FDA download catalog %s: %s",
                self.catalog_url,
                exc,
            )
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload

    # -------------------------------------------------------------------------
    def resolve_dataset_entry(self, catalog_payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(catalog_payload, dict):
            return {}
        results = catalog_payload.get("results")
        if not isinstance(results, dict):
            return {}
        category = results.get(self.dataset_category)
        if not isinstance(category, dict):
            return {}
        dataset_entry = category.get(self.dataset_key)
        if isinstance(dataset_entry, dict):
            return dataset_entry
        return {}

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
            logger.error(
                "Failed to decode FDA index response %s: %s", self.index_url, exc
            )
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload

    # -------------------------------------------------------------------------
    def parse_index_payload(self, index_payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(index_payload, dict):
            return {}
        results = index_payload.get("results")
        if isinstance(results, list) and results:
            first_entry = results[0]
            if isinstance(first_entry, dict):
                return first_entry
        return {}

    # -------------------------------------------------------------------------
    def get_partition_entries(
        self, dataset_payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if not isinstance(dataset_payload, dict):
            return []
        partitions = dataset_payload.get("partitions")
        if isinstance(partitions, list):
            return [
                partition for partition in partitions if isinstance(partition, dict)
            ]
        return []

    # -------------------------------------------------------------------------
    def get_partition_reference(self, partition: dict[str, Any]) -> str | None:
        for key in ("file", "path", "url"):
            value = partition.get(key)
            if isinstance(value, str):
                normalized = value.strip()
                if normalized:
                    return normalized
        return None

    # -------------------------------------------------------------------------
    def build_partition_metadata(self, partition: dict[str, Any]) -> dict[str, Any]:
        size_value = self.convert_to_int(partition.get("size"))
        records_value = self.convert_to_int(partition.get("records"))
        metadata = {
            "last_modified": partition.get("last_modified")
            or partition.get("lastModified"),
            "size": size_value,
            "records": records_value,
            "sha256": self.get_partition_checksum(partition),
        }
        return {key: value for key, value in metadata.items() if value is not None}

    # -------------------------------------------------------------------------
    def should_download_partition(
        self,
        destination: str,
        stored_metadata: dict[str, Any] | None,
        remote_metadata: dict[str, Any] | None,
    ) -> bool:
        if self.redownload:
            return True
        if not os.path.isfile(destination):
            return True
        if not remote_metadata:
            return False
        return not self.metadata_matches(stored_metadata, remote_metadata)

    # -------------------------------------------------------------------------
    def build_partition_url(self, partition: dict[str, Any]) -> str | None:
        reference = self.get_partition_reference(partition)
        if not reference:
            return None
        if reference.startswith("http://") or reference.startswith("https://"):
            return reference
        base_download = f"{self.download_base_url.rstrip('/')}/"
        if "/" in reference:
            return urljoin(base_download, reference)
        return urljoin(self.dataset_base_url, reference)

    # -------------------------------------------------------------------------
    def download_partition(
        self,
        client: httpx.Client,
        url: str,
        destination: str,
        total_size: int | None,
        expected_checksum: str | None,
    ) -> bool:
        temp_path = f"{destination}.download"
        digest = hashlib.sha256() if expected_checksum else None
        try:
            with client.stream("GET", url, follow_redirects=True) as response:
                response.raise_for_status()
                with open(temp_path, "wb") as output:
                    iterator = response.iter_bytes(chunk_size=self.chunk_size)
                    if total_size and total_size > 0:
                        with tqdm(
                            total=total_size,
                            unit="B",
                            unit_scale=True,
                            desc=f"FDA {os.path.basename(destination)}",
                            ncols=80,
                        ) as progress:
                            for chunk in iterator:
                                if not chunk:
                                    continue
                                output.write(chunk)
                                if digest:
                                    digest.update(chunk)
                                progress.update(len(chunk))
                    else:
                        for chunk in iterator:
                            if not chunk:
                                continue
                            output.write(chunk)
                            if digest:
                                digest.update(chunk)
        except httpx.HTTPError as exc:
            logger.error("Failed to download FDA partition %s: %s", url, exc)
            self.safe_remove(temp_path)
            return False
        except OSError as exc:
            logger.error("Failed to write FDA partition %s: %s", destination, exc)
            self.safe_remove(temp_path)
            return False

        if digest and not self.verify_checksum(digest, expected_checksum):
            logger.error("Checksum mismatch for FDA partition %s", destination)
            self.safe_remove(temp_path)
            return False

        try:
            os.replace(temp_path, destination)
        except OSError as exc:
            logger.error("Failed to finalize FDA partition %s: %s", destination, exc)
            self.safe_remove(temp_path)
            return False
        return True

    # -------------------------------------------------------------------------
    def verify_checksum(
        self,
        digest: Any,
        expected_checksum: str | None,
    ) -> bool:
        if not expected_checksum:
            return True
        normalized_expected = expected_checksum.strip().lower()
        if not normalized_expected:
            return True
        if normalized_expected.startswith("sha256:"):
            normalized_expected = normalized_expected.split(":", 1)[1].strip()
        computed = digest.hexdigest().lower()
        return computed == normalized_expected

    # -------------------------------------------------------------------------
    def safe_remove(self, path: str) -> None:
        if not path:
            return
        try:
            os.remove(path)
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.debug("Failed to remove temporary file %s: %s", path, exc)

    # -------------------------------------------------------------------------
    def convert_to_int(self, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    # -------------------------------------------------------------------------
    def get_partition_checksum(self, partition: dict[str, Any]) -> str | None:
        if not isinstance(partition, dict):
            return None
        candidates = [
            partition.get("sha256"),
            partition.get("sha_256"),
            partition.get("checksum"),
        ]
        for candidate in candidates:
            if isinstance(candidate, dict):
                for key in ("sha256", "sha_256", "value"):
                    value = candidate.get(key)
                    if isinstance(value, str):
                        normalized = value.strip()
                        if normalized:
                            return normalized
            elif isinstance(candidate, str):
                normalized = candidate.strip()
                if normalized:
                    return normalized
        return None

    # -------------------------------------------------------------------------
    def extract_records_from_payload(self, payload: Any) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            items = payload.get("results")
            if isinstance(items, list):
                records.extend([item for item in items if isinstance(item, dict)])
            else:
                records.append(payload)
        elif isinstance(payload, list):
            records.extend([item for item in payload if isinstance(item, dict)])
        return records

    # -------------------------------------------------------------------------
    def stream_partition_records(self, path: str) -> Iterator[dict[str, Any]]:
        if not os.path.isfile(path):
            return
        try:
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    yielded = False
                    with archive.open(member) as handle:
                        with io.TextIOWrapper(handle, encoding="utf-8") as text_stream:
                            for payload in self.iterate_results_stream(text_stream):
                                yielded = True
                                for record in self.extract_records_from_payload(
                                    payload
                                ):
                                    yield record
                    if yielded:
                        continue
                    with archive.open(member) as handle:
                        with io.TextIOWrapper(handle, encoding="utf-8") as text_stream:
                            for payload in self.iterate_ndjson_stream(text_stream):
                                for record in self.extract_records_from_payload(
                                    payload
                                ):
                                    yield record
        except (OSError, zipfile.BadZipFile) as exc:
            logger.error("Failed to read FDA partition %s: %s", path, exc)

    # -------------------------------------------------------------------------
    def iterate_results_stream(self, stream: io.TextIOWrapper) -> Iterator[Any]:
        decoder = json.JSONDecoder()
        buffer = ""
        results_found = False
        while True:
            chunk = stream.read(self.chunk_size)
            if not chunk:
                break
            buffer += chunk
            if not results_found:
                key_index = buffer.find('"results"')
                if key_index == -1:
                    if len(buffer) > 8192:
                        buffer = buffer[-8192:]
                    continue
                bracket_index = buffer.find("[", key_index)
                if bracket_index == -1:
                    continue
                buffer = buffer[bracket_index + 1 :]
                results_found = True
            while results_found:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer.startswith("]"):
                    buffer = buffer[1:]
                    results_found = False
                    break
                try:
                    payload, offset = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                yield payload
                buffer = buffer[offset:].lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]
        if not results_found:
            return
        buffer = buffer.lstrip()
        while buffer:
            if buffer.startswith("]"):
                break
            try:
                payload, offset = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                logger.debug("Incomplete JSON data at end of FDA results stream")
                break
            yield payload
            buffer = buffer[offset:].lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:]

    # -------------------------------------------------------------------------
    def iterate_ndjson_stream(self, stream: io.TextIOWrapper) -> Iterator[Any]:
        for line in stream:
            normalized = line.strip()
            if not normalized:
                continue
            try:
                yield json.loads(normalized)
            except json.JSONDecodeError:
                logger.debug(
                    "Skipping invalid NDJSON line in FDA partition: %s",
                    normalized[:200],
                )

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
        remote: dict[str, Any] | None,
    ) -> bool:
        if not stored or not remote:
            return False
        stored_size = self.convert_to_int(stored.get("size"))
        remote_size = self.convert_to_int(remote.get("size"))
        if remote_size is not None and stored_size != remote_size:
            return False
        stored_last_modified = (
            stored.get("last_modified", "").strip()
            if isinstance(stored.get("last_modified"), str)
            else stored.get("last_modified")
        )
        remote_last_modified = (
            remote.get("last_modified", "").strip()
            if isinstance(remote.get("last_modified"), str)
            else remote.get("last_modified")
        )
        if (
            stored_last_modified or remote_last_modified
        ) and stored_last_modified != remote_last_modified:
            return False
        stored_checksum = (
            stored.get("sha256", "").strip().lower()
            if isinstance(stored.get("sha256"), str)
            else None
        )
        remote_checksum = (
            remote.get("sha256", "").strip().lower()
            if isinstance(remote.get("sha256"), str)
            else None
        )
        if remote_checksum and stored_checksum != remote_checksum:
            return False
        return True
