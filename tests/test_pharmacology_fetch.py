from __future__ import annotations

import asyncio
import io
import os
import tarfile
from typing import Any

import pytest

from Pharmagent.app.constants import LIVERTOX_ARCHIVE
from Pharmagent.app.scripts.update_database import LiverToxUpdater
from Pharmagent.app.utils.jobs import _await_livertox_job


# -----------------------------------------------------------------------------
def _write_tar_archive(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        data = b"placeholder"
        info = tarfile.TarInfo(name="NBK1/sample.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


###############################################################################
class _StubFrame:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    @property
    def empty(self) -> bool:
        return not self._rows

    def to_dict(self, orient: str = "records") -> list[dict[str, Any]]:
        assert orient == "records"
        return list(self._rows)


###############################################################################
class _StubSerializer:
    def __init__(self) -> None:
        self.saved: list[dict[str, Any]] | None = None

    def sanitize_livertox_records(self, records: list[dict[str, Any]]) -> _StubFrame:
        return _StubFrame(records)

    def save_livertox_records(self, records: list[dict[str, Any]]) -> None:
        self.saved = list(records)


###############################################################################
class _StubRxNavClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def fetch_drug_terms(self, raw_name: str) -> tuple[list[str], list[str]]:
        self.queries.append(raw_name)
        return [f"{raw_name} name"], [f"{raw_name} synonym"]


###############################################################################
class _StubDatabase:
    def __init__(self, expected_count: int) -> None:
        self.expected_count = expected_count
        self.table: str | None = None

    def count_rows(self, table_name: str) -> int:
        self.table = table_name
        return self.expected_count


###############################################################################
class _StubLiverToxClient:
    def __init__(self, archive_path: str, *, create_on_download: bool) -> None:
        self.archive_path = archive_path
        self.create_on_download = create_on_download
        self.download_calls = 0
        self.collect_calls: list[str] = []
        self.convert_calls = 0

    async def download_bulk_data(self, dest_path: str) -> dict[str, Any]:
        self.download_calls += 1
        os.makedirs(dest_path, exist_ok=True)
        if self.create_on_download:
            _write_tar_archive(self.archive_path)
        size = os.path.getsize(self.archive_path)
        return {
            "file_path": self.archive_path,
            "size": size,
            "last_modified": "remote",
        }

    def collect_monographs(self, archive_path: str) -> list[dict[str, str]]:
        self.collect_calls.append(archive_path)
        return [
            {
                "nbk_id": "NBK1",
                "drug_name": "Sample",
                "excerpt": "Sample text",
            }
        ]

    def convert_file_to_dataframe(self):
        self.convert_calls += 1
        return []


# -----------------------------------------------------------------------------
def test_updater_uses_existing_archive(tmp_path) -> None:
    archive_path = tmp_path / LIVERTOX_ARCHIVE
    _write_tar_archive(str(archive_path))

    client = _StubLiverToxClient(str(archive_path), create_on_download=False)
    serializer = _StubSerializer()
    database = _StubDatabase(expected_count=1)

    updater = LiverToxUpdater(
        str(tmp_path),
        redownload=False,
        convert_to_dataframe=True,
        livertox_client=client,
        rx_client=_StubRxNavClient(),
        serializer=serializer,
        database_client=database,
    )

    result = updater.run()

    assert result["file_path"] == str(archive_path)
    assert result["records"] == 1
    assert result["processed_entries"] == 1
    assert client.download_calls == 0
    assert client.collect_calls == [str(archive_path)]
    assert client.convert_calls == 1
    assert serializer.saved is not None
    assert serializer.saved[0]["additional_names"] == "Sample name"
    assert serializer.saved[0]["synonyms"] == "Sample synonym"
    assert database.table == "LIVERTOX_MONOGRAPHS"


# -----------------------------------------------------------------------------
def test_updater_downloads_archive_when_requested(tmp_path) -> None:
    archive_path = tmp_path / LIVERTOX_ARCHIVE

    client = _StubLiverToxClient(str(archive_path), create_on_download=True)
    serializer = _StubSerializer()
    database = _StubDatabase(expected_count=1)

    updater = LiverToxUpdater(
        str(tmp_path),
        redownload=True,
        convert_to_dataframe=False,
        livertox_client=client,
        rx_client=_StubRxNavClient(),
        serializer=serializer,
        database_client=database,
    )

    result = updater.run()

    assert client.download_calls == 1
    assert os.path.isfile(result["file_path"])
    assert result["records"] == 1
    assert serializer.saved is not None


# -----------------------------------------------------------------------------
def test_updater_requires_local_archive_when_skip_download(tmp_path) -> None:
    archive_path = tmp_path / LIVERTOX_ARCHIVE

    client = _StubLiverToxClient(str(archive_path), create_on_download=False)
    serializer = _StubSerializer()
    database = _StubDatabase(expected_count=0)

    updater = LiverToxUpdater(
        str(tmp_path),
        redownload=False,
        convert_to_dataframe=False,
        livertox_client=client,
        rx_client=_StubRxNavClient(),
        serializer=serializer,
        database_client=database,
    )

    with pytest.raises(RuntimeError):
        updater.run()


###############################################################################
class _StubStatusResponse:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


###############################################################################
class _StubPollingClient:
    def __init__(self, payloads: list[dict[str, Any]]):
        self.payloads = payloads
        self.calls = 0

    async def get(self, url: str) -> _StubStatusResponse:
        idx = self.calls if self.calls < len(self.payloads) else len(self.payloads) - 1
        self.calls += 1
        await asyncio.sleep(0)
        return _StubStatusResponse(self.payloads[idx])


# -----------------------------------------------------------------------------
def test_await_livertox_job_handles_timeout() -> None:
    initial_status = {
        "job_id": "abc",
        "status": "running",
        "detail": "Job started",
        "result": None,
    }
    client = _StubPollingClient(
        [
            {"status": "running", "detail": "Still working"},
        ]
    )

    message = asyncio.run(
        _await_livertox_job(
            client,
            initial_status,
            poll_interval=0.01,
            timeout=0.05,
        )
    )

    assert isinstance(message, str)
    assert "[INFO]" in message
    assert "Job ID: abc" in message
    assert "Still working" in message


# -----------------------------------------------------------------------------
def test_await_livertox_job_completes() -> None:
    initial_status = {
        "job_id": "xyz",
        "status": "running",
        "detail": "Downloading",
        "result": None,
    }
    client = _StubPollingClient(
        [
            {"status": "running", "detail": "Extracting"},
            {
                "status": "completed",
                "detail": "Done",
                "result": {"file_path": "/tmp/archive.tar.gz"},
            },
        ]
    )

    result = asyncio.run(
        _await_livertox_job(
            client,
            initial_status,
            poll_interval=0.01,
            timeout=1.0,
        )
    )

    assert isinstance(result, tuple)
    payload, progress_log = result
    assert payload["file_path"] == "/tmp/archive.tar.gz"
    assert any("Extracting" in entry for entry in progress_log)
