import asyncio
import io
import os
import sys
import tarfile
import types

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

if "fastapi" not in sys.modules:
    fastapi_stub = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str | None = None) -> None:
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    def Query(default=None, **kwargs):  # type: ignore[override]
        return default

    class APIRouter:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            return None

        def get(self, *args, **kwargs):  # type: ignore[override]
            def decorator(func):
                return func

            return decorator

    fastapi_stub.HTTPException = HTTPException
    fastapi_stub.Query = Query
    fastapi_stub.APIRouter = APIRouter
    sys.modules["fastapi"] = fastapi_stub

from fastapi import HTTPException

from Pharmagent.app.constants import LIVERTOX_ARCHIVE


###############################################################################
class _StubLiverToxClient:
    def __init__(self, archive_path: str, *, create_on_download: bool) -> None:
        self.archive_path = archive_path
        self.create_on_download = create_on_download
        self.download_called = False
        self.collected_paths: list[str] = []

    async def download_bulk_data(self, dest_path: str) -> dict[str, str | int | None]:
        self.download_called = True
        if self.create_on_download:
            os.makedirs(dest_path, exist_ok=True)
            _write_tar_archive(self.archive_path)
        size = os.path.getsize(self.archive_path)
        return {
            "file_path": self.archive_path,
            "size": size,
            "last_modified": "remote",
        }

    def collect_monographs(self, archive_path: str) -> list[dict[str, str]]:
        self.collected_paths.append(archive_path)
        return [
            {
                "nbk_id": "NBK1",
                "drug_name": "Sample",
                "excerpt": "Sample text",
                "text": "Sample text",
            }
        ]

    def convert_file_to_dataframe(self):
        raise AssertionError("convert_file_to_dataframe should not be called in tests")


###############################################################################
class _StubSerializer:
    def __init__(self) -> None:
        self.saved_records: list[dict[str, str]] | None = None

    def save_livertox_records(self, records: list[dict[str, str]]) -> None:
        self.saved_records = records


###############################################################################
class _StubDatabase:
    def __init__(self, expected_count: int) -> None:
        self.expected_count = expected_count
        self.table: str | None = None

    def count_rows(self, table_name: str) -> int:
        self.table = table_name
        return self.expected_count


# -----------------------------------------------------------------------------
def _write_tar_archive(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        data = b"placeholder"
        info = tarfile.TarInfo(name="NBK1/sample.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


# -----------------------------------------------------------------------------
def test_fetch_bulk_livertox_skip_download(monkeypatch, tmp_path) -> None:
    from Pharmagent.app.api.endpoints import pharmacology as module

    archive_path = tmp_path / LIVERTOX_ARCHIVE
    _write_tar_archive(str(archive_path))

    client = _StubLiverToxClient(str(archive_path), create_on_download=False)
    serializer = _StubSerializer()
    database = _StubDatabase(expected_count=1)

    monkeypatch.setattr(module, "LT_client", client)
    monkeypatch.setattr(module, "serializer", serializer)
    monkeypatch.setattr(module, "database", database)
    monkeypatch.setattr(module, "SOURCES_PATH", str(tmp_path))

    result = asyncio.run(module.fetch_bulk_livertox(False, True))

    assert client.download_called is False
    assert client.collected_paths == [str(archive_path)]
    assert result["file_path"] == str(archive_path)
    assert result["processed_entries"] == 1
    assert serializer.saved_records is not None


# -----------------------------------------------------------------------------
def test_fetch_bulk_livertox_triggers_download(monkeypatch, tmp_path) -> None:
    from Pharmagent.app.api.endpoints import pharmacology as module

    archive_path = tmp_path / LIVERTOX_ARCHIVE

    client = _StubLiverToxClient(str(archive_path), create_on_download=True)
    serializer = _StubSerializer()
    database = _StubDatabase(expected_count=1)

    monkeypatch.setattr(module, "LT_client", client)
    monkeypatch.setattr(module, "serializer", serializer)
    monkeypatch.setattr(module, "database", database)
    monkeypatch.setattr(module, "SOURCES_PATH", str(tmp_path))

    result = asyncio.run(module.fetch_bulk_livertox(False, False))

    assert client.download_called is True
    assert client.collected_paths == [str(archive_path)]
    assert os.path.isfile(result["file_path"])
    assert result["processed_entries"] == 1
    assert serializer.saved_records is not None


# -----------------------------------------------------------------------------
def test_fetch_bulk_livertox_skip_without_archive(monkeypatch, tmp_path) -> None:
    from Pharmagent.app.api.endpoints import pharmacology as module

    client = _StubLiverToxClient(os.path.join(str(tmp_path), LIVERTOX_ARCHIVE), create_on_download=False)
    serializer = _StubSerializer()
    database = _StubDatabase(expected_count=0)

    monkeypatch.setattr(module, "LT_client", client)
    monkeypatch.setattr(module, "serializer", serializer)
    monkeypatch.setattr(module, "database", database)
    monkeypatch.setattr(module, "SOURCES_PATH", str(tmp_path))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(module.fetch_bulk_livertox(False, True))

    assert exc.value.status_code == 404
    assert client.download_called is False
    assert serializer.saved_records is None
