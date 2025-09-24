import asyncio
import io
import os
import sys
import tarfile
import types
from typing import Any

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

if "pandas" not in sys.modules:
    pandas_stub = types.ModuleType("pandas")

    class _StubDataFrame:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("DataFrame should not be instantiated in tests")

    def _stub_concat(*args, **kwargs):  # type: ignore[override]
        raise AssertionError("pandas.concat should not be called in tests")

    pandas_stub.DataFrame = _StubDataFrame  # type: ignore[attr-defined]
    pandas_stub.concat = _stub_concat  # type: ignore[attr-defined]
    sys.modules["pandas"] = pandas_stub

if "sqlalchemy" not in sys.modules:
    sqlalchemy_stub = types.ModuleType("sqlalchemy")

    class _UniqueConstraint:
        def __init__(self, *columns) -> None:
            self.columns = types.SimpleNamespace(keys=lambda: list(columns))

    class _Result:
        def scalar(self) -> int:
            return 0

    class _Connection:
        def execute(self, *args, **kwargs):  # type: ignore[override]
            return _Result()

        def __enter__(self):  # type: ignore[override]
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[override]
            return False

    class _BeginContext:
        def __enter__(self) -> _Connection:  # type: ignore[override]
            return _Connection()

        def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[override]
            return False

    def _create_engine(*args, **kwargs):  # type: ignore[override]
        return types.SimpleNamespace(begin=lambda: _BeginContext(), connect=lambda: _Connection())

    def _text(query):  # type: ignore[override]
        return query

    sqlalchemy_stub.Column = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sqlalchemy_stub.Float = float  # type: ignore[attr-defined]
    sqlalchemy_stub.String = str  # type: ignore[attr-defined]
    sqlalchemy_stub.Text = str  # type: ignore[attr-defined]
    sqlalchemy_stub.UniqueConstraint = _UniqueConstraint  # type: ignore[attr-defined]
    sqlalchemy_stub.create_engine = _create_engine  # type: ignore[attr-defined]
    sqlalchemy_stub.text = _text  # type: ignore[attr-defined]
    sys.modules["sqlalchemy"] = sqlalchemy_stub

    orm_stub = types.ModuleType("sqlalchemy.orm")

    class _Session:
        def execute(self, *args, **kwargs):  # type: ignore[override]
            return None

        def commit(self) -> None:
            return None

        def close(self) -> None:
            return None

    class _SessionFactory:
        def __call__(self, *args, **kwargs):  # type: ignore[override]
            return _Session()

    def _sessionmaker(*args, **kwargs):  # type: ignore[override]
        return _SessionFactory()

    def _declarative_base():  # type: ignore[override]
        class _Base:
            metadata = types.SimpleNamespace(create_all=lambda *a, **k: None)
            _registry: list[type] = []

            def __init_subclass__(cls, **kwargs):  # type: ignore[override]
                super().__init_subclass__(**kwargs)
                if cls is not _Base:
                    _Base._registry.append(cls)

            @classmethod
            def __subclasses__(cls):  # type: ignore[override]
                return list(_Base._registry)

        return _Base

    orm_stub.declarative_base = _declarative_base  # type: ignore[attr-defined]
    orm_stub.sessionmaker = _sessionmaker  # type: ignore[attr-defined]
    sys.modules["sqlalchemy.orm"] = orm_stub

    dialects_stub = types.ModuleType("sqlalchemy.dialects")
    sqlite_stub = types.ModuleType("sqlalchemy.dialects.sqlite")

    class _Insert:
        def values(self, *args, **kwargs):  # type: ignore[override]
            return self

        def on_conflict_do_update(self, **kwargs):  # type: ignore[override]
            return self

        def __getattr__(self, name):  # type: ignore[override]
            return self

    def _insert(table):  # type: ignore[override]
        return _Insert()

    sqlite_stub.insert = _insert  # type: ignore[attr-defined]
    dialects_stub.sqlite = sqlite_stub  # type: ignore[attr-defined]
    sys.modules["sqlalchemy.dialects"] = dialects_stub
    sys.modules["sqlalchemy.dialects.sqlite"] = sqlite_stub

if "httpx" not in sys.modules:
    httpx_stub = types.ModuleType("httpx")

    class _AsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            return None

        async def __aenter__(self):  # type: ignore[override]
            raise AssertionError("httpx.AsyncClient should not be used in tests")

        async def __aexit__(self, exc_type, exc, tb):  # type: ignore[override]
            return False

    httpx_stub.AsyncClient = _AsyncClient  # type: ignore[attr-defined]
    sys.modules["httpx"] = httpx_stub


if "pdfminer" not in sys.modules:
    pdfminer_stub = types.ModuleType("pdfminer")
    pdfminer_high_level = types.ModuleType("pdfminer.high_level")

    def _extract_text(*args, **kwargs):  # type: ignore[override]
        raise AssertionError("pdfminer should not be used in tests")

    pdfminer_high_level.extract_text = _extract_text  # type: ignore[attr-defined]
    sys.modules["pdfminer"] = pdfminer_stub
    sys.modules["pdfminer.high_level"] = pdfminer_high_level

if "pypdf" not in sys.modules:
    pypdf_stub = types.ModuleType("pypdf")

    class _PdfReader:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("PdfReader should not be instantiated in tests")

    pypdf_stub.PdfReader = _PdfReader  # type: ignore[attr-defined]
    sys.modules["pypdf"] = pypdf_stub

if "tqdm" not in sys.modules:
    tqdm_stub = types.ModuleType("tqdm")

    class _Tqdm:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def __enter__(self):  # type: ignore[override]
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[override]
            return False

        def update(self, *args, **kwargs):  # type: ignore[override]
            return None

    def _tqdm(*args, **kwargs):  # type: ignore[override]
        return _Tqdm()

    tqdm_stub.tqdm = _tqdm  # type: ignore[attr-defined]
    sys.modules["tqdm"] = tqdm_stub

from fastapi import HTTPException

from Pharmagent.app.constants import LIVERTOX_ARCHIVE
from Pharmagent.app.utils.jobs import JobManager, JobStatus
from Pharmagent.app.client.livertox_jobs import _await_livertox_job


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
async def _run_fetch(module, convert_to_dataframe: bool, skip_download: bool):
    initial_status = await module.fetch_bulk_livertox(convert_to_dataframe, skip_download)
    job_id = initial_status["job_id"]
    final_status = await module.job_manager.wait_for_completion(job_id)
    return initial_status, final_status


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
    monkeypatch.setattr(module, "job_manager", JobManager())

    initial, final = asyncio.run(_run_fetch(module, False, True))

    assert initial["status"] in {
        JobStatus.QUEUED.value,
        JobStatus.RUNNING.value,
    }
    assert final is not None
    assert final["status"] == JobStatus.COMPLETED.value
    result = final["result"]
    assert isinstance(result, dict)
    assert result["file_path"] == str(archive_path)
    assert result["processed_entries"] == 1
    assert result["records"] == 1
    assert client.download_called is False
    assert client.collected_paths == [str(archive_path)]
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
    monkeypatch.setattr(module, "job_manager", JobManager())

    initial, final = asyncio.run(_run_fetch(module, False, False))

    assert initial["status"] in {
        JobStatus.QUEUED.value,
        JobStatus.RUNNING.value,
    }
    assert final is not None
    assert final["status"] == JobStatus.COMPLETED.value
    result = final["result"]
    assert isinstance(result, dict)
    assert os.path.isfile(result["file_path"])
    assert result["processed_entries"] == 1
    assert result["records"] == 1
    assert client.download_called is True
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
    monkeypatch.setattr(module, "job_manager", JobManager())

    with pytest.raises(HTTPException) as exc:
        asyncio.run(module.fetch_bulk_livertox(False, True))

    assert exc.value.status_code == 404
    assert client.download_called is False
    assert serializer.saved_records is None


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
    client = _StubPollingClient([
        {"status": "running", "detail": "Still working"},
    ])

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
