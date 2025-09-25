import asyncio
import io
import os
import sys
import tarfile
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

if "pydantic" not in sys.modules:
    pydantic_stub = types.ModuleType("pydantic")

    class BaseModel:  # type: ignore[override]
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(default=None, *, default_factory=None, **kwargs):  # type: ignore[override]
        if default_factory is not None:
            return default_factory()
        return default

    def field_validator(*args, **kwargs):  # type: ignore[override]
        def decorator(func):
            return func

        return decorator

    def model_validator(*args, **kwargs):  # type: ignore[override]
        def decorator(func):
            return func

        return decorator

    pydantic_stub.BaseModel = BaseModel
    pydantic_stub.Field = Field
    pydantic_stub.field_validator = field_validator
    pydantic_stub.model_validator = model_validator
    sys.modules["pydantic"] = pydantic_stub

if "httpx" not in sys.modules:
    httpx_stub = types.ModuleType("httpx")

    class _StubResponse:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {}

        @property
        def text(self) -> str:
            return ""

    class _StubStream:
        async def __aenter__(self) -> "_StubStream":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self, chunk_size: int = 8192):
            if False:
                yield b""
            return

        async def aiter_lines(self):
            if False:
                yield ""
            return

    class _StubAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            return None

        async def __aenter__(self) -> "_StubAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def head(self, *args, **kwargs) -> _StubResponse:
            return _StubResponse()

        async def get(self, *args, **kwargs) -> _StubResponse:
            return _StubResponse()

        async def post(self, *args, **kwargs) -> _StubResponse:
            return _StubResponse()

        def stream(self, *args, **kwargs) -> _StubStream:
            return _StubStream()

        async def aclose(self) -> None:
            return None

    class _StubLimits:
        def __init__(self, *args, **kwargs) -> None:
            return None

    class _StubTimeout:
        def __init__(self, *args, **kwargs) -> None:
            return None

    class TimeoutException(Exception):
        pass

    class HTTPStatusError(Exception):
        pass

    class RequestError(Exception):
        pass

    httpx_stub.AsyncClient = _StubAsyncClient
    httpx_stub.Limits = _StubLimits
    httpx_stub.Timeout = _StubTimeout
    httpx_stub.TimeoutException = TimeoutException
    httpx_stub.HTTPStatusError = HTTPStatusError
    httpx_stub.RequestError = RequestError

    sys.modules["httpx"] = httpx_stub

if "Pharmagent.app.api.models.providers" not in sys.modules:
    providers_stub = types.ModuleType("Pharmagent.app.api.models.providers")

    class _StubLLMClient:
        async def llm_structured_call(self, *args, **kwargs):
            raise RuntimeError("LLM call not expected in tests")

    def initialize_llm_client(*args, **kwargs):  # type: ignore[override]
        return _StubLLMClient()

    providers_stub.initialize_llm_client = initialize_llm_client
    sys.modules["Pharmagent.app.api.models.providers"] = providers_stub

if "Pharmagent.app.logger" not in sys.modules:
    logger_stub = types.ModuleType("Pharmagent.app.logger")

    class _StubLogger:
        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

        def debug(self, *args, **kwargs):
            return None

    logger_stub.logger = _StubLogger()
    sys.modules["Pharmagent.app.logger"] = logger_stub

from Pharmagent.app.api.schemas.clinical import DrugEntry, PatientDrugs
from Pharmagent.app.utils.services.clinical import DrugToxicityEssay
from Pharmagent.app.utils.services.livertox import LiverToxMatcher
from Pharmagent.app.utils.services.scraper import LiverToxClient
from Pharmagent.app.constants import LIVERTOX_ARCHIVE


###############################################################################
class _DummyLLMClient:
    # -----------------------------------------------------------------------------
    async def llm_structured_call(self, *args, **kwargs):
        raise RuntimeError("LLM call not expected in tests")


# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _patch_llm_client(monkeypatch):
    monkeypatch.setattr(
        "Pharmagent.app.utils.services.clinical.initialize_llm_client",
        lambda *args, **kwargs: _DummyLLMClient(),
    )
    yield


# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _patch_serializer(monkeypatch):
    records = [
        {
            "nbk_id": "NBK100",
            "drug_name": "Acetaminophen (Tylenol)",
            "excerpt": "High doses cause liver injury.",
        },
        {
            "nbk_id": "NBK200",
            "drug_name": "Amoxicillin",
            "excerpt": "Rare hypersensitivity reactions.",
        },
    ]

    class _SerializerStub:
        def get_livertox_records(self):
            return pd.DataFrame(records)

    monkeypatch.setattr(
        "Pharmagent.app.utils.services.clinical.DataSerializer",
        lambda: _SerializerStub(),
    )
    yield


# -----------------------------------------------------------------------------
@pytest.fixture()
def sample_archive(tmp_path: Path) -> Path:
    archive_path = tmp_path / LIVERTOX_ARCHIVE
    html_one = """
<html><head><title>Acetaminophen (Tylenol)</title></head>
<body>
<p>Synonyms: Tylenol, Paracetamol.</p>
<h2>Hepatotoxicity</h2>
<p>High doses cause liver injury.</p>
<h2>Mechanism of Injury</h2>
<p>Metabolites form reactive intermediates.</p>
<h2>Outcome and Management</h2>
<p>Supportive care is recommended.</p>
</body></html>
"""
    html_two = """
<html><head><title>Amoxicillin</title></head>
<body>
<p>Synonyms: Amoxil.</p>
<h2>Hepatotoxicity</h2>
<p>Rare hypersensitivity reactions.</p>
<h2>Mechanism</h2>
<p>Likely immune mediated.</p>
</body></html>
"""
    with tarfile.open(archive_path, "w:gz") as tar:
        for name, html_text in {
            "NBK100.html": html_one,
            "NBK200.html": html_two,
        }.items():
            data = html_text.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return archive_path


# -----------------------------------------------------------------------------
def test_collect_monographs_from_archive(sample_archive: Path):
    client = LiverToxClient()
    entries = client.collect_monographs(str(sample_archive))
    assert len(entries) == 2
    nbk_ids = {entry["nbk_id"] for entry in entries}
    assert nbk_ids == {"NBK100", "NBK200"}
    names = {entry["drug_name"] for entry in entries}
    assert "Acetaminophen (Tylenol)" in names
    assert "Amoxicillin" in names
    assert all(entry.get("excerpt") for entry in entries)


# -----------------------------------------------------------------------------
def test_matcher_direct_match():
    df = pd.DataFrame(
        [
            {
                "nbk_id": "NBK100",
                "drug_name": "Acetaminophen (Tylenol)",
                "excerpt": "High doses cause liver injury.",
            },
            {
                "nbk_id": "NBK200",
                "drug_name": "Amoxicillin",
                "excerpt": "Rare hypersensitivity reactions.",
            },
        ]
    )
    matcher = LiverToxMatcher(df, llm_client=_DummyLLMClient())
    matches = asyncio.run(matcher.match_drug_names(["Acetaminophen"]))
    match = matches["Acetaminophen"]
    assert match is not None
    assert match.nbk_id == "NBK100"
    assert match.reason == "direct_match"


# -----------------------------------------------------------------------------
def test_matcher_alias_match():
    df = pd.DataFrame(
        [
            {
                "nbk_id": "NBK100",
                "drug_name": "Acetaminophen (Tylenol)",
                "excerpt": "High doses cause liver injury.",
            }
        ]
    )
    matcher = LiverToxMatcher(df, llm_client=_DummyLLMClient())
    matches = asyncio.run(matcher.match_drug_names(["Tylenol"]))
    match = matches["Tylenol"]
    assert match is not None
    assert match.nbk_id == "NBK100"
    assert match.reason in {"alias_match", "direct_match", "fuzzy_match"}


# -----------------------------------------------------------------------------
def test_essay_returns_mapping():
    drugs = PatientDrugs(entries=[DrugEntry(name="Acetaminophen"), DrugEntry(name="Unknown")])
    essay = DrugToxicityEssay(drugs)
    result = asyncio.run(essay.run_analysis())
    assert set(result) == {"Acetaminophen", "Unknown"}
    acetaminophen_row = result["Acetaminophen"]["matched_livertox_row"]
    assert acetaminophen_row is not None
    assert acetaminophen_row["nbk_id"] == "NBK100"
    assert result["Unknown"]["matched_livertox_row"] is None
    assert result["Unknown"]["extracted_excerpts"] == []


# -----------------------------------------------------------------------------
def test_essay_handles_empty_database(monkeypatch):
    class _EmptySerializer:
        def get_livertox_records(self):
            return pd.DataFrame()

    monkeypatch.setattr(
        "Pharmagent.app.utils.services.clinical.DataSerializer",
        lambda: _EmptySerializer(),
    )

    drugs = PatientDrugs(entries=[DrugEntry(name="Acetaminophen")])
    essay = DrugToxicityEssay(drugs)
    result = asyncio.run(essay.run_analysis())
    assert result["Acetaminophen"]["matched_livertox_row"] is None
