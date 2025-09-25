import asyncio
import io
import os
import sys
import tarfile
import types
from pathlib import Path
from typing import Any

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
            self.status_code = 200

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
    httpx_stub.get = lambda *args, **kwargs: _StubResponse()

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

from Pharmagent.app.api.schemas.clinical import (
    DrugEntry,
    LiverToxBatchMatchItem,
    LiverToxBatchMatchSuggestion,
    PatientDrugs,
)
from Pharmagent.app.utils.services.clinical import DrugToxicityEssay
from Pharmagent.app.utils.services.livertox import LiverToxMatcher
from Pharmagent.app.utils.services.scraper import LiverToxClient
from Pharmagent.app.constants import LIVERTOX_ARCHIVE


###############################################################################
class _DummyLLMClient:
    # -----------------------------------------------------------------------------
    async def llm_structured_call(self, *args, **kwargs):
        raise RuntimeError("LLM call not expected in tests")


###############################################################################
class _LLMBatchStub:
    def __init__(self, suggestion: LiverToxBatchMatchSuggestion) -> None:
        self.suggestion = suggestion
        self.calls: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    async def llm_structured_call(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.suggestion


# -----------------------------------------------------------------------------
class _RxNormStub:
    def __init__(self, mapping: dict[str, dict[str, str]] | None = None) -> None:
        self.mapping = mapping or {}

    # -------------------------------------------------------------------------
    def expand(self, raw_name: str) -> set[str]:
        key = raw_name.lower()
        entries = self.mapping.get(key)
        if entries is None:
            return {key}
        return set(entries)

    # -------------------------------------------------------------------------
    def get_candidate_kind(self, original: str, candidate: str) -> str:
        key = original.lower()
        entries = self.mapping.get(key)
        if not entries:
            return "unknown"
        return entries.get(candidate, "unknown")


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
def _disable_rxnorm(monkeypatch):
    monkeypatch.setattr(
        "Pharmagent.app.utils.services.retrieval.RXNORM_EXPANSION_ENABLED",
        False,
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
    assert len(matches) == 1
    match = matches[0]
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
    assert len(matches) == 1
    match = matches[0]
    assert match is not None
    assert match.nbk_id == "NBK100"
    assert match.reason in {"alias_match", "direct_match", "fuzzy_match"}


# -----------------------------------------------------------------------------
def test_matcher_preserves_order_and_length():
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
    inputs = ["Tylenol", "Tylenol", "Amoxicillin"]
    matches = asyncio.run(matcher.match_drug_names(inputs))
    assert len(matches) == len(inputs)
    assert [match.nbk_id if match else None for match in matches] == [
        "NBK100",
        "NBK100",
        "NBK200",
    ]


# -----------------------------------------------------------------------------
def test_matcher_uses_rxnorm_brand_to_match(monkeypatch):
    monkeypatch.setattr(
        "Pharmagent.app.utils.services.retrieval.RXNORM_EXPANSION_ENABLED",
        True,
    )
    df = pd.DataFrame(
        [
            {
                "nbk_id": "NBK300",
                "drug_name": "Duloxetine",
                "excerpt": "SNRI",
            }
        ]
    )
    retriever = _RxNormStub(
        {
            "cymbalta": {"cymbalta": "brand", "duloxetine": "ingredient"},
            "duloxetine": {"duloxetine": "ingredient", "cymbalta": "brand"},
        }
    )
    matcher = LiverToxMatcher(df, llm_client=_DummyLLMClient(), retriever=retriever)
    matches = asyncio.run(matcher.match_drug_names(["Cymbalta", "Duloxetine"]))
    assert [match.nbk_id if match else None for match in matches] == [
        "NBK300",
        "NBK300",
    ]


# -----------------------------------------------------------------------------
def test_matcher_prefers_single_ingredient_over_combo(monkeypatch):
    monkeypatch.setattr(
        "Pharmagent.app.utils.services.retrieval.RXNORM_EXPANSION_ENABLED",
        True,
    )
    df = pd.DataFrame(
        [
            {
                "nbk_id": "NBK400",
                "drug_name": "Amlodipine",
                "excerpt": "Calcium channel blocker",
            },
            {
                "nbk_id": "NBK401",
                "drug_name": "Atorvastatin",
                "excerpt": "Statin",
            },
            {
                "nbk_id": "NBK402",
                "drug_name": "Amlodipine Atorvastatin",
                "excerpt": "Combo",
            },
        ]
    )
    retriever = _RxNormStub(
        {
            "caduet": {
                "amlodipine": "ingredient",
                "atorvastatin": "ingredient",
                "amlodipine / atorvastatin": "ingredient_combo",
            }
        }
    )
    matcher = LiverToxMatcher(df, llm_client=_DummyLLMClient(), retriever=retriever)
    matches = asyncio.run(matcher.match_drug_names(["Caduet"]))
    assert matches[0] is not None
    assert matches[0].nbk_id in {"NBK400", "NBK401"}
    assert matches[0].reason == "direct_match"

# -----------------------------------------------------------------------------
def test_essay_returns_mapping():
    drugs = PatientDrugs(entries=[DrugEntry(name="Acetaminophen"), DrugEntry(name="Unknown")])
    essay = DrugToxicityEssay(drugs)
    result = asyncio.run(essay.run_analysis())
    assert [entry["drug_name"] for entry in result] == ["Acetaminophen", "Unknown"]
    acetaminophen_row = result[0]["matched_livertox_row"]
    assert acetaminophen_row is not None
    assert acetaminophen_row["nbk_id"] == "NBK100"
    assert result[1]["matched_livertox_row"] is None
    assert result[1]["extracted_excerpts"] == []


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
    assert result[0]["matched_livertox_row"] is None


# -----------------------------------------------------------------------------
def test_llm_fallback_handles_reordered_matches():
    df = pd.DataFrame(
        [
            {
                "nbk_id": "NBK001",
                "drug_name": "Acetylsalicylic Acid",
                "excerpt": "Classic analgesic.",
            },
            {
                "nbk_id": "NBK002",
                "drug_name": "Rivaroxaban",
                "excerpt": "Anticoagulant.",
            },
            {
                "nbk_id": "NBK003",
                "drug_name": "Pregabalin",
                "excerpt": "Neuropathic pain treatment.",
            },
        ]
    )
    suggestion = LiverToxBatchMatchSuggestion(
        matches=[
            LiverToxBatchMatchItem(
                drug_name="Lyrica",
                match_name="Pregabalin",
                confidence=0.61,
                rationale="Brand matched to generic.",
            ),
            LiverToxBatchMatchItem(
                drug_name="Aspirin",
                match_name="Acetylsalicylic Acid",
                confidence=0.72,
            ),
            LiverToxBatchMatchItem(
                drug_name="Xarelto",
                match_name="Rivaroxaban",
                confidence=0.68,
            ),
        ]
    )
    llm_stub = _LLMBatchStub(suggestion)
    matcher = LiverToxMatcher(df, llm_client=llm_stub)
    inputs = ["aspirin", "xarelto", "lyrica"]
    results = asyncio.run(matcher.match_drug_names(inputs))
    assert [match.nbk_id if match else None for match in results] == [
        "NBK001",
        "NBK002",
        "NBK003",
    ]
    assert all(match and match.reason == "llm_fallback" for match in results if match)
    assert len(llm_stub.calls) == 1


# -----------------------------------------------------------------------------
def test_llm_fallback_ignores_no_match_entries():
    df = pd.DataFrame(
        [
            {
                "nbk_id": "NBK010",
                "drug_name": "Acetylsalicylic Acid",
                "excerpt": "Classic analgesic.",
            },
            {
                "nbk_id": "NBK020",
                "drug_name": "Rivaroxaban",
                "excerpt": "Anticoagulant.",
            },
        ]
    )
    suggestion = LiverToxBatchMatchSuggestion(
        matches=[
            LiverToxBatchMatchItem(
                drug_name="Aspirin",
                match_name="Acetylsalicylic Acid",
                confidence=0.7,
            ),
            LiverToxBatchMatchItem(
                drug_name="Lyrica",
                match_name="No match",
                confidence=0.1,
            ),
        ]
    )
    llm_stub = _LLMBatchStub(suggestion)
    matcher = LiverToxMatcher(df, llm_client=llm_stub)
    inputs = ["aspirin", "lyrica", "xarelto"]
    results = asyncio.run(matcher.match_drug_names(inputs))
    assert [match.nbk_id if match else None for match in results] == [
        "NBK010",
        None,
        None,
    ]
    assert len(llm_stub.calls) == 1
